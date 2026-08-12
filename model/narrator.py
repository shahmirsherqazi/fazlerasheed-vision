"""
narrator.py — Stage 5: Per-person periodic activity narration via Azure OpenAI
===============================================================================
Implements the cloud layer described in person-activity-tracker-plan.md.

Behaviour
---------
  - Accepts a list of TrackResult objects each pipeline iteration.
  - For each tracked person, if NARRATE_INTERVAL_S seconds have elapsed since
    that person's last narration, crop their bounding box from the frame and
    send it to the configured vision LLM.
  - The LLM returns a short activity label (standing / sitting / bending /
    drinking / eating / talking / opening a fridge / opening a drawer).
  - The result is logged with timestamp + person_id.
  - HIGH-PRIORITY ALERT: if the description contains drawer/fridge keywords,
    an immediate alert is logged and printed.
  - On demand (key 'R'), generate a human-readable shift summary.

LLM providers (set LLM_PROVIDER env var):
  "azure"    — Azure OpenAI GPT-4o (primary, per the plan)
  "gemini"   — Google Gemini Flash (fallback)
  "openai"   — OpenAI GPT-4o-mini direct (fallback)
  "disabled" — Stage 5 completely off

Azure env vars required when LLM_PROVIDER=azure:
  AZURE_OPENAI_API_KEY      — your Azure API key
  AZURE_OPENAI_ENDPOINT     — e.g. https://<resource>.openai.azure.com/
  AZURE_OPENAI_DEPLOYMENT   — your GPT-4o deployment name (e.g. "gpt-4o")
  AZURE_OPENAI_API_VERSION  — e.g. "2024-02-01"  (default provided)

Privacy note:
  Set BLUR_FACES_BEFORE_SEND=true to anonymise faces in crops before they
  leave the device.  Off by default for prototype.
"""

import os
import base64
import time
import threading
import re

import cv2
import numpy as np

# Load .env before reading any config values
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

from logger import EventLogger

# ── Config ────────────────────────────────────────────────
LLM_PROVIDER          = os.getenv("LLM_PROVIDER", "azure")
GEMINI_API_KEY        = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY        = os.getenv("OPENAI_API_KEY", "")

AZURE_OPENAI_API_KEY    = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT   = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-11-20")

NARRATE_INTERVAL_S     = int(os.getenv("NARRATE_INTERVAL_S", "3"))  # per-person interval
BOX_PAD_PX             = int(os.getenv("BOX_PAD_PX", "20"))         # padding around crop
BLUR_FACES_BEFORE_SEND = os.getenv("BLUR_FACES_BEFORE_SEND", "false").lower() == "true"

# Keywords that trigger a high-priority alert
HIGH_PRIORITY_KEYWORDS = [
    "drawer", "drawers", "fridge", "refrigerator",
    "cabinet", "cabinets", "cupboard",
]

# ── Prompts ───────────────────────────────────────────────
ACTIVITY_PROMPT = (
    "You are a security monitoring assistant analysing a cropped image of one person. "
    "In a single short phrase (5 words or fewer), describe the most likely action this person is performing. "
    "Choose the best match from this list if applicable: "
    "standing, sitting, bending, drinking, eating, talking, opening a fridge, opening a drawer. "
    "If none fit, give your own brief description. "
    "Do NOT identify the person by name. Do NOT add extra commentary."
)

SUMMARY_PROMPT_TEMPLATE = (
    "You are generating an end-of-shift activity report for a security system. "
    "Below are timestamped per-person activity descriptions collected over a period. "
    "Write a concise, professional plain-English summary (3–5 sentences). "
    "Note any notable events, patterns, or high-priority alerts.\n\n"
    "Activity log:\n{narrations}\n\n"
    "Event counts:\n{event_counts}"
)


# ── Frame helpers ─────────────────────────────────────────

def _crop_box(frame: np.ndarray, box: tuple, pad: int = BOX_PAD_PX) -> np.ndarray:
    """Crop a padded region from frame given (x1,y1,x2,y2)."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    return frame[y1:y2, x1:x2]


def _blur_faces(frame: np.ndarray) -> np.ndarray:
    """Blur detected faces before sending to external API."""
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascade_path)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
        out = frame.copy()
        for (x, y, fw, fh) in faces:
            roi = out[y:y+fh, x:x+fw]
            out[y:y+fh, x:x+fw] = cv2.GaussianBlur(roi, (51, 51), 0)
        return out
    except Exception as e:
        print(f"[Narrator] Face blur failed ({e}) — sending unblurred crop.")
        return frame


def _frame_to_b64_jpeg(frame: np.ndarray, quality: int = 75) -> str:
    """Encode a BGR numpy frame as a base64 JPEG string."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _has_high_priority_keyword(text: str) -> str | None:
    """Return the matched keyword if text contains a high-priority word, else None."""
    lower = text.lower()
    for kw in HIGH_PRIORITY_KEYWORDS:
        if re.search(r'\b' + re.escape(kw) + r'\b', lower):
            return kw
    return None


# ── LLM clients ───────────────────────────────────────────

def _call_azure(b64_image: str) -> str:
    """Send crop to Azure OpenAI GPT-4o vision."""
    try:
        from openai import AzureOpenAI
        client = AzureOpenAI(
            api_key        = AZURE_OPENAI_API_KEY,
            azure_endpoint = AZURE_OPENAI_ENDPOINT,
            api_version    = AZURE_OPENAI_API_VERSION,
        )
        response = client.chat.completions.create(
            model    = AZURE_OPENAI_DEPLOYMENT,
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": ACTIVITY_PROMPT},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{b64_image}",
                        "detail": "low",
                    }},
                ],
            }],
            max_tokens = 60,
        )
        return response.choices[0].message.content.strip()
    except ImportError:
        return "[Narrator] openai package not installed. Run: pip install openai"
    except Exception as e:
        return f"[Narrator] Azure API error: {e}"


_GEMINI_VISION_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash-latest",
    "gemini-1.5-flash",
]

_GEMINI_TEXT_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash-latest",
    "gemini-1.5-flash",
]


def _gemini_generate(model_names: list, contents) -> str:
    import google.generativeai as genai
    last_err = None
    for model_name in model_names:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(contents)
            print(f"[Narrator] Used model: {model_name}")
            return response.text.strip()
        except Exception as e:
            last_err = e
            if "404" in str(e) or "not found" in str(e).lower():
                continue
            break
    return f"[Narrator] Gemini error: {last_err}"


def _call_gemini(b64_image: str) -> str:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        image_part = {"mime_type": "image/jpeg", "data": b64_image}
        return _gemini_generate(_GEMINI_VISION_MODELS, [ACTIVITY_PROMPT, image_part])
    except ImportError:
        return "[Narrator] google-generativeai not installed. Run: pip install google-generativeai"


def _call_openai(b64_image: str) -> str:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model    = "gpt-4o-mini",
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": ACTIVITY_PROMPT},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{b64_image}",
                        "detail": "low",
                    }},
                ],
            }],
            max_tokens = 60,
        )
        return response.choices[0].message.content.strip()
    except ImportError:
        return "[Narrator] openai not installed. Run: pip install openai"
    except Exception as e:
        return f"[Narrator] OpenAI error: {e}"


def _describe_crop(crop: np.ndarray) -> str:
    """Route crop to the configured LLM provider."""
    if LLM_PROVIDER == "disabled":
        return ""

    send = _blur_faces(crop) if BLUR_FACES_BEFORE_SEND else crop
    b64  = _frame_to_b64_jpeg(send)

    if LLM_PROVIDER == "azure":
        return _call_azure(b64)
    elif LLM_PROVIDER == "gemini":
        return _call_gemini(b64)
    elif LLM_PROVIDER == "openai":
        return _call_openai(b64)
    else:
        return f"[Narrator] Unknown provider: {LLM_PROVIDER}"


# ── Main narrator class ───────────────────────────────────

class Narrator:
    """
    Per-person periodic activity narration.

    Call `maybe_narrate(frame, tracks)` every pipeline iteration.
    It maintains a per-track-ID timer and fires async LLM calls only when
    each person's interval has elapsed.

    Parameters
    ----------
    logger : EventLogger
        Shared event log instance.
    """

    def __init__(self, logger: EventLogger):
        self._log            = logger
        self._last_narrate   = {}   # track_id -> last narration timestamp
        self._lock           = threading.Lock()
        self._active         = True

        self._check_config()

    def _check_config(self):
        """Validate provider config and print status."""
        if LLM_PROVIDER == "disabled":
            self._active = False
            print("[Narrator] Stage 5 disabled (LLM_PROVIDER=disabled).")

        elif LLM_PROVIDER == "azure":
            missing = []
            if not AZURE_OPENAI_API_KEY:    missing.append("AZURE_OPENAI_API_KEY")
            if not AZURE_OPENAI_ENDPOINT:   missing.append("AZURE_OPENAI_ENDPOINT")
            if missing:
                self._active = False
                print(f"[Narrator] Stage 5 DISABLED — missing env vars: {', '.join(missing)}")
                print( "           Set them and restart, or set LLM_PROVIDER=gemini/openai.")
            else:
                print(f"[Narrator] Ready. Provider=azure  deployment={AZURE_OPENAI_DEPLOYMENT}"
                      f"  interval={NARRATE_INTERVAL_S}s  blur={BLUR_FACES_BEFORE_SEND}")

        elif LLM_PROVIDER == "gemini":
            if not GEMINI_API_KEY:
                self._active = False
                print("[Narrator] Stage 5 DISABLED — GEMINI_API_KEY not set.")
            else:
                print(f"[Narrator] Ready. Provider=gemini  interval={NARRATE_INTERVAL_S}s")

        elif LLM_PROVIDER == "openai":
            if not OPENAI_API_KEY:
                self._active = False
                print("[Narrator] Stage 5 DISABLED — OPENAI_API_KEY not set.")
            else:
                print(f"[Narrator] Ready. Provider=openai  interval={NARRATE_INTERVAL_S}s")

        else:
            self._active = False
            print(f"[Narrator] Unknown LLM_PROVIDER: {LLM_PROVIDER!r} — Stage 5 disabled.")

    # ── Public API ────────────────────────────────────────

    def maybe_narrate(self, frame: np.ndarray, tracks: list, force: bool = False):
        """
        For each track in `tracks`, trigger an async narration if the
        per-person interval has elapsed (or if `force` is True).

        Parameters
        ----------
        frame  : np.ndarray  — current full frame (not modified)
        tracks : list[TrackResult]  — current detections from PersonTracker
        force  : bool  — bypass the interval check (e.g. key press)
        """
        if not self._active or not tracks:
            return

        now = time.time()
        for t in tracks:
            last = self._last_narrate.get(t.track_id, 0.0)
            if force or (now - last) >= NARRATE_INTERVAL_S:
                self._last_narrate[t.track_id] = now
                crop = _crop_box(frame, t.box)
                if crop.size == 0:
                    continue
                crop_copy = crop.copy()
                threading.Thread(
                    target = self._narrate_async,
                    args   = (crop_copy, t.track_id),
                    daemon = True,
                ).start()

    def _narrate_async(self, crop: np.ndarray, track_id: int):
        """Blocking LLM call — runs in its own thread."""
        with self._lock:
            print(f"[Narrator] Describing person ID={track_id}…")
            description = _describe_crop(crop)
            if not description:
                return

            self._log.activity_detected(label=description, person_id=str(track_id))
            tag = f"ID={track_id}"
            print(f"[Narrator] {tag}: {description}")

            # High-priority alert check
            kw = _has_high_priority_keyword(description)
            if kw:
                alert_msg = f"HIGH PRIORITY — ID={track_id} may be '{kw}': {description}"
                print(f"\n{'='*60}")
                print(f"  ⚠  ALERT: {alert_msg}")
                print(f"{'='*60}\n")
                self._log.high_priority_alert(alert_msg)

    def generate_summary(self) -> str:
        """
        Pull today's activity log + event counts and ask the LLM for a
        human-readable shift summary.  Returns the summary string.
        """
        if not self._active:
            return "Stage 5 not active — no summary available."

        narrations   = self._log.today_narrations()
        event_counts = self._log.today_summary()

        if not narrations:
            return "No activity narrations recorded yet — run the pipeline longer."

        narration_text = "\n".join(
            f"  [{r['timestamp']}] Person {r['person_id']}: {r['extra']}"
            for r in narrations
        )
        counts_text = "\n".join(f"  {k}: {v}" for k, v in event_counts.items())

        prompt = SUMMARY_PROMPT_TEMPLATE.format(
            narrations   = narration_text,
            event_counts = counts_text,
        )

        print("\n[Narrator] Generating shift summary…")
        try:
            if LLM_PROVIDER == "azure":
                from openai import AzureOpenAI
                client = AzureOpenAI(
                    api_key        = AZURE_OPENAI_API_KEY,
                    azure_endpoint = AZURE_OPENAI_ENDPOINT,
                    api_version    = AZURE_OPENAI_API_VERSION,
                )
                response = client.chat.completions.create(
                    model    = AZURE_OPENAI_DEPLOYMENT,
                    messages = [{"role": "user", "content": prompt}],
                    max_tokens = 300,
                )
                summary = response.choices[0].message.content.strip()
            elif LLM_PROVIDER == "gemini":
                import google.generativeai as genai
                genai.configure(api_key=GEMINI_API_KEY)
                summary = _gemini_generate(_GEMINI_TEXT_MODELS, prompt)
            elif LLM_PROVIDER == "openai":
                from openai import OpenAI
                client  = OpenAI(api_key=OPENAI_API_KEY)
                response = client.chat.completions.create(
                    model    = "gpt-4o-mini",
                    messages = [{"role": "user", "content": prompt}],
                    max_tokens = 300,
                )
                summary = response.choices[0].message.content.strip()
            else:
                summary = "Unknown provider."
        except Exception as e:
            summary = f"Summary generation failed: {e}"

        print(f"\n=== Shift Summary ===\n{summary}\n")
        self._log.llm_narration(f"[SHIFT SUMMARY] {summary}")
        return summary
