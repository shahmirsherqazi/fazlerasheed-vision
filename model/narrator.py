"""
narrator.py — Stage 5: Periodic vision-LLM narration (activity summaries)
==========================================================================
Runs alongside the local detection pipeline (YOLO + face recognition).
Does NOT replace them. Does NOT run on every frame.

Behaviour:
  - Every NARRATE_INTERVAL_S seconds, OR when triggered by a detected event,
    grab the latest frame, optionally blur faces, and send it to a vision LLM
    with a scene-description prompt.
  - Store the returned description in the EventLogger under event_type
    "llm_narration".
  - On demand (or at end of shift), generate a human-readable summary from
    the accumulated narrations + structured events.

Supported providers (set LLM_PROVIDER in config or env):
  "gemini"   — Google Gemini 1.5 Flash (recommended; needs GEMINI_API_KEY)
  "openai"   — OpenAI GPT-4o-mini (needs OPENAI_API_KEY)
  "disabled" — Stage 5 completely off (local-only mode)

Privacy note:
  Set BLUR_FACES_BEFORE_SEND=True to anonymise faces in frames before
  they leave the device. Off by default for prototype — confirm with client.
"""

import os
import base64
import time
import threading
from io import BytesIO

import cv2
import numpy as np

from logger import EventLogger

# ── Config (override via environment variables) ────────────
LLM_PROVIDER          = os.getenv("LLM_PROVIDER", "gemini")   # gemini | openai | disabled
GEMINI_API_KEY        = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY        = os.getenv("OPENAI_API_KEY", "")
NARRATE_INTERVAL_S    = int(os.getenv("NARRATE_INTERVAL_S", "180"))  # every 3 min
BLUR_FACES_BEFORE_SEND = os.getenv("BLUR_FACES_BEFORE_SEND", "false").lower() == "true"

SCENE_PROMPT = (
    "You are a security monitoring assistant. Describe what is happening in this "
    "scene in 1–2 plain sentences. Focus on: what people are doing, their general "
    "appearance (clothing colour, posture), and anything unusual. "
    "Do NOT attempt to identify who the people are by name."
)

SUMMARY_PROMPT_TEMPLATE = (
    "You are generating an end-of-shift activity report for a security system. "
    "Below are timestamped scene descriptions collected over a period. "
    "Write a concise, professional plain-English summary of the day's activity "
    "(3–5 sentences). Note any notable events or patterns.\n\n"
    "Scene descriptions:\n{narrations}\n\n"
    "Event counts:\n{event_counts}"
)


# ── Frame helpers ──────────────────────────────────────────

def _blur_faces(frame: np.ndarray) -> np.ndarray:
    """
    Blur detected faces in `frame` before sending to external API.
    Uses OpenCV's DNN face detector (no extra dependencies).
    Falls back gracefully if the model file isn't available.
    """
    try:
        # Use a simple cascade as a lightweight fallback
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascade_path)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
        out = frame.copy()
        for (x, y, w, h) in faces:
            roi = out[y:y+h, x:x+w]
            out[y:y+h, x:x+w] = cv2.GaussianBlur(roi, (51, 51), 0)
        return out
    except Exception as e:
        print(f"[Narrator] Face blur failed ({e}) — sending unblurred frame.")
        return frame


def _frame_to_b64_jpeg(frame: np.ndarray, quality: int = 70) -> str:
    """Encode a BGR numpy frame as a base64 JPEG string."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


# ── LLM clients ────────────────────────────────────────────

def _call_gemini(b64_image: str) -> str:
    """Send frame to Gemini 1.5 Flash vision API."""
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")
        image_part = {
            "mime_type": "image/jpeg",
            "data": b64_image,
        }
        response = model.generate_content([SCENE_PROMPT, image_part])
        return response.text.strip()
    except ImportError:
        return "[Narrator] google-generativeai not installed. Run: pip install google-generativeai"
    except Exception as e:
        return f"[Narrator] Gemini API error: {e}"


def _call_openai(b64_image: str) -> str:
    """Send frame to OpenAI GPT-4o-mini vision API."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": SCENE_PROMPT},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{b64_image}",
                        "detail": "low",
                    }},
                ],
            }],
            max_tokens=150,
        )
        return response.choices[0].message.content.strip()
    except ImportError:
        return "[Narrator] openai not installed. Run: pip install openai"
    except Exception as e:
        return f"[Narrator] OpenAI API error: {e}"


def _describe_frame(frame: np.ndarray) -> str:
    """Route to the configured LLM provider."""
    if LLM_PROVIDER == "disabled":
        return ""

    send_frame = _blur_faces(frame) if BLUR_FACES_BEFORE_SEND else frame
    b64 = _frame_to_b64_jpeg(send_frame)

    if LLM_PROVIDER == "gemini":
        return _call_gemini(b64)
    elif LLM_PROVIDER == "openai":
        return _call_openai(b64)
    else:
        return f"[Narrator] Unknown provider: {LLM_PROVIDER}"


# ── Main narrator class ────────────────────────────────────

class Narrator:
    """
    Wraps the periodic narration logic.
    Call `maybe_narrate(frame, force=False)` each iteration — it decides
    internally whether enough time has passed to trigger a new narration.
    Call `generate_summary()` at end of shift.
    """

    def __init__(self, logger: EventLogger):
        self._log = logger
        self._last_narrate = 0.0
        self._lock = threading.Lock()

        if LLM_PROVIDER == "disabled":
            print("[Narrator] Stage 5 disabled (LLM_PROVIDER=disabled).")
        elif LLM_PROVIDER == "gemini" and not GEMINI_API_KEY:
            print("[Narrator] WARNING — GEMINI_API_KEY not set. "
                  "Set it as an environment variable or in .env to enable narration.")
        elif LLM_PROVIDER == "openai" and not OPENAI_API_KEY:
            print("[Narrator] WARNING — OPENAI_API_KEY not set.")
        else:
            print(f"[Narrator] Ready. Provider={LLM_PROVIDER}, "
                  f"interval={NARRATE_INTERVAL_S}s, "
                  f"blur_faces={BLUR_FACES_BEFORE_SEND}")

    def maybe_narrate(self, frame: np.ndarray, force: bool = False) -> bool:
        """
        If `force` is True or NARRATE_INTERVAL_S have elapsed, describe the
        frame and log it. Runs in a background thread so it never blocks the
        main loop.

        Returns True if a narration was triggered.
        """
        if LLM_PROVIDER == "disabled":
            return False

        now = time.time()
        if not force and (now - self._last_narrate) < NARRATE_INTERVAL_S:
            return False

        self._last_narrate = now
        # Take a copy so the main loop can keep using the original frame
        frame_copy = frame.copy()
        threading.Thread(
            target=self._narrate_async,
            args=(frame_copy,),
            daemon=True,
        ).start()
        return True

    def _narrate_async(self, frame: np.ndarray):
        """Blocking LLM call — runs in its own thread."""
        with self._lock:
            print("[Narrator] Sending frame for scene description…")
            description = _describe_frame(frame)
            if description:
                self._log.llm_narration(description)
                print(f"[Narrator] Scene: {description[:120]}…" if len(description) > 120
                      else f"[Narrator] Scene: {description}")

    def generate_summary(self) -> str:
        """
        Pull today's narrations + event counts from the log and ask the LLM
        to produce a human-readable shift summary.
        Returns the summary string (also prints it).
        """
        if LLM_PROVIDER == "disabled":
            return "Stage 5 disabled — no LLM summary available."

        narrations = self._log.today_narrations()
        event_counts = self._log.today_summary()

        if not narrations:
            return "No scene narrations recorded yet — run the pipeline longer."

        narration_text = "\n".join(
            f"  [{r['timestamp']}] {r['extra']}" for r in narrations
        )
        counts_text = "\n".join(f"  {k}: {v}" for k, v in event_counts.items())

        prompt = SUMMARY_PROMPT_TEMPLATE.format(
            narrations=narration_text,
            event_counts=counts_text,
        )

        print("\n[Narrator] Generating shift summary…")
        # For the summary we send text only (no image)
        try:
            if LLM_PROVIDER == "gemini":
                import google.generativeai as genai
                genai.configure(api_key=GEMINI_API_KEY)
                model = genai.GenerativeModel("gemini-1.5-flash")
                response = model.generate_content(prompt)
                summary = response.text.strip()
            elif LLM_PROVIDER == "openai":
                from openai import OpenAI
                client = OpenAI(api_key=OPENAI_API_KEY)
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300,
                )
                summary = response.choices[0].message.content.strip()
            else:
                summary = "Unknown provider."
        except Exception as e:
            summary = f"Summary generation failed: {e}"

        print(f"\n=== Shift Summary ===\n{summary}\n")
        self._log.llm_narration(f"[SHIFT SUMMARY] {summary}")
        return summary
