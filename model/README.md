# AI Camera Prototype — Webcam

Local AI pipeline: person detection (YOLOv8n) + face recognition (insightface) + SQLite event logging + periodic vision-LLM scene narration.

Designed to swap from webcam → RTSP/IP camera in the next phase with zero downstream changes.

## File structure

```
model/
├── main.py          ← entry point — run this
├── capture.py       ← Stage 1: swappable camera source
├── detector.py      ← Stage 2: YOLOv8n person detection
├── recognizer.py    ← Stage 3: insightface face recognition
├── logger.py        ← Stage 4: SQLite event log
├── narrator.py      ← Stage 5: periodic vision-LLM narration
├── requirements.txt
├── known_faces/     ← created when you enroll a person
│   └── encodings.pkl
└── events.db        ← created on first run
```

## Setup (Windows, Python 3.10+)

```powershell
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

> **insightface** downloads the `buffalo_l` model (~200 MB) on first run — needs internet.

## Enrol your face (do this before running)

```powershell
python recognizer.py --enroll YourName
```

- A webcam window opens. Press **SPACE** 5 times to capture references.
- Move your head slightly between captures for better coverage.
- Press **Q** to cancel.

## Run the pipeline

```powershell
python main.py                    # full pipeline (Stages 1–5)
python main.py --no-recognition   # YOLO only — no face matching
python main.py --no-narration     # local-only — no LLM API calls
python main.py --source 1         # use second camera
```

### Controls while running

| Key | Action |
|-----|--------|
| Q   | Quit |
| S   | Print today's event summary to console |
| N   | Force an immediate LLM scene narration (Stage 5) |
| R   | Generate and print end-of-shift summary (Stage 5) |

---

## Stage 5 — Vision-LLM narration setup

Stage 5 periodically sends a single frame to a vision LLM for a plain-English
scene description. It **never** runs on every frame and **never** uses the LLM
for identity/face recognition — that stays local.

### Using Gemini (recommended)

```powershell
$env:LLM_PROVIDER  = "gemini"
$env:GEMINI_API_KEY = "your-api-key-here"
python main.py
```

### Using OpenAI GPT-4o

```powershell
$env:LLM_PROVIDER  = "openai"
$env:OPENAI_API_KEY = "your-api-key-here"
python main.py
```

### Disable Stage 5 completely

```powershell
python main.py --no-narration
```

### Privacy — face blurring before sending frames

By default, frames are sent as-is. To blur faces before sending to the external API:

```powershell
$env:BLUR_FACES_BEFORE_SEND = "true"
python main.py
```

> ⚠️ Confirm with your client's data-handling policy before enabling cloud API narration.

### Narration interval

Default: one narration every **3 minutes**. Change it:

```powershell
$env:NARRATE_INTERVAL_S = "120"   # every 2 minutes
```

---

## Performance tuning (if the feed feels slow)

Applied in order (all already enabled in `main.py`):

1. **Frame downscaling** — inference runs at 640px wide, display stays full res
2. **Skip-N detection** — YOLO runs every 2nd frame; display shows every frame
3. **Gate recognition** — face recognition only triggers when a person is detected
4. **Threaded capture** — frames are read in a background thread so slow inference never blocks the camera

If still slow, reduce `DETECT_EVERY_N_FRAMES` or lower `INFERENCE_WIDTH` in `main.py`.

---

## View the raw event log

```powershell
python logger.py
```

Prints today's summary and the 10 most recent events from `events.db`.

---

## Swapping to an IP / RTSP camera

In `main.py`, change the one line inside `run()`:

```python
# Before (webcam)
with CameraSource(0) as cam:

# After (IP / RTSP camera)
with CameraSource("rtsp://user:pass@192.168.1.100:554/stream") as cam:
```

Everything else — detection, recognition, logging, narration — is unchanged.
