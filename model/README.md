# AI Camera Prototype — Webcam

Local AI pipeline: person detection (YOLOv8n) + face recognition (insightface) + SQLite event logging.
Designed to swap from webcam → RTSP/IP camera in the next phase with zero downstream changes.

## File structure

```
model/
├── main.py          ← entry point, run this
├── capture.py       ← swappable camera source (Stage 1)
├── detector.py      ← YOLOv8n person detection (Stage 2)
├── recognizer.py    ← insightface face recognition (Stage 3)
├── logger.py        ← SQLite event log (Stage 4)
├── requirements.txt
├── known_faces/     ← created automatically when you enroll
│   └── encodings.pkl
└── events.db        ← created automatically on first run
```

## Setup (Windows, Python 3.10+)

```powershell
# 1. Create and activate a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

> **Note:** `insightface` will download the `buffalo_l` model (~200 MB) on first run.
> Make sure you have an internet connection the first time.

## Enrol your face (do this before running the pipeline)

```powershell
python recognizer.py --enroll YourName
```

- A webcam window opens.
- Press **SPACE** to capture a reference photo (need 5).
- Move your head slightly between captures for better coverage.
- Press **Q** to cancel.

## Run the pipeline

```powershell
python main.py                    # full pipeline (detection + recognition)
python main.py --no-recognition   # YOLO only (faster, no face matching)
python main.py --source 1         # use second camera
```

### Controls while running
| Key | Action |
|-----|--------|
| Q   | Quit |
| S   | Print today's event summary to console |

## View the event log

```powershell
python logger.py
```

Prints today's summary and the 10 most recent logged events from `events.db`.

## Swapping to an IP / RTSP camera

In `main.py`, change:
```python
with CameraSource(0) as cam:      # webcam
```
to:
```python
with CameraSource("rtsp://user:pass@192.168.1.100:554/stream") as cam:
```
Everything downstream (`detector`, `recognizer`, `logger`) stays unchanged.
