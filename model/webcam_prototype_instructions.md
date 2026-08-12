# AI Camera Prototype — Webcam Phase (Build Instructions)

## Goal
Build a proof-of-concept pipeline on a laptop webcam that will later be ported
to a real IP/security camera without rewriting the core logic. The camera
input is just a frame source — everything downstream (detection, recognition)
must be written so the input source can be swapped later (webcam → RTSP
stream → Pi camera) with minimal changes.

## Environment
- OS: Windows
- Language: Python 3.10+
- Camera source for this phase: default laptop webcam (`cv2.VideoCapture(0)`)

## Dependencies to install
```
pip install opencv-python ultralytics face_recognition
```
Note: `face_recognition` depends on `dlib`, which can be difficult to build
on Windows. If installation fails, fall back to `insightface` as the face
recognition library instead and adjust code accordingly.

## Build in this order (do not skip ahead — each stage must work before the next)

### Stage 1 — Camera capture baseline
- Open the default webcam with OpenCV
- Display the live feed in a window
- Confirm frame rate and resolution are acceptable
- Exit cleanly on 'q' keypress
- Wrap the frame source in a function/class so it can later be swapped for
  an RTSP URL (`cv2.VideoCapture("rtsp://...")`) without changing anything
  downstream

### Stage 2 — Person detection
- Integrate YOLOv8n (via `ultralytics`) for person detection on each frame
- Draw bounding boxes around detected people on the display window
- Log a simple event whenever a person enters/exits frame (console print is
  fine for now — this will later feed the attendance/summary system)

### Stage 3 — Face recognition (attendance proof of concept)
- Capture 3–5 reference photos of one known person (the user) and store
  face encodings
- For each detected face in the live feed, compare against known encodings
- Label recognized faces with the person's name on screen; label unknown
  faces as "Unknown"
- Log recognition events with timestamps (this is the attendance data
  source later)

### Stage 4 — Structured event logging
- Instead of just printing to console, write detection/recognition events
  to a local file or lightweight database (SQLite is fine) with:
  timestamp, event type (person detected / face recognized / unknown face),
  person ID if known
- This structured log is what the later "twice-daily summary" feature will
  read from

### Stage 5 — Periodic vision-LLM narration (activity summaries)
- Do NOT call a vision LLM API on every frame — too slow and expensive.
  Local detection (Stages 2–4) stays the real-time engine.
- On a fixed interval (e.g. every few minutes) or on a triggered event
  (e.g. new person detected), capture a single frame and send it to a
  vision-capable LLM API (Gemini/GPT-4V/Claude) with a prompt asking for a
  plain-language description of the scene (activity, general appearance,
  anything notable)
- Store each returned description alongside its timestamp in the same
  event log used in Stage 4
- At the end of a day/night period, feed the collected descriptions to an
  LLM to generate the final human-readable summary text (the "twice-daily
  summary" feature)
- Do NOT use the vision LLM for identity/face recognition — that stays
  the job of the local face_recognition/InsightFace pipeline from Stage 3.
  Vision LLMs are for describing activity/scene, not for identifying who
  a person is.
- Privacy decision to make before sending any frames to a third-party API:
  whether frames sent to the vision LLM should have faces blurred/cropped
  first, since this sends images of people to an external company's
  servers rather than keeping everything local. Confirm this with the
  client's data-handling policy before implementing.

### Stage 5 — Vision-LLM narration layer (activity summaries)
- This stage adds a *separate* layer on top of the local detection pipeline
  — it does not replace face recognition or YOLO detection, which remain
  the source of truth for "who" and "is a person present"
- Local pipeline (YOLO + face recognition) keeps running continuously for
  real-time detection/attendance — this stays fast and cheap
- Periodically (e.g. once every few minutes, or once per detected event —
  not every frame) capture a single frame and send it to a vision-capable
  LLM API (Gemini/GPT-4V/Claude) with a prompt asking for a plain-language
  description of the scene (what the person is doing, general appearance)
- Store these periodic natural-language descriptions alongside the
  structured event log from Stage 4
- At the end of a shift/period, feed the collected descriptions + event log
  into an LLM to generate the final human-readable day/night summary
- Do NOT use the vision LLM for identity/face matching — that remains the
  job of the local face recognition system. Vision LLMs describe scenes,
  they do not reliably or appropriately perform facial identification, and
  most providers restrict that use case in their terms of service
- Note: sending frames to a third-party cloud API is a bigger privacy step
  than local-only processing. Consider whether frames need faces
  blurred/cropped before being sent, pending the client's data policy

## Explicitly out of scope for this phase
- Audio recording/transcription (separate pipeline, comes later, and
  requires a resolved consent/legal policy before implementation)
- Multi-camera support
- Any networking/RTSP work — webcam only for now
- The app/dashboard UI
- Any cloud or external server component — keep everything local for the
  prototype

## Success criteria for this phase
- Webcam feed opens and displays reliably
- People are detected with bounding boxes in real time (a few fps on CPU is
  acceptable)
- The known reference person is correctly recognized by name across
  multiple frames/angles
- Unknown faces are labeled as such, not misidentified
- Events are logged to a file with timestamps, in a format that could
  later be summarized (e.g. "how many people were detected today")

## How frame capture works (context for the swappable source design)
`cv2.VideoCapture` handles frame feeding automatically — there is no manual
step where frames are individually supplied. Once opened, calling `.read()`
inside a loop pulls whatever frame is currently available from the camera:

```python
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()   # pulls the next available frame
    # pass `frame` (a numpy array) into detection/recognition here
```

The loop runs as fast as its slowest step. If the camera delivers frames
faster than the detection model can process them, `.read()` will simply
return newer frames on each call — frames get skipped, not queued. This is
expected behavior for a real-time prototype, not a bug.

This is exactly why the frame source must be abstracted (see design
constraint below): `cap = cv2.VideoCapture(0)` for a webcam and
`cap = cv2.VideoCapture("rtsp://...")` for an IP camera work identically
from this point on — the `.read()` loop and everything downstream does not
need to change.

## Performance notes (fix for choppy/slow feed)
If the live feed feels slow or not smooth, apply these in order:
1. Downscale frames before running inference (e.g. resize to 640x360) —
   biggest speed win for least effort, accuracy loss is negligible at
   prototype distances
2. Do not run inference on every frame — process every 2nd or 3rd frame,
   but still call `cv2.imshow` on every frame so the displayed video stays
   visually smooth even though detection updates less often
3. Run face recognition less frequently than person detection — detection
   is comparatively cheap, face recognition is the expensive step, so gate
   it behind "a person was detected" rather than running it every cycle
4. If still too slow, move capture and inference to separate threads so a
   slow inference step doesn't block frame reads (more complex, use only
   if steps 1-3 aren't enough)
5. If `face_recognition` (dlib-based) is the bottleneck, consider switching
   to `insightface`, which is generally faster on CPU

## Design constraint (important)
Write the frame-source and inference code as separate, swappable pieces.
The webcam is a placeholder input for this phase — the same detection and
recognition code must run unmodified against an RTSP camera stream or a
Raspberry Pi camera in the next phase. Do not hardcode anything that
assumes a local webcam beyond the initial `VideoCapture` source.