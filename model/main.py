"""
main.py — AI Camera Prototype (Stages 1–5)
==========================================
Run this to start the live webcam pipeline.

  python main.py                    # default webcam (index 0)
  python main.py --source 1         # second webcam
  python main.py --no-recognition   # YOLO only, skip face recognition
  python main.py --no-narration     # disable Stage 5 (LLM narration)

Enrol your face first:
  python recognizer.py --enroll YourName

For Stage 5 (LLM scene narration), set one of these before running:
  $env:LLM_PROVIDER = "gemini"        (PowerShell)
  $env:GEMINI_API_KEY = "your-key"
  -- or --
  $env:LLM_PROVIDER = "openai"
  $env:OPENAI_API_KEY = "your-key"

Controls:
  Q — quit
  S — print today's event summary to console
  N — force an immediate LLM narration (Stage 5)
  R — generate and print end-of-shift summary (Stage 5)
"""

import argparse
import time
import threading
import cv2

from capture    import CameraSource
from detector   import PersonDetector
from recognizer import FaceRecognizer
from logger     import EventLogger
from narrator   import Narrator


# ── Performance tuning ────────────────────────────────────
# Per the instructions: "downscale before inference, skip frames,
# gate face recognition behind person detection"
INFERENCE_WIDTH       = 640    # downscale frames to this width before detection
                               # (keeps display at full resolution)
DETECT_EVERY_N_FRAMES = 2      # run YOLO every N frames, display all frames
RECOGNITION_EVERY_S   = 0.5    # face recognition max rate (seconds between runs)


# ── Threaded frame reader ──────────────────────────────────
class ThreadedReader:
    """
    Reads frames from CameraSource in a background thread so slow inference
    never blocks the capture. Implements the instruction's Step 4 fix.
    """

    def __init__(self, cam: CameraSource):
        self._cam   = cam
        self._frame = None
        self._ret   = False
        self._lock  = threading.Lock()
        self._stop  = threading.Event()
        t = threading.Thread(target=self._read_loop, daemon=True)
        t.start()

    def _read_loop(self):
        while not self._stop.is_set():
            ret, frame = self._cam.read()
            with self._lock:
                self._ret   = ret
                self._frame = frame

    def read(self):
        with self._lock:
            return self._ret, (self._frame.copy() if self._frame is not None else None)

    def stop(self):
        self._stop.set()


def run(source=0, use_recognition=True, use_narration=True):
    log        = EventLogger()
    detector   = PersonDetector()
    recognizer = FaceRecognizer() if use_recognition else None
    narrator   = Narrator(log)   if use_narration   else None

    last_recog_time  = 0.0
    frame_counter    = 0
    last_boxes       = []         # reuse last detection between YOLO frames
    _seen_names: dict = {}        # rate-limit logging per person

    with CameraSource(source) as cam:
        reader = ThreadedReader(cam)
        print("\n[Main] Pipeline running.")
        print("       Q=Quit  S=Summary  N=Narrate now  R=Shift report\n")

        while True:
            ret, frame = reader.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            now = time.time()
            frame_counter += 1

            # ── Stage 2: Person detection (every N frames) ──
            # Downscale for inference, keep full-res frame for display
            if frame_counter % DETECT_EVERY_N_FRAMES == 0:
                h, w = frame.shape[:2]
                scale = INFERENCE_WIDTH / w if w > INFERENCE_WIDTH else 1.0
                small = cv2.resize(frame, (0, 0), fx=scale, fy=scale) if scale < 1 else frame

                boxes_small, small = detector.detect(small)

                # Scale boxes back to original resolution
                if scale < 1:
                    last_boxes = [
                        (int(x1/scale), int(y1/scale),
                         int(x2/scale), int(y2/scale), conf)
                        for x1, y1, x2, y2, conf in boxes_small
                    ]
                    # Redraw boxes at full res
                    for (x1, y1, x2, y2, conf) in last_boxes:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 255), 2)
                        cv2.putText(frame, f"Person {conf:.0%}",
                                    (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.55, (0, 180, 255), 2)
                else:
                    last_boxes = boxes_small
                    frame = small

                if last_boxes:
                    log.person_detected(count=len(last_boxes))

            else:
                # Display only — redraw last known boxes without re-running YOLO
                for (x1, y1, x2, y2, conf) in last_boxes:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 255), 2)
                    cv2.putText(frame, f"Person {conf:.0%}",
                                (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, (0, 180, 255), 2)

            # ── Stage 3: Face recognition ──────────────────
            # Gate behind "a person was detected" (per instructions Step 3)
            if (use_recognition and recognizer is not None
                    and last_boxes
                    and now - last_recog_time >= RECOGNITION_EVERY_S):
                faces, frame = recognizer.identify(frame)
                for f in faces:
                    name = f["name"]
                    if now - _seen_names.get(name, 0) > 10:
                        if name == "Unknown":
                            log.unknown_face()
                        else:
                            log.face_recognized(name)
                        _seen_names[name] = now
                last_recog_time = now

            # ── Stage 5: Periodic narration ────────────────
            if narrator is not None:
                narrator.maybe_narrate(frame)

            # ── Display ────────────────────────────────────
            cv2.putText(frame, "Q=Quit  S=Summary  N=Narrate  R=Report",
                        (10, frame.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

            cv2.imshow("Fazlerasheed Vision — Prototype", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            elif key == ord("s"):
                summary = log.today_summary()
                print("\n=== Today's Event Summary ===")
                for k, v in summary.items():
                    print(f"  {k:30s} : {v}")
                print()

            elif key == ord("n") and narrator is not None:
                narrator.maybe_narrate(frame, force=True)

            elif key == ord("r") and narrator is not None:
                # Generate shift summary in a thread so UI doesn't freeze
                threading.Thread(
                    target=narrator.generate_summary, daemon=True
                ).start()

    reader.stop()
    cv2.destroyAllWindows()
    log.close()
    print("[Main] Stopped.")


# ── Entry point ───────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fazlerasheed Vision — AI Camera Prototype"
    )
    parser.add_argument(
        "--source", type=int, default=0,
        help="Camera index (default: 0 = built-in webcam)"
    )
    parser.add_argument(
        "--no-recognition", action="store_true",
        help="Disable face recognition (run YOLO only)"
    )
    parser.add_argument(
        "--no-narration", action="store_true",
        help="Disable Stage 5 LLM narration (fully local mode)"
    )
    args = parser.parse_args()

    run(
        source=args.source,
        use_recognition=not args.no_recognition,
        use_narration=not args.no_narration,
    )
