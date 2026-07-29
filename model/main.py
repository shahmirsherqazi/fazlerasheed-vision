"""
main.py — AI Camera Prototype (All 4 Stages)
=============================================
Run this to start the live webcam pipeline.

  python main.py                    # default webcam (index 0)
  python main.py --source 1         # second webcam
  python main.py --no-recognition   # YOLO only, skip face recognition

Enrol your face first:
  python recognizer.py --enroll YourName

Controls:
  Q     — quit
  S     — print today's event summary to console
"""

import argparse
import time
import cv2

from capture    import CameraSource
from detector   import PersonDetector
from recognizer import FaceRecognizer
from logger     import EventLogger


# ── Throttle settings ─────────────────────────────────────
# Run heavy inference at most this many times per second to keep CPU happy.
DETECTION_FPS_LIMIT   = 10   # YOLO inference max rate
RECOGNITION_FPS_LIMIT = 3    # Face recognition max rate (slower model)


def run(source=0, use_recognition=True):
    log       = EventLogger()
    detector  = PersonDetector()
    recognizer = FaceRecognizer() if use_recognition else None

    last_detect_time  = 0.0
    last_recog_time   = 0.0

    # Track which names we've already logged this "session"
    # so we don't spam the DB with a new row every frame.
    _seen_this_second: dict = {}

    with CameraSource(source) as cam:
        print("\n[Main] Pipeline running. Press Q to quit, S for summary.\n")

        while True:
            ret, frame = cam.read()
            if not ret:
                print("[Main] Failed to grab frame — stopping.")
                break

            now = time.time()

            # ── Stage 2: Person detection ──────────────────
            if now - last_detect_time >= 1.0 / DETECTION_FPS_LIMIT:
                boxes, frame = detector.detect(frame)
                if boxes:
                    log.person_detected(count=len(boxes))
                last_detect_time = now

            # ── Stage 3: Face recognition ──────────────────
            if use_recognition and recognizer is not None:
                if now - last_recog_time >= 1.0 / RECOGNITION_FPS_LIMIT:
                    faces, frame = recognizer.identify(frame)
                    for f in faces:
                        name = f["name"]
                        # Log each person at most once per 10 s
                        if now - _seen_this_second.get(name, 0) > 10:
                            if name == "Unknown":
                                log.unknown_face()
                            else:
                                log.face_recognized(name)
                            _seen_this_second[name] = now
                    last_recog_time = now

            # ── Display ────────────────────────────────────
            # Overlay instructions
            cv2.putText(frame, "Q=Quit  S=Summary",
                        (10, frame.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("Fazlerasheed Vision — Prototype", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                summary = log.today_summary()
                print("\n=== Today's Summary ===")
                for k, v in summary.items():
                    print(f"  {k:25s} : {v}")
                print()

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
    args = parser.parse_args()

    run(source=args.source, use_recognition=not args.no_recognition)
