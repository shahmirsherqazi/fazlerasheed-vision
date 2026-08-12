"""
main.py — AI Camera Prototype (ByteTrack multi-person + Azure activity narration)
==================================================================================
Run this to start the live webcam pipeline.

  python main.py                    # default webcam (index 0)
  python main.py --source 1         # second webcam
  python main.py --no-recognition   # skip face recognition
  python main.py --no-narration     # disable Stage 5 (LLM narration)

Enrol your face first:
  python recognizer.py --enroll YourName

Stage 5 — Azure OpenAI (primary, per the activity-tracker plan):
  Fill in .env (in this directory) with:
    AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT

  -- or fall back to Gemini --
  Set LLM_PROVIDER=gemini and GEMINI_API_KEY in .env

Controls:
  Q — quit
  S — print today's event summary to console
  N — force immediate narration for all visible persons
  R — generate and print end-of-shift summary
  A — show today's high-priority alerts
"""

import os
import argparse
import time
import threading
import cv2

# Load .env first — must happen before any module reads os.getenv()
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

from capture    import CameraSource
from tracker    import PersonTracker
from recognizer import FaceRecognizer
from logger     import EventLogger
from narrator   import Narrator


# ── Performance tuning ─────────────────────────────────────
# ByteTrack runs on every frame — reduce scale if CPU is too slow.
INFERENCE_WIDTH       = 640     # downscale to this width before track()
RECOGNITION_EVERY_S   = 1.0     # face recognition max rate (seconds between runs)


# ── Threaded frame reader ───────────────────────────────────
class ThreadedReader:
    """
    Reads frames from CameraSource in a background thread so slow inference
    never blocks the capture loop.
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
            if not ret:
                time.sleep(0.005)

    def read(self):
        with self._lock:
            return self._ret, (self._frame.copy() if self._frame is not None else None)

    def stop(self):
        self._stop.set()


def run(source=0, use_recognition=True, use_narration=True):
    log        = EventLogger()
    tracker    = PersonTracker()
    recognizer = FaceRecognizer() if use_recognition else None
    narrator   = Narrator(log)   if use_narration   else None

    last_recog_time = 0.0
    _seen_names: dict = {}    # rate-limit face-recognition logging per person

    with CameraSource(source) as cam:
        reader = ThreadedReader(cam)
        print("\n[Main] Pipeline running.")
        print("       Q=Quit  S=Summary  N=Narrate  R=Report  A=Alerts\n")

        while True:
            ret, frame = reader.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            now = time.time()

            # ── Stage 2+3: Detection + ByteTrack tracking ──────────
            # Downscale for inference, keep full-res for display
            h, w = frame.shape[:2]
            scale = INFERENCE_WIDTH / w if w > INFERENCE_WIDTH else 1.0

            if scale < 1.0:
                small  = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
                tracks_small = tracker.update(small)
                # Scale track boxes back to full-res
                for t in tracks_small:
                    x1, y1, x2, y2 = t.box
                    t.box = (
                        int(x1 / scale), int(y1 / scale),
                        int(x2 / scale), int(y2 / scale),
                    )
                tracks = tracks_small
                # Redraw boxes on full-res frame (tracker annotated the small copy)
                from tracker import _id_colour
                for t in tracks:
                    col = _id_colour(t.track_id)
                    x1, y1, x2, y2 = t.box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
                    cv2.putText(frame, f"ID {t.track_id}  {t.conf:.0%}",
                                (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
            else:
                tracks = tracker.update(frame)

            if tracks:
                log.person_detected(count=len(tracks))

            # ── Stage 4: Face recognition ───────────────────────────
            # Gate behind person detection; run at limited rate
            if (use_recognition and recognizer is not None
                    and tracks
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

            # ── Stage 5: Per-person activity narration ──────────────
            if narrator is not None and tracks:
                narrator.maybe_narrate(frame, tracks)

            # ── Overlay the latest activity label per visible ID ────
            # (narrator runs async — we just show whatever was last logged)
            # (no extra overlay needed — track boxes already have ID labels)

            # ── Display ─────────────────────────────────────────────
            cv2.putText(frame, "Q=Quit  S=Summary  N=Narrate  R=Report  A=Alerts",
                        (10, frame.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

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
                if tracks:
                    narrator.maybe_narrate(frame, tracks, force=True)
                else:
                    print("[Main] No tracked persons — nothing to narrate.")

            elif key == ord("r") and narrator is not None:
                threading.Thread(
                    target=narrator.generate_summary, daemon=True
                ).start()

            elif key == ord("a"):
                alerts = log.today_high_priority()
                if alerts:
                    print(f"\n=== ⚠  High-Priority Alerts Today ({len(alerts)}) ===")
                    for a in alerts:
                        print(f"  {a['timestamp']}  {a['detail']}")
                    print()
                else:
                    print("[Main] No high-priority alerts today.")

    reader.stop()
    cv2.destroyAllWindows()
    log.close()
    print("[Main] Stopped.")


# ── Entry point ────────────────────────────────────────────
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
        help="Disable face recognition (run YOLO+ByteTrack only)"
    )
    parser.add_argument(
        "--no-narration", action="store_true",
        help="Disable Stage 5 LLM narration (fully local mode)"
    )
    args = parser.parse_args()

    run(
        source           = args.source,
        use_recognition  = not args.no_recognition,
        use_narration    = not args.no_narration,
    )
