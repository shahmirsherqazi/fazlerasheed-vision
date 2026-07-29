"""
recognizer.py — Face recognition (Stage 3)
==========================================
Uses insightface (Buffalo_L model) for robust face recognition on Windows.
Falls back gracefully if the model isn't available.

Workflow:
  1. Run `python recognizer.py --enroll YourName` to capture reference photos
     from the webcam and save face encodings to known_faces/
  2. The main pipeline calls FaceRecognizer.identify(frame, person_boxes)
     which returns labelled results for each detected face region.
"""

import os
import pickle
import argparse
import time
import cv2
import numpy as np

KNOWN_FACES_DIR = os.path.join(os.path.dirname(__file__), "known_faces")
ENCODINGS_FILE  = os.path.join(KNOWN_FACES_DIR, "encodings.pkl")
ENROLL_PHOTOS   = 5          # number of reference photos to capture per person
SIMILARITY_THRESHOLD = 0.4  # cosine distance threshold for a match


def _load_insightface():
    """Load insightface app. Returns app or None if not installed."""
    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(640, 640))
        print("[Recognizer] insightface loaded (buffalo_l).")
        return app
    except ImportError:
        print("[Recognizer] insightface not installed. Recognition disabled.")
        print("             Run: pip install insightface onnxruntime")
        return None


def _cosine_distance(a, b):
    """Lower = more similar. 0 = identical, 1 = completely different."""
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return 1.0 - float(np.dot(a, b))


class FaceRecognizer:
    def __init__(self):
        os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
        self._app = _load_insightface()
        self._known = self._load_known()

    # ── Encoding store ────────────────────────────────────
    def _load_known(self):
        if os.path.exists(ENCODINGS_FILE):
            with open(ENCODINGS_FILE, "rb") as f:
                known = pickle.load(f)
            print(f"[Recognizer] Loaded {len(known)} known person(s): "
                  f"{list(known.keys())}")
            return known
        return {}

    def _save_known(self):
        with open(ENCODINGS_FILE, "wb") as f:
            pickle.dump(self._known, f)

    # ── Enrolment ─────────────────────────────────────────
    def enroll(self, name: str, source=0):
        """
        Open the webcam, capture ENROLL_PHOTOS frames with a detected face,
        compute embeddings, and save them under `name`.
        Press SPACE to capture, Q to abort.
        """
        if self._app is None:
            print("[Recognizer] Cannot enroll — insightface not available.")
            return

        cap = cv2.VideoCapture(source)
        encodings = []
        print(f"\nEnrolling '{name}'. Press SPACE to capture, Q to quit.")
        print(f"Need {ENROLL_PHOTOS} good captures.\n")

        while len(encodings) < ENROLL_PHOTOS:
            ret, frame = cap.read()
            if not ret:
                break

            display = frame.copy()
            faces = self._app.get(frame)

            if faces:
                face = faces[0]  # take largest / first face
                box = face.bbox.astype(int)
                cv2.rectangle(display, (box[0], box[1]), (box[2], box[3]),
                              (0, 255, 0), 2)
                cv2.putText(display, f"Captured: {len(encodings)}/{ENROLL_PHOTOS}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 0), 2)
            else:
                cv2.putText(display, "No face detected",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 0, 255), 2)

            cv2.imshow(f"Enroll — {name}", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(" ") and faces:
                encodings.append(faces[0].embedding.copy())
                print(f"  Captured {len(encodings)}/{ENROLL_PHOTOS}")
                time.sleep(0.3)
            elif key == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

        if encodings:
            # Average the captured embeddings for robustness
            self._known[name] = np.mean(encodings, axis=0)
            self._save_known()
            print(f"[Recognizer] '{name}' enrolled with {len(encodings)} sample(s).")
        else:
            print("[Recognizer] No encodings captured — enrolment cancelled.")

    # ── Live identification ───────────────────────────────
    def identify(self, frame):
        """
        Detect all faces in `frame`, match against known encodings.

        Returns
        -------
        results : list of dict
            [{"name": str, "box": (x1,y1,x2,y2), "distance": float}, ...]
        frame   : np.ndarray  (annotated in-place)
        """
        results = []
        if self._app is None:
            return results, frame

        faces = self._app.get(frame)
        for face in faces:
            box = face.bbox.astype(int)
            emb = face.embedding

            name = "Unknown"
            best_dist = SIMILARITY_THRESHOLD

            for known_name, known_emb in self._known.items():
                dist = _cosine_distance(emb, known_emb)
                if dist < best_dist:
                    best_dist = dist
                    name = known_name

            color = (0, 220, 80) if name != "Unknown" else (0, 60, 220)
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            label = name if name == "Unknown" else f"{name} ({1-best_dist:.0%})"
            cv2.putText(frame, label,
                        (box[0], box[1] - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2)

            results.append({
                "name": name,
                "box": (int(box[0]), int(box[1]), int(box[2]), int(box[3])),
                "distance": float(best_dist),
            })

        return results, frame


# ── Enrolment CLI ─────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enroll a known face.")
    parser.add_argument("--enroll", metavar="NAME",
                        help="Name of the person to enroll")
    parser.add_argument("--source", type=int, default=0,
                        help="Camera index (default: 0)")
    args = parser.parse_args()

    if args.enroll:
        rec = FaceRecognizer()
        rec.enroll(args.enroll, source=args.source)
    else:
        parser.print_help()
