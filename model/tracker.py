"""
tracker.py — YOLOv8n + ByteTrack multi-person detection & tracking (Stage 2/3)
===============================================================================
Replaces the old two-step (PersonDetector → CSRT) approach.

Uses ultralytics' built-in `model.track()` which runs YOLOv8n detection AND
ByteTrack assignment in a single call.  ByteTrack assigns a stable integer ID
to every person and keeps it consistent across frames — even through brief
occlusions.

Usage
-----
    from tracker import PersonTracker

    pt = PersonTracker()
    tracks = pt.update(frame)   # returns list[TrackResult], draws on frame
    for t in tracks:
        print(t.track_id, t.box)   # (x1, y1, x2, y2)
"""

import cv2
import numpy as np
from ultralytics import YOLO

PERSON_CLASS_ID      = 0      # YOLO COCO class for "person"
CONFIDENCE_THRESHOLD = 0.40   # minimum detection confidence
MODEL_PATH           = "yolov8n.pt"

# Colour palette — up to 16 distinct colours for track IDs
_PALETTE = [
    (0, 180, 255), (0, 255, 120), (255, 80, 0), (180, 0, 255),
    (255, 220, 0), (0, 220, 220), (255, 0, 140), (100, 255, 0),
    (0, 100, 255), (255, 140, 0), (0, 255, 200), (200, 0, 200),
    (255, 60, 60), (60, 255, 60), (60, 60, 255), (200, 200, 0),
]


def _id_colour(track_id: int):
    return _PALETTE[track_id % len(_PALETTE)]


class TrackResult:
    """One tracked person in one frame."""
    __slots__ = ("track_id", "box", "conf")

    def __init__(self, track_id: int, box: tuple[int, int, int, int], conf: float):
        self.track_id = track_id   # persistent ByteTrack ID
        self.box      = box        # (x1, y1, x2, y2) in original frame coords
        self.conf     = conf


class PersonTracker:
    """
    Wraps YOLOv8n + ByteTrack via ultralytics.

    Call `update(frame)` on every frame.  The method:
      1. Runs `model.track()` (detection + ByteTrack in one pass).
      2. Draws colour-coded bounding boxes + ID labels on the frame.
      3. Returns a list of TrackResult objects for this frame.

    Notes
    -----
    - `persist=True` is required — it tells ultralytics to carry state between
      calls so ByteTrack can maintain track continuity.
    - Results with no track_id (fresh detections not yet assigned) are skipped.
    - The frame is annotated in-place; pass a copy if you need the original.
    """

    def __init__(self, model_path: str = MODEL_PATH):
        print(f"[PersonTracker] Loading YOLOv8n + ByteTrack from: {model_path}")
        self._model = YOLO(model_path)
        self._prev_ids: set = set()
        print("[PersonTracker] Ready.")

    def update(self, frame: np.ndarray) -> list[TrackResult]:
        """
        Run detection + tracking on `frame`.

        Parameters
        ----------
        frame : np.ndarray  (BGR, HxWx3)
            Current video frame.  Annotated in-place.

        Returns
        -------
        list[TrackResult]
            One entry per person currently being tracked.
            Empty list if no persons detected.
        """
        results = self._model.track(
            frame,
            classes   = [PERSON_CLASS_ID],
            conf      = CONFIDENCE_THRESHOLD,
            tracker   = "bytetrack.yaml",
            persist   = True,
            verbose   = False,
        )

        tracks: list[TrackResult] = []
        current_ids: set = set()

        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                # skip boxes without a track ID (ultralytics sets .id = None
                # for detections that ByteTrack hasn't confirmed yet)
                if box.id is None:
                    continue

                track_id = int(box.id[0])
                conf     = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                current_ids.add(track_id)
                tracks.append(TrackResult(track_id, (x1, y1, x2, y2), conf))

                # Draw annotated box
                colour = _id_colour(track_id)
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                label = f"ID {track_id}  {conf:.0%}"
                cv2.putText(frame, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)

        # Console events for enter/exit
        entered = current_ids - self._prev_ids
        left    = self._prev_ids - current_ids
        for tid in entered:
            print(f"[Tracker] Person ENTERED  ID={tid}")
        for tid in left:
            print(f"[Tracker] Person LEFT     ID={tid}")
        self._prev_ids = current_ids

        return tracks
