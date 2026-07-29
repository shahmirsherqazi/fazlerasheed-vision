"""
detector.py — YOLOv8n person detection (Stage 2)
=================================================
Wraps ultralytics YOLOv8n.
Input  : a raw BGR frame (numpy array from CameraSource.read())
Output : list of bounding boxes for detected persons
         + draws boxes on the frame in-place
"""

from ultralytics import YOLO
import cv2

PERSON_CLASS_ID = 0          # YOLO class index for "person"
CONFIDENCE_THRESHOLD = 0.45  # only report detections above this confidence


class PersonDetector:
    def __init__(self, model_path="yolov8n.pt"):
        """
        model_path : str
            YOLOv8 weights file. 'yolov8n.pt' is downloaded automatically
            by ultralytics on first run.
        """
        print(f"[PersonDetector] Loading model: {model_path}")
        self.model = YOLO(model_path)
        self._prev_count = 0

    def detect(self, frame):
        """
        Run inference on `frame`.

        Returns
        -------
        boxes : list of (x1, y1, x2, y2, confidence)
            One entry per detected person.
        frame : np.ndarray
            Same frame with bounding boxes drawn on it.
        """
        results = self.model(frame, classes=[PERSON_CLASS_ID],
                             conf=CONFIDENCE_THRESHOLD, verbose=False)

        boxes = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                boxes.append((x1, y1, x2, y2, conf))

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 255), 2)
                cv2.putText(frame, f"Person {conf:.0%}",
                            (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (0, 180, 255), 2)

        # Log enter/exit events
        current_count = len(boxes)
        if current_count > self._prev_count:
            print(f"[Event] Person ENTERED frame  (count: {current_count})")
        elif current_count < self._prev_count:
            print(f"[Event] Person LEFT frame     (count: {current_count})")
        self._prev_count = current_count

        return boxes, frame
