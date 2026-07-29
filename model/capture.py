"""
capture.py — Swappable frame source
====================================
Abstract the camera input so the same code works with:
  - Webcam:        CameraSource(0)
  - RTSP stream:   CameraSource("rtsp://user:pass@192.168.1.100:554/stream")
  - Video file:    CameraSource("recording.mp4")

Nothing downstream should import cv2.VideoCapture directly.
"""

import cv2


class CameraSource:
    """
    Thin wrapper around cv2.VideoCapture.
    Swap `source` to change the input — everything downstream stays the same.
    """

    def __init__(self, source=0):
        """
        source : int | str
            0          → default laptop webcam
            "rtsp://…" → IP / RTSP camera (next phase)
            "path.mp4" → video file for testing
        """
        self.source = source
        self._cap = cv2.VideoCapture(source)

        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open camera source: {source!r}\n"
                "Check that your webcam is connected and not in use."
            )

        # Log actual resolution & FPS
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        print(f"[CameraSource] Opened {source!r}  {w}×{h}  {fps:.1f} fps")

    def read(self):
        """Return (success: bool, frame: np.ndarray | None)."""
        return self._cap.read()

    def release(self):
        """Release the capture device."""
        self._cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()
