
from ultralytics import YOLO


class ProductDetector:
    def __init__(
        self,
        model_path,
        conf=0.20,
        imgsz=1280,
        device=0
    ):
        self.model_path = model_path
        self.model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz
        self.device = device

    @property
    def names(self):
        return self.model.names

    def detect(self, frame):
        """
        Run YOLO detection and ByteTrack tracking
        on one frame.
        """
        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False
        )

        return results

    def reset(self):
        """Reload the model to reset tracking state."""
        self.model = YOLO(self.model_path)