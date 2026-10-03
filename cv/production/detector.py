
from ultralytics import YOLO


class ProductDetector:
    def __init__(
        self,
        model_path,
        conf=0.20,
        imgsz=640,
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
        results = self.model.predict(
            source=frame,
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False
        )
        return results

    def detect_many(self, frames):
        """
        Run detection on multiple video frames using
        one shared YOLO model.

        Tracking will be handled separately for each
        video in the next pipeline step.
        """
        if not frames:
            return []

        results_per_frame = []

        for frame in frames:
            if frame is None:
                raise ValueError("One of the input frames is empty.")

            results = self.model.predict(
                source=frame,
                conf=self.conf,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False
            )

            results_per_frame.append(results[0])

        return results_per_frame

    def reset(self):
        """
        Reload the model to clear tracking state
        for the legacy single-video track method.
        """
        self.model = YOLO(self.model_path)