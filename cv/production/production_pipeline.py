
import cv2

from cv.production.detector import ProductDetector
from cv.production.tracker import TrackProcessor
from cv.production.counter import LineCounter


class ProductionPipeline:
    def __init__(
        self,
        model_path,
        conf=0.20,
        imgsz=1280,
        device=0,
        window_seconds=60,
        fps=30
    ):
        # Initialize detector and tracker
        self.detector = ProductDetector(
            model_path=model_path,
            conf=conf,
            imgsz=imgsz,
            device=device
        )

        self.track_processor = TrackProcessor()

        # Configuration
        self.model_path = model_path
        self.window_seconds = window_seconds
        self.fps = fps

        # Counting components
        self.counter = None
        self.line_y = None
        self.width = None
        self.height = None

        # Throughput timestamps by class
        self.crossing_times = {}

        # Frame tracking
        self.frame_no = 0

        self.class_names = [
            "brake_pad",
            "bearing",
            "spark_plug",
            "gear"
        ]

    def _initialize_counter(self, frame):
        """Initialize counting line using frame dimensions."""
        self.height, self.width = frame.shape[:2]
        self.line_y = self.height // 2

        self.counter = LineCounter(
            line_start=(0, self.line_y),
            line_end=(self.width - 1, self.line_y),
            direction="any"
        )

    def _calculate_throughput(self, newly_counted, timestamp):
        """Calculate rolling throughput in products per minute."""
        for class_name in newly_counted:
            self.crossing_times.setdefault(
                class_name, []
            ).append(timestamp)

        rates = {}

        for class_name in self.class_names:
            timestamps = self.crossing_times.get(class_name, [])

            recent = [
                t for t in timestamps
                if 0 <= timestamp - t <= self.window_seconds
            ]

            self.crossing_times[class_name] = recent

            rates[class_name] = (
                len(recent) * 60 / self.window_seconds
            )

        return rates

    def process_frame(self, frame, timestamp=None):
        """
        Process one video frame.

        Returns:
            annotated_frame, counts, rates, newly_counted
        """
        if frame is None:
            raise ValueError("Input frame is empty.")

        if self.counter is None:
            self._initialize_counter(frame)

        if timestamp is None:
            timestamp = self.frame_no / self.fps

        self.frame_no += 1

        # 1. Detect objects and track them
        results = self.detector.detect(frame)

        # 2. Convert results into standardized track data
        tracks = self.track_processor.process(
            results,
            self.detector.names
        )

        # 3. Draw bounding boxes and tracking labels
        for track in tracks:
            x1, y1, x2, y2 = map(int, track["bbox"])
            class_name = track["class_name"]
            track_id = track["track_id"]
            confidence = track["confidence"]

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"{class_name} ID:{track_id} {confidence:.2f}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )

        # 4. Update line counter
        self.counter.update(tracks)

        newly_counted = self.counter.get_newly_counted()
        counts = self.counter.get_counts()

        # 5. Calculate throughput
        rates = self._calculate_throughput(
            newly_counted,
            timestamp
        )

        # 6. Draw counting line
        cv2.line(
            frame,
            (0, self.line_y),
            (self.width - 1, self.line_y),
            (0, 0, 255),
            3
        )

        # 7. Display counts and throughput
        y = 35

        for class_name in self.class_names:
            count = counts.get(class_name, 0)
            rate = rates.get(class_name, 0.0)

            cv2.putText(
                frame,
                f"{class_name}: {count}",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )
            y += 28

            cv2.putText(
                frame,
                f"Rate: {rate:.1f} products/min",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
            y += 35

        return frame, counts, rates, newly_counted

    def get_counts(self):
        """Return cumulative class-wise counts."""
        if self.counter is None:
            return {}
        return self.counter.get_counts()

    def reset(self):
        """Reset counting, throughput and tracking for a new video."""
        if self.counter is not None:
            self.counter.reset()

        self.crossing_times.clear()
        self.frame_no = 0

        # Reload detector to clear its tracking state
        self.detector.reset()

        # Recreate counter on the next frame
        self.counter = None
        self.line_y = None
        self.width = None
        self.height = None