
import cv2

from cv.production.detector import ProductDetector
from cv.production.tracker import TrackProcessor
from cv.production.counter import LineCounter


class ProductionPipeline:
    def __init__(
        self,
        model_path,
        conf=0.20,
        imgsz=480,
        device=0,
        window_seconds=60,
        fps=30,
        line_orientation="horizontal",
        line_position=0.5,
        detector=None
    ):
        # Validate line configuration
        if line_orientation not in ("horizontal", "vertical"):
            raise ValueError(
                "line_orientation must be 'horizontal' or 'vertical'"
            )

        if not 0.0 < line_position < 1.0:
            raise ValueError(
                "line_position must be between 0 and 1"
            )

        # Initialize detector
        if detector is not None:
            self.detector = detector
        else:
            self.detector = ProductDetector(
                model_path=model_path,
                conf=conf,
                imgsz=imgsz,
                device=device
            )

        # Independent tracker for this pipeline/camera
        self.track_processor = TrackProcessor()

        # Configuration
        self.model_path = model_path
        self.window_seconds = window_seconds
        self.fps = fps
        self.line_orientation = line_orientation
        self.line_position = line_position

        # Counting components
        self.counter = None
        self.line_x = None
        self.line_y = None
        self.line_start = None
        self.line_end = None
        self.width = None
        self.height = None

        # Throughput timestamps by class
        self.crossing_times = {}

        # Frame tracking
        self.frame_no = 0

        # Get class names from the trained YOLO model
        if isinstance(self.detector.names, dict):
            self.class_names = [
                self.detector.names[i]
                for i in sorted(self.detector.names)
            ]
        else:
            self.class_names = list(self.detector.names)

    # --------------------------------------------------
    # INITIALIZE COUNTING LINE
    # --------------------------------------------------

    def _initialize_counter(self, frame):
        """Initialize horizontal or vertical counting line."""
        self.height, self.width = frame.shape[:2]

        if self.line_orientation == "horizontal":
            self.line_y = int(self.height * self.line_position)

            self.line_start = (0, self.line_y)
            self.line_end = (self.width - 1, self.line_y)

        else:
            self.line_x = int(self.width * self.line_position)

            self.line_start = (self.line_x, 0)
            self.line_end = (self.line_x, self.height - 1)

        self.counter = LineCounter(
            line_start=self.line_start,
            line_end=self.line_end,
            direction="any"
        )

    # --------------------------------------------------
    # THROUGHPUT
    # --------------------------------------------------

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

    # --------------------------------------------------
    # DRAW COUNTING LINE
    # --------------------------------------------------

    def _draw_counting_line(self, frame):
        """Draw the configured counting line."""
        cv2.line(
            frame,
            self.line_start,
            self.line_end,
            (0, 0, 255),
            3
        )

    # --------------------------------------------------
    # DISPLAY COUNTS
    # --------------------------------------------------

    def _draw_statistics(self, frame, counts, rates):
        """Draw class counts and throughput on the frame."""
        y = 35

        for class_name in self.class_names:
            count = counts.get(class_name, 0)
            rate = rates.get(class_name, 0.0)

            cv2.putText(
                frame,
                f"{class_name}: {count}",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2
            )
            y += 27

            cv2.putText(
                frame,
                f"Rate: {rate:.1f} products/min",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1
            )
            y += 33

    # --------------------------------------------------
    # PROCESS FRAME
    # --------------------------------------------------

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

        # 1. Detect objects
        results = self.detector.detect(frame)

        # 2. Assign independent track IDs
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

            label = (
                f"{class_name} "
                f"ID:{track_id} "
                f"{confidence:.2f}"
            )

            cv2.putText(
                frame,
                label,
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
        self._draw_counting_line(frame)

        # 7. Draw statistics
        self._draw_statistics(frame, counts, rates)

        return frame, counts, rates, newly_counted

    # --------------------------------------------------
    # GET COUNTS
    # --------------------------------------------------

    def get_counts(self):
        """Return cumulative class-wise counts."""
        if self.counter is None:
            return {}

        return self.counter.get_counts()

    # --------------------------------------------------
    # RESET
    # --------------------------------------------------

    def reset(self):
        """Reset counter, throughput and independent tracking state."""
        if self.counter is not None:
            self.counter.reset()

        self.crossing_times.clear()
        self.frame_no = 0

        # Reset tracker without reloading YOLO
        self.track_processor.reset()

        # Recreate counter on next frame
        self.counter = None
        self.line_x = None
        self.line_y = None
        self.line_start = None
        self.line_end = None
        self.width = None
        self.height = None
