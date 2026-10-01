
from collections import defaultdict

class LineCounter:
    """
    Generic line-crossing counter.

    Counts each tracked object once when it crosses
    the virtual line and records newly counted objects
    for throughput calculation.
    """

    def __init__(self, line_start, line_end, direction="any"):
        self.line_start = line_start
        self.line_end = line_end
        self.direction = direction

        # Track IDs that have already been counted
        self.counted_ids = set()

        # Total count for each product class
        self.counts = defaultdict(int)

        # Previous center position of each tracked object
        self.previous_positions = {}

        # Objects counted in the current update only
        self.newly_counted = []

    # ========================================================
    # LINE SIDE
    # ========================================================

    def _get_side(self, point):
        px, py = point

        x1, y1 = self.line_start
        x2, y2 = self.line_end

        value = (
            (x2 - x1) * (py - y1)
            - (y2 - y1) * (px - x1)
        )

        return value

    # ========================================================
    # CHECK CROSSING
    # ========================================================

    def _has_crossed(self, previous_point, current_point):
        previous_side = self._get_side(previous_point)
        current_side = self._get_side(current_point)

        crossed = previous_side * current_side < 0

        if not crossed:
            return False

        if self.direction == "any":
            return True

        if self.direction == "positive_to_negative":
            return previous_side > 0 and current_side < 0

        if self.direction == "negative_to_positive":
            return previous_side < 0 and current_side > 0

        raise ValueError(
            "Invalid direction. Use 'any', "
            "'positive_to_negative', or "
            "'negative_to_positive'."
        )

    # ========================================================
    # UPDATE
    # ========================================================

    def update(self, tracks):
        """
        Update the counter using tracked objects.

        newly_counted is cleared at the start of every
        update and contains only the objects counted
        during this frame.
        """

        self.newly_counted = []

        for track in tracks:
            track_id = track["track_id"]
            class_name = track["class_name"]
            bbox = track["bbox"]

            x1, y1, x2, y2 = bbox

            # Calculate object center
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            current_position = (center_x, center_y)

            # First time seeing this track
            if track_id not in self.previous_positions:
                self.previous_positions[track_id] = current_position
                continue

            previous_position = self.previous_positions[track_id]

            # Check line crossing
            crossed = self._has_crossed(
                previous_position,
                current_position
            )

            # Count each track only once
            if crossed and track_id not in self.counted_ids:
                self.counted_ids.add(track_id)
                self.counts[class_name] += 1

                # Record this crossing for throughput
                self.newly_counted.append(class_name)

                print(
                    f"COUNTED | "
                    f"ID: {track_id} | "
                    f"Class: {class_name} | "
                    f"Total: {self.counts[class_name]}"
                )

            # Save current position
            self.previous_positions[track_id] = current_position

    # ========================================================
    # GET COUNTS
    # ========================================================

    def get_counts(self):
        """Return cumulative counts for each class."""
        return dict(self.counts)

    # ========================================================
    # GET COUNT FOR ONE CLASS
    # ========================================================

    def get_count(self, class_name):
        """Return count for a specific class."""
        return self.counts.get(class_name, 0)

    # ========================================================
    # GET NEWLY COUNTED
    # ========================================================

    def get_newly_counted(self):
        """
        Return class names of objects counted
        in the most recent update only.
        """
        return self.newly_counted.copy()

    # ========================================================
    # RESET
    # ========================================================

    def reset(self):
        """Reset all counting state."""
        self.counted_ids.clear()
        self.counts.clear()
        self.previous_positions.clear()
        self.newly_counted.clear()