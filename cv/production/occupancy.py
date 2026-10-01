
class OccupancyCounter:
    def __init__(self, roi):
        """
        roi: list of points defining the region,
             e.g. [(160, 80), (1120, 80),
                   (1120, 640), (160, 640)]
        """
        self.roi = roi
        self.current_objects = {}

    def _is_inside(self, point):
        """Check whether a point is inside the ROI polygon."""
        import cv2
        import numpy as np

        polygon = np.array(self.roi, dtype=np.int32)
        result = cv2.pointPolygonTest(
            polygon,
            (float(point[0]), float(point[1])),
            False
        )
        return result >= 0

    def update(self, tracks):
        """
        tracks: list of tracked objects containing:
          track_id, class_name, bbox
        """
        current = {}

        for track in tracks:
            track_id = track["track_id"]
            class_name = track["class_name"]
            x1, y1, x2, y2 = track["bbox"]

            # Calculate the center of the bounding box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            # Count only objects inside the ROI
            if self._is_inside((center_x, center_y)):
                current[track_id] = class_name

        # Current objects inside the ROI
        self.current_objects = current

        return self.get_occupancy()

    def get_occupancy(self):
        """Return current occupancy count by class."""
        counts = {}

        for class_name in self.current_objects.values():
            counts[class_name] = counts.get(class_name, 0) + 1

        return counts

    def get_total(self):
        """Return total number of objects inside the ROI."""
        return len(self.current_objects)

    def reset(self):
        """Clear current occupancy."""
        self.current_objects.clear()