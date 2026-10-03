
class TrackProcessor:
    """
    Lightweight IoU-based tracker.
    Keep one instance per camera to isolate tracking IDs.
    """

    def __init__(self, iou_threshold=0.3, max_lost=20):
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self.next_id = 1
        self.tracks = {}

    @staticmethod
    def _iou(box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        x1 = max(ax1, bx1)
        y1 = max(ay1, by1)
        x2 = min(ax2, bx2)
        y2 = min(ay2, by2)

        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

        union = area_a + area_b - intersection
        return intersection / union if union > 0 else 0.0

    def process(self, results, class_names):
        # The detector returns a list containing one Results object.
        if not results:
            detections = []
        else:
            result = results[0]

            detections = []
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                classes = result.boxes.cls.int().cpu().tolist()
                confidences = result.boxes.conf.cpu().tolist()

                for bbox, class_id, confidence in zip(
                    boxes, classes, confidences
                ):
                    class_name = (
                        class_names[class_id]
                        if not isinstance(class_names, dict)
                        else class_names[class_id]
                    )

                    detections.append({
                        "bbox": bbox.tolist(),
                        "class_name": class_name,
                        "confidence": float(confidence)
                    })

        # Age existing tracks before matching.
        for track in self.tracks.values():
            track["missed"] += 1

        # Build all valid detection-to-track matches.
        candidates = []

        for det_index, detection in enumerate(detections):
            for track_id, track in self.tracks.items():
                if detection["class_name"] != track["class_name"]:
                    continue

                overlap = self._iou(
                    detection["bbox"],
                    track["bbox"]
                )

                if overlap >= self.iou_threshold:
                    candidates.append(
                        (overlap, det_index, track_id)
                    )

        # Greedily assign the highest-overlap matches.
        candidates.sort(reverse=True)

        matched_detections = set()
        matched_tracks = set()
        assigned = {}

        for overlap, det_index, track_id in candidates:
            if det_index in matched_detections:
                continue
            if track_id in matched_tracks:
                continue

            matched_detections.add(det_index)
            matched_tracks.add(track_id)
            assigned[det_index] = track_id

        # Update matched tracks and create new ones.
        output = []

        for det_index, detection in enumerate(detections):
            if det_index in assigned:
                track_id = assigned[det_index]
                self.tracks[track_id]["bbox"] = detection["bbox"]
                self.tracks[track_id]["missed"] = 0
            else:
                track_id = self.next_id
                self.next_id += 1

                self.tracks[track_id] = {
                    "bbox": detection["bbox"],
                    "class_name": detection["class_name"],
                    "missed": 0
                }

            output.append({
                "track_id": track_id,
                "class_name": detection["class_name"],
                "bbox": detection["bbox"],
                "confidence": detection["confidence"]
            })

        # Remove tracks not seen for too many frames.
        expired = [
            track_id
            for track_id, track in self.tracks.items()
            if track["missed"] > self.max_lost
        ]

        for track_id in expired:
            del self.tracks[track_id]

        return output

    def reset(self):
        self.next_id = 1
        self.tracks.clear()

