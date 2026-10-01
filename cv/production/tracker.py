
class TrackProcessor:
    def process(self, results, class_names):
        """
        Convert YOLO tracking results into a list
        of standardized track dictionaries.
        """
        tracks = []

        for result in results:
            if result.boxes is None or result.boxes.id is None:
                continue

            boxes = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.int().cpu().tolist()
            classes = result.boxes.cls.int().cpu().tolist()
            confidences = result.boxes.conf.cpu().tolist()

            for bbox, track_id, class_id, confidence in zip(
                boxes, ids, classes, confidences
            ):
                class_name = class_names[class_id]

                tracks.append({
                    "track_id": track_id,
                    "class_name": class_name,
                    "bbox": bbox.tolist(),
                    "confidence": float(confidence)
                })

        return tracks
    