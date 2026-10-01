
import cv2
from ultralytics import YOLO
from cv.production.occupancy import OccupancyCounter

MODEL_PATH = "runs/detect/runs/detect/production_v4_synthetic_spark-2/weights/best.pt"
VIDEO_PATH = "data/videos/test/new_video.mp4"

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Central monitoring area
roi = [
    (160, 80),
    (1120, 80),
    (1120, 640),
    (160, 640)
]

occupancy = OccupancyCounter(roi)

while True:
    success, frame = cap.read()
    if not success:
        break

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.20,
        imgsz=1280,
        device=0,
        verbose=False
    )

    tracks = []

    for result in results:
        if result.boxes is None or result.boxes.id is None:
            continue

        boxes = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id.int().cpu().tolist()
        classes = result.boxes.cls.int().cpu().tolist()

        for bbox, track_id, class_id in zip(boxes, ids, classes):
            class_name = model.names[class_id]
            tracks.append({
                "track_id": track_id,
                "class_name": class_name,
                "bbox": bbox.tolist()
            })

            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{class_name} ID:{track_id}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

    counts = occupancy.update(tracks)

    # Draw ROI
    for i in range(len(roi)):
        cv2.line(
            frame,
            roi[i],
            roi[(i + 1) % len(roi)],
            (0, 0, 255),
            3
        )

    # Display occupancy
    cv2.putText(
        frame,
        f"Total Occupancy: {occupancy.get_total()}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    y = 70
    for class_name in ["brake_pad", "bearing", "spark_plug", "gear"]:
        cv2.putText(
            frame,
            f"{class_name}: {counts.get(class_name, 0)}",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )
        y += 30

    cv2.imshow("Occupancy Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()