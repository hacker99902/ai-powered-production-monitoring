
import cv2

from cv.production.production_pipeline import ProductionPipeline

MODEL_PATH = "runs/detect/runs/detect/production_v4_synthetic_spark-2/weights/best.pt"
VIDEO_PATH = "data/videos/test/new_video.mp4"
OUTPUT_PATH = "data/videos/test/new_test_counted.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30

pipeline = ProductionPipeline(
    model_path=MODEL_PATH,
    conf=0.20,
    imgsz=1280,
    device=0,
    window_seconds=60,
    fps=fps
)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    fps,
    (width, height)
)

while True:
    success, frame = cap.read()

    if not success:
        break

    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

    annotated, counts, rates, newly_counted = (
        pipeline.process_frame(frame, timestamp)
    )

    cv2.imshow("Production Pipeline", annotated)
    writer.write(annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()

print("Final counts:", pipeline.get_counts())
print("Saved video:", OUTPUT_PATH)