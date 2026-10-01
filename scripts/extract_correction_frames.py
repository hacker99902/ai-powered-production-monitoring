import cv2
import os

video_path = "data/videos/test/new.mp4"
output_dir = "data/dataset/raw/conveyor_correction_frames"

os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

print("Total frames:", total_frames)
print("FPS:", fps)

# Extract every 8th frame
frame_numbers = range(0, total_frames, 8)

for frame_no in frame_numbers:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)

    ret, frame = cap.read()

    if not ret:
        continue

    filename = os.path.join(
        output_dir,
        f"frame_{frame_no:04d}.jpg"
    )

    cv2.imwrite(filename, frame)

cap.release()

print("Frames saved to:", output_dir)