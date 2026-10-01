
from pathlib import Path
import shutil

# Project paths
source_dir = Path("data/dataset/raw/conveyor_correction_frames")
output_dir = Path("data/dataset/raw/conveyor_spark_correction")

images_dir = output_dir / "images"
labels_dir = output_dir / "labels"

images_dir.mkdir(parents=True, exist_ok=True)
labels_dir.mkdir(parents=True, exist_ok=True)

# Selected frames with multiple spark plugs
selected_frames = [
    40, 48, 56, 64, 72, 80, 88,
    128, 136, 144
]

copied = 0

for frame_no in selected_frames:
    filename = f"frame_{frame_no:04d}.jpg"
    source = source_dir / filename
    destination = images_dir / filename

    if not source.exists():
        print(f"Missing: {filename}")
        continue

    shutil.copy2(source, destination)

    # Create an empty label file for manual annotation
    label_file = labels_dir / f"frame_{frame_no:04d}.txt"
    label_file.touch(exist_ok=True)

    copied += 1
    print(f"Prepared: {filename}")

print(f"\nPrepared {copied} images.")
print(f"Images: {images_dir}")
print(f"Labels: {labels_dir}")
