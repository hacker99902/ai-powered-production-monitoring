from pathlib import Path
import random
import cv2
import matplotlib.pyplot as plt


# -----------------------------
# Configuration
# -----------------------------
DATASET = Path("data/dataset/production_yolo_final")
SPLIT = "train"

IMAGE_DIR = DATASET / "images" / SPLIT
LABEL_DIR = DATASET / "labels" / SPLIT

CLASS_NAMES = {
    0: "brake_pad",
    1: "bearing",
    2: "spark_plug",
    3: "gear"
}

SAMPLES_PER_CLASS = 3


# -----------------------------
# Find images containing class
# -----------------------------
class_images = {class_id: [] for class_id in CLASS_NAMES}

for label_file in LABEL_DIR.glob("*.txt"):

    classes_in_image = set()

    with open(label_file, "r") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) != 5:
                continue

            class_id = int(parts[0])
            classes_in_image.add(class_id)

    for class_id in classes_in_image:

        if class_id in class_images:

            image_file = IMAGE_DIR / (label_file.stem + ".jpg")

            if not image_file.exists():
                image_file = IMAGE_DIR / (label_file.stem + ".png")

            if image_file.exists():
                class_images[class_id].append(image_file)


# -----------------------------
# Select samples
# -----------------------------
selected_images = []

for class_id, images in class_images.items():

    random.shuffle(images)

    selected = images[:SAMPLES_PER_CLASS]

    for image in selected:
        selected_images.append((class_id, image))


# -----------------------------
# Draw bounding boxes
# -----------------------------
fig, axes = plt.subplots(
    3,
    4,
    figsize=(16, 12)
)

axes = axes.flatten()

plot_index = 0

for target_class, image_path in selected_images:

    image = cv2.imread(str(image_path))

    if image is None:
        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    height, width = image.shape[:2]

    label_path = LABEL_DIR / (image_path.stem + ".txt")

    with open(label_path, "r") as f:

        for line in f:

            parts = line.strip().split()

            if len(parts) != 5:
                continue

            class_id = int(parts[0])

            x_center = float(parts[1])
            y_center = float(parts[2])
            box_width = float(parts[3])
            box_height = float(parts[4])

            # YOLO normalized coordinates → pixels
            x1 = int((x_center - box_width / 2) * width)
            y1 = int((y_center - box_height / 2) * height)

            x2 = int((x_center + box_width / 2) * width)
            y2 = int((y_center + box_height / 2) * height)

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )

            cv2.putText(
                image,
                CLASS_NAMES[class_id],
                (x1, max(y1 - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )

    axes[plot_index].imshow(image)
    axes[plot_index].set_title(
        f"Target: {CLASS_NAMES[target_class]}\n{image_path.name}"
    )
    axes[plot_index].axis("off")

    plot_index += 1


# Hide unused plots
for i in range(plot_index, len(axes)):
    axes[i].axis("off")


plt.tight_layout()

output_dir = DATASET / "preview"
output_dir.mkdir(exist_ok=True)

output_file = output_dir / "dataset_preview.jpg"

plt.savefig(output_file, dpi=150)

print()
print("Preview created:")
print(output_file)
print()
print("Check whether every bounding box is correctly placed.")