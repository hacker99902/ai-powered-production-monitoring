from pathlib import Path
import random
import cv2


# ============================================================
# PATHS
# ============================================================

ROOT = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "dataset"
    / "production_yolo_v2"
)

OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "dataset"
    / "v2_preview"
)


CLASS_NAMES = {
    0: "brake_pad",
    1: "bearing",
    2: "spark_plug",
    3: "gear",
}


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# DRAW BOXES
# ============================================================

def draw_labels(image, label_file):

    height, width = image.shape[:2]

    lines = label_file.read_text(
        encoding="utf-8"
    ).splitlines()

    for line in lines:

        parts = line.strip().split()

        if len(parts) != 5:
            continue

        class_id = int(parts[0])

        x_center = float(parts[1])
        y_center = float(parts[2])
        box_width = float(parts[3])
        box_height = float(parts[4])

        # Convert YOLO coordinates to pixels

        x1 = int(
            (x_center - box_width / 2) * width
        )

        y1 = int(
            (y_center - box_height / 2) * height
        )

        x2 = int(
            (x_center + box_width / 2) * width
        )

        y2 = int(
            (y_center + box_height / 2) * height
        )

        # Keep coordinates inside image

        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))

        name = CLASS_NAMES.get(
            class_id,
            f"class_{class_id}"
        )

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2,
        )

        cv2.putText(
            image,
            name,
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    return image


# ============================================================
# MAIN
# ============================================================

def main():

    random.seed(42)

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # We will preview the TRAINING data.

    image_dir = ROOT / "train" / "images"
    label_dir = ROOT / "train" / "labels"

    images = [
        p
        for p in image_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    # Select 40 random images

    selected = random.sample(
        images,
        min(40, len(images)),
    )

    print("=" * 60)
    print("PRODUCTION YOLO V2 VISUALIZATION")
    print("=" * 60)

    print("Images available:", len(images))
    print("Images selected:", len(selected))

    for index, image_path in enumerate(selected):

        label_path = (
            label_dir
            / f"{image_path.stem}.txt"
        )

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            continue

        if label_path.exists():

            image = draw_labels(
                image,
                label_path,
            )

        output_path = (
            OUTPUT
            / f"preview_{index:03d}.jpg"
        )

        cv2.imwrite(
            str(output_path),
            image,
        )

    print("\nPreview created at:")
    print(OUTPUT)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()