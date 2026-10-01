from pathlib import Path
import shutil
import random

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE = (
    PROJECT_ROOT
    / "data"
    / "dataset"
    / "raw"
    / "brake_pads"
    / "Brake Pads x.v2i.yolov8"
)

OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "dataset"
    / "processed"
    / "brake_pads_clean"
)

RANDOM_SEED = 42

# Source classes
BRAKE_PAD_CLASS = 0

# Our project class mapping
PROJECT_BRAKE_PAD_CLASS = 0


def polygon_to_bbox(points):
    """
    Convert normalized polygon points into
    YOLO normalized bounding box.

    points = [(x1,y1), (x2,y2), ...]
    """

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    xmin = min(xs)
    xmax = max(xs)
    ymin = min(ys)
    ymax = max(ys)

    x_center = (xmin + xmax) / 2
    y_center = (ymin + ymax) / 2

    width = xmax - xmin
    height = ymax - ymin

    return x_center, y_center, width, height


def parse_label_file(label_file):
    """
    Read a label file and extract only BrakePad
    annotations.

    Supports:
    - normal YOLO box:
        class xc yc w h

    - polygon:
        class x1 y1 x2 y2 x3 y3 ...
    """

    brake_boxes = []

    if not label_file.exists():
        return brake_boxes

    with open(label_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line_number, line in enumerate(lines, start=1):

        line = line.strip()

        if not line:
            continue

        parts = line.split()

        try:
            values = [float(x) for x in parts]
        except ValueError:
            print(
                f"[WARNING] Non-numeric annotation: "
                f"{label_file} line {line_number}"
            )
            continue

        if len(values) < 5:
            print(
                f"[WARNING] Too few values: "
                f"{label_file} line {line_number}"
            )
            continue

        source_class = int(values[0])

        # We only need BrakePad
        if source_class != BRAKE_PAD_CLASS:
            continue

        coordinates = values[1:]

        # ------------------------------------------------
        # Normal YOLO bounding box
        # ------------------------------------------------
        if len(coordinates) == 4:

            xc, yc, width, height = coordinates

        # ------------------------------------------------
        # Polygon
        # ------------------------------------------------
        else:

            # Polygon must contain x,y pairs
            if len(coordinates) % 2 != 0:
                print(
                    f"[WARNING] Odd number of polygon coordinates: "
                    f"{label_file} line {line_number}"
                )
                continue

            points = []

            for i in range(0, len(coordinates), 2):

                x = coordinates[i]
                y = coordinates[i + 1]

                points.append((x, y))

            if len(points) < 3:
                print(
                    f"[WARNING] Polygon has fewer than 3 points: "
                    f"{label_file} line {line_number}"
                )
                continue

            xc, yc, width, height = polygon_to_bbox(points)

        # ------------------------------------------------
        # Validate normalized YOLO coordinates
        # ------------------------------------------------

        if not (
            0 <= xc <= 1
            and 0 <= yc <= 1
            and 0 < width <= 1
            and 0 < height <= 1
        ):
            print(
                f"[WARNING] Invalid bbox values: "
                f"{label_file} line {line_number}"
            )
            continue

        # Our final project class
        brake_boxes.append(
            f"{PROJECT_BRAKE_PAD_CLASS} "
            f"{xc:.6f} "
            f"{yc:.6f} "
            f"{width:.6f} "
            f"{height:.6f}"
        )

    return brake_boxes


def collect_images():

    samples = []

    for split in ["train", "valid", "test"]:

        image_dir = SOURCE / split / "images"
        label_dir = SOURCE / split / "labels"

        if not image_dir.exists():
            continue

        for image_path in image_dir.iterdir():

            if image_path.suffix.lower() not in [
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".webp",
            ]:
                continue

            label_path = label_dir / f"{image_path.stem}.txt"

            boxes = parse_label_file(label_path)

            # Keep image only if it contains BrakePad
            if boxes:
                samples.append(
                    {
                        "image": image_path,
                        "boxes": boxes,
                    }
                )

    return samples


def split_dataset(samples):

    random.seed(RANDOM_SEED)

    random.shuffle(samples)

    n = len(samples)

    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    return {
        "train": samples[:train_end],
        "valid": samples[train_end:val_end],
        "test": samples[val_end:],
    }


def copy_dataset(splits):

    for split_name, samples in splits.items():

        image_dir = OUTPUT / split_name / "images"
        label_dir = OUTPUT / split_name / "labels"

        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        for sample in samples:

            image_path = sample["image"]

            destination_image = image_dir / image_path.name

            shutil.copy2(
                image_path,
                destination_image,
            )

            destination_label = (
                label_dir
                / f"{image_path.stem}.txt"
            )

            with open(
                destination_label,
                "w",
                encoding="utf-8",
            ) as f:

                f.write(
                    "\n".join(sample["boxes"])
                )

                f.write("\n")


def write_yaml():

    yaml_path = OUTPUT / "data.yaml"

    yaml_content = f"""path: {OUTPUT.as_posix()}

train: train/images
val: valid/images
test: test/images

nc: 1

names:
  0: brake_pad
"""

    with open(
        yaml_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(yaml_content)


def main():

    print("=" * 70)
    print("BRAKE PAD DATASET CLEANING")
    print("=" * 70)

    print(f"\nSource:")
    print(SOURCE)

    print(f"\nOutput:")
    print(OUTPUT)

    samples = collect_images()

    print(
        f"\nBrake-pad images found: {len(samples)}"
    )

    if not samples:

        print(
            "\nERROR: No valid BrakePad images found."
        )
        return

    splits = split_dataset(samples)

    print("\nNew split:")

    for split_name, items in splits.items():

        print(
            f"  {split_name}: {len(items)} images"
        )

    if OUTPUT.exists():

        print(
            "\nOutput directory already exists."
        )

        print(
            "Delete it manually if you want to rebuild it."
        )

        return

    copy_dataset(splits)

    write_yaml()

    print("\n" + "=" * 70)
    print("CLEANING COMPLETE")
    print("=" * 70)

    print(
        f"\nClean dataset created at:\n{OUTPUT}"
    )

    print(
        "\nRaw dataset was NOT modified."
    )


if __name__ == "__main__":
    main()         