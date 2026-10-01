from pathlib import Path
from collections import defaultdict
from PIL import Image


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ============================================================
# DATASET CONFIGURATION
# ============================================================

DATASETS = {
    "SPARK PLUG": {
        "root": (
            PROJECT_ROOT
            / "data"
            / "dataset"
            / "raw"
            / "spark_plugs"
            / "enginepartsdetection.v1i.yolov8"
        ),

        # From data.yaml:
        # 10 = Spark plug
        "target_class_id": 10,
        "target_class_name": "spark_plug",
    },

    "BRAKE PAD": {
        "root": (
            PROJECT_ROOT
            / "data"
            / "dataset"
            / "raw"
            / "brake_pads"
            / "Brake Pads x.v2i.yolov8"
        ),

        # From data.yaml:
        # 0 = BrakePad
        "target_class_id": 0,
        "target_class_name": "brake_pad",
    },
}


# ============================================================
# IMAGE EXTENSIONS
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# FULL IMAGE THRESHOLD
# ============================================================

# A box is considered full-image if both width and height
# are effectively 1.0.
FULL_IMAGE_THRESHOLD = 0.999


# ============================================================
# FIND IMAGE
# ============================================================

def find_image(label_path, image_dir):

    for extension in IMAGE_EXTENSIONS:

        image_path = (
            image_dir
            / f"{label_path.stem}{extension}"
        )

        if image_path.exists():
            return image_path

    return None


# ============================================================
# VALIDATE IMAGE
# ============================================================

def validate_image(image_path):

    try:

        with Image.open(image_path) as img:

            width, height = img.size

            if width <= 0 or height <= 0:
                return False, None

            return True, (width, height)

    except Exception:

        return False, None


# ============================================================
# CHECK YOLO BOX
# ============================================================

def parse_yolo_line(line):

    parts = line.split()

    if len(parts) != 5:

        return None

    try:

        class_id = int(parts[0])

        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])

    except ValueError:

        return None

    # -----------------------------------------------
    # YOLO coordinate validation
    # -----------------------------------------------

    if not (
        0 <= x_center <= 1
        and 0 <= y_center <= 1
        and 0 < width <= 1
        and 0 < height <= 1
    ):

        return None

    return {
        "class_id": class_id,
        "x": x_center,
        "y": y_center,
        "width": width,
        "height": height,
    }


# ============================================================
# AUDIT ONE DATASET
# ============================================================

def audit_dataset(dataset_name, config):

    root = config["root"]

    target_class_id = config["target_class_id"]
    target_class_name = config["target_class_name"]

    print("\n")
    print("=" * 80)
    print(dataset_name)
    print("=" * 80)

    print(f"\nDataset:")
    print(root)

    print(
        f"\nTarget class:"
        f"\n  ID   : {target_class_id}"
        f"\n  Name : {target_class_name}"
    )

    # --------------------------------------------------------
    # Check root
    # --------------------------------------------------------

    if not root.exists():

        print(
            "\nERROR: Dataset folder does not exist."
        )

        return

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    stats = {

        "images": 0,
        "label_files": 0,

        "target_images": 0,
        "target_objects": 0,

        "all_objects": 0,

        "normal_target_boxes": 0,
        "full_image_target_boxes": 0,

        "invalid_lines": 0,
        "invalid_images": 0,

        "missing_images": 0,

        "target_widths": [],
        "target_heights": [],

    }

    split_stats = defaultdict(
        lambda: {
            "images": 0,
            "labels": 0,
            "target_images": 0,
            "target_objects": 0,
            "normal_boxes": 0,
            "full_boxes": 0,
        }
    )

    # ========================================================
    # PROCESS TRAIN / VALID / TEST
    # ========================================================

    for split in [
        "train",
        "valid",
        "val",
        "test",
    ]:

        image_dir = root / split / "images"
        label_dir = root / split / "labels"

        # Some datasets may not have this split
        if not label_dir.exists():
            continue

        label_files = list(
            label_dir.glob("*.txt")
        )

        print(
            f"\nScanning {split}: "
            f"{len(label_files)} label files"
        )

        for index, label_path in enumerate(
            label_files,
            start=1
        ):

            if index % 100 == 0:

                print(
                    f"  {index}/{len(label_files)}",
                    end="\r"
                )

            stats["label_files"] += 1

            split_stats[split]["labels"] += 1

            # ------------------------------------------------
            # Find corresponding image
            # ------------------------------------------------

            image_path = find_image(
                label_path,
                image_dir
            )

            if image_path is None:

                stats["missing_images"] += 1

                continue

            # ------------------------------------------------
            # Validate image
            # ------------------------------------------------

            valid, image_size = validate_image(
                image_path
            )

            if not valid:

                stats["invalid_images"] += 1

                continue

            stats["images"] += 1

            split_stats[split]["images"] += 1

            # ------------------------------------------------
            # Read labels
            # ------------------------------------------------

            try:

                with open(
                    label_path,
                    "r",
                    encoding="utf-8"
                ) as f:

                    lines = [
                        line.strip()
                        for line in f
                        if line.strip()
                    ]

            except Exception:

                continue

            target_found_in_image = False

            for line in lines:

                box = parse_yolo_line(line)

                if box is None:

                    stats["invalid_lines"] += 1

                    continue

                class_id = box["class_id"]

                stats["all_objects"] += 1

                # ------------------------------------------------
                # Only inspect target class
                # ------------------------------------------------

                if class_id != target_class_id:

                    continue

                target_found_in_image = True

                stats["target_objects"] += 1

                split_stats[split][
                    "target_objects"
                ] += 1

                width = box["width"]
                height = box["height"]

                stats["target_widths"].append(
                    width
                )

                stats["target_heights"].append(
                    height
                )

                # ------------------------------------------------
                # Full-image detection
                # ------------------------------------------------

                is_full_image = (
                    width >= FULL_IMAGE_THRESHOLD
                    and height >= FULL_IMAGE_THRESHOLD
                )

                if is_full_image:

                    stats[
                        "full_image_target_boxes"
                    ] += 1

                    split_stats[split][
                        "full_boxes"
                    ] += 1

                else:

                    stats[
                        "normal_target_boxes"
                    ] += 1

                    split_stats[split][
                        "normal_boxes"
                    ] += 1

            # ------------------------------------------------
            # Target class present in image
            # ------------------------------------------------

            if target_found_in_image:

                stats["target_images"] += 1

                split_stats[split][
                    "target_images"
                ] += 1

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\n")

    print("-" * 80)
    print("OVERALL RESULTS")
    print("-" * 80)

    print(
        f"\nImages found:                 "
        f"{stats['images']}"
    )

    print(
        f"Label files found:            "
        f"{stats['label_files']}"
    )

    print(
        f"Target-class images:          "
        f"{stats['target_images']}"
    )

    print(
        f"Target-class objects:         "
        f"{stats['target_objects']}"
    )

    print(
        f"All objects in dataset:       "
        f"{stats['all_objects']}"
    )

    print(
        f"\nNormal target boxes:          "
        f"{stats['normal_target_boxes']}"
    )

    print(
        f"Full-image target boxes:      "
        f"{stats['full_image_target_boxes']}"
    )

    print(
        f"\nInvalid annotation lines:     "
        f"{stats['invalid_lines']}"
    )

    print(
        f"Missing images:               "
        f"{stats['missing_images']}"
    )

    print(
        f"Invalid images:               "
        f"{stats['invalid_images']}"
    )

    # ========================================================
    # FULL IMAGE PERCENTAGE
    # ========================================================

    target_objects = stats["target_objects"]
    full_boxes = stats["full_image_target_boxes"]

    if target_objects > 0:

        full_percentage = (
            full_boxes
            / target_objects
            * 100
        )

    else:

        full_percentage = 0

    print(
        f"\nFull-image percentage:        "
        f"{full_percentage:.2f}%"
    )

    # ========================================================
    # BOUNDING BOX SIZE
    # ========================================================

    widths = stats["target_widths"]
    heights = stats["target_heights"]

    if widths:

        print(
            "\nTarget bounding-box statistics:"
        )

        print(
            f"  Width  min:    {min(widths):.4f}"
        )

        print(
            f"  Width  max:    {max(widths):.4f}"
        )

        print(
            f"  Width  avg:    "
            f"{sum(widths) / len(widths):.4f}"
        )

        print(
            f"  Height min:    {min(heights):.4f}"
        )

        print(
            f"  Height max:    {max(heights):.4f}"
        )

        print(
            f"  Height avg:    "
            f"{sum(heights) / len(heights):.4f}"
        )

    # ========================================================
    # SPLIT RESULTS
    # ========================================================

    print("\n")
    print("-" * 80)
    print("SPLIT RESULTS")
    print("-" * 80)

    print(
        f"\n{'Split':<12}"
        f"{'Images':>10}"
        f"{'Labels':>10}"
        f"{'Target Img':>14}"
        f"{'Objects':>12}"
        f"{'Normal':>12}"
        f"{'Full':>10}"
    )

    print("-" * 80)

    for split in [
        "train",
        "valid",
        "val",
        "test",
    ]:

        if split not in split_stats:
            continue

        s = split_stats[split]

        print(
            f"{split:<12}"
            f"{s['images']:>10}"
            f"{s['labels']:>10}"
            f"{s['target_images']:>14}"
            f"{s['target_objects']:>12}"
            f"{s['normal_boxes']:>12}"
            f"{s['full_boxes']:>10}"
        )

    # ========================================================
    # QUALITY ASSESSMENT
    # ========================================================

    print("\n")
    print("-" * 80)
    print("INITIAL QUALITY CHECK")
    print("-" * 80)

    if stats["target_objects"] == 0:

        print(
            "\n❌ No target objects found."
        )

    elif full_percentage >= 50:

        print(
            f"\n⚠️ WARNING:"
            f"\n{full_percentage:.1f}% of target boxes "
            f"are full-image boxes."
        )

        print(
            "\nThis dataset should NOT be merged yet."
        )

    elif full_percentage >= 20:

        print(
            f"\n⚠️ CAUTION:"
            f"\n{full_percentage:.1f}% of target boxes "
            f"are full-image boxes."
        )

        print(
            "\nWe should inspect the annotations "
            "before merging."
        )

    else:

        print(
            f"\n✓ Full-image boxes: "
            f"{full_percentage:.1f}%"
        )

        print(
            "\nThe annotation statistics look "
            "reasonable so far."
        )

    # --------------------------------------------------------
    # Target image count
    # --------------------------------------------------------

    print(
        f"\nTarget images available: "
        f"{stats['target_images']}"
    )

    print(
        f"Target objects available: "
        f"{stats['target_objects']}"
    )

    print(
        "\nIMPORTANT:"
        "\nThese statistics do NOT prove that "
        "the annotations are correct."
    )

    print(
        "We should visually inspect samples "
        "before merging."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("NEW DATASET ANNOTATION AUDIT")
    print("=" * 80)

    print(
        "\nThis script ONLY READS the datasets."
    )

    print(
        "It does NOT modify, delete, or merge anything."
    )

    # --------------------------------------------------------
    # Audit datasets
    # --------------------------------------------------------

    for dataset_name, config in DATASETS.items():

        audit_dataset(
            dataset_name,
            config
        )

    # --------------------------------------------------------
    # Finish
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)

    print(
        "\nNo files were modified."
    )


if __name__ == "__main__":
    main()