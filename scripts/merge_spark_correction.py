from pathlib import Path
import shutil


# ============================================================
# PATHS
# ============================================================

# Your correction dataset
CORRECTION_DIR = Path(
    "data/dataset/raw/conveyor_spark_plug_correction_dataset"
)

# Existing production dataset
PROJECT_ROOT = Path(
    "data/dataset/production_yolo_v2"
)

# Source
SOURCE_IMAGES = CORRECTION_DIR / "images"
SOURCE_LABELS = CORRECTION_DIR / "labels"

# Destination: TRAIN ONLY
TARGET_IMAGES = PROJECT_ROOT / "train" / "images"
TARGET_LABELS = PROJECT_ROOT / "train" / "labels"


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n==============================================")
    print("MERGING SPARK-PLUG CORRECTION DATASET")
    print("==============================================")

    # --------------------------------------------------------
    # 1. Check source directories
    # --------------------------------------------------------

    if not SOURCE_IMAGES.exists():
        raise FileNotFoundError(
            f"\nImages folder not found:\n{SOURCE_IMAGES}"
        )

    if not SOURCE_LABELS.exists():
        raise FileNotFoundError(
            f"\nLabels folder not found:\n{SOURCE_LABELS}"
        )

    # Create destination directories if necessary
    TARGET_IMAGES.mkdir(
        parents=True,
        exist_ok=True
    )

    TARGET_LABELS.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 2. Find images and labels
    # --------------------------------------------------------

    images = sorted(
        SOURCE_IMAGES.glob("*.jpg")
    )

    labels = sorted(
        SOURCE_LABELS.glob("*.txt")
    )

    print(f"\nSource images : {len(images)}")
    print(f"Source labels : {len(labels)}")

    if len(images) == 0:
        print("\nERROR: No images found.")
        return

    if len(labels) == 0:
        print("\nERROR: No labels found.")
        return

    # --------------------------------------------------------
    # 3. Check image-label matching
    # --------------------------------------------------------

    image_stems = {
        image.stem
        for image in images
    }

    label_stems = {
        label.stem
        for label in labels
    }

    missing_labels = image_stems - label_stems
    missing_images = label_stems - image_stems

    if missing_labels:

        print("\nERROR: Images without labels:")

        for name in sorted(missing_labels):
            print(f"  {name}")

        print("\nNo files were copied.")
        return

    if missing_images:

        print("\nERROR: Labels without images:")

        for name in sorted(missing_images):
            print(f"  {name}")

        print("\nNo files were copied.")
        return

    print(
        "\nEvery image has a matching label."
    )

    # --------------------------------------------------------
    # 4. Validate YOLO labels
    # --------------------------------------------------------

    total_objects = 0

    for label_file in labels:

        with open(
            label_file,
            "r",
            encoding="utf-8"
        ) as f:

            lines = [
                line.strip()
                for line in f
                if line.strip()
            ]

        if not lines:

            print(
                f"\nERROR: Empty label file:"
                f"\n{label_file}"
            )

            print("\nNo files were copied.")
            return

        for line_number, line in enumerate(
            lines,
            start=1
        ):

            values = line.split()

            # YOLO format:
            # class x_center y_center width height

            if len(values) != 5:

                print(
                    f"\nERROR: Invalid YOLO label:"
                )

                print(
                    f"File: {label_file}"
                )

                print(
                    f"Line: {line_number}"
                )

                print(
                    f"Content: {line}"
                )

                print("\nNo files were copied.")
                return

            try:
                class_id = int(values[0])

                x_center = float(values[1])
                y_center = float(values[2])
                width = float(values[3])
                height = float(values[4])

            except ValueError:

                print(
                    f"\nERROR: Non-numeric label:"
                )

                print(
                    f"File: {label_file}"
                )

                print(
                    f"Line: {line_number}"
                )

                print(
                    f"Content: {line}"
                )

                print("\nNo files were copied.")
                return

            # Spark plug must be class 2
            if class_id != 2:

                print(
                    f"\nERROR: Unexpected class ID"
                )

                print(
                    f"File: {label_file}"
                )

                print(
                    f"Found class: {class_id}"
                )

                print(
                    "Expected class: 2 (spark_plug)"
                )

                print("\nNo files were copied.")
                return

            # Check normalized coordinates
            if not (
                0 <= x_center <= 1
                and 0 <= y_center <= 1
                and 0 < width <= 1
                and 0 < height <= 1
            ):

                print(
                    f"\nERROR: Invalid bounding box"
                )

                print(
                    f"File: {label_file}"
                )

                print(
                    f"Line: {line_number}"
                )

                print(
                    f"Content: {line}"
                )

                print("\nNo files were copied.")
                return

            total_objects += 1

    print(
        f"Spark-plug objects found: "
        f"{total_objects}"
    )

    # --------------------------------------------------------
    # 5. Check duplicate filenames
    # --------------------------------------------------------

    duplicate_images = []
    duplicate_labels = []

    for image in images:

        destination = TARGET_IMAGES / image.name

        if destination.exists():
            duplicate_images.append(
                image.name
            )

    for label in labels:

        destination = TARGET_LABELS / label.name

        if destination.exists():
            duplicate_labels.append(
                label.name
            )

    if duplicate_images or duplicate_labels:

        print(
            "\nERROR: Some files already exist."
        )

        if duplicate_images:

            print("\nDuplicate images:")

            for name in duplicate_images:
                print(f"  {name}")

        if duplicate_labels:

            print("\nDuplicate labels:")

            for name in duplicate_labels:
                print(f"  {name}")

        print(
            "\nNo files were copied."
        )

        return

    # --------------------------------------------------------
    # 6. Copy files
    # --------------------------------------------------------

    print(
        "\nCopying correction images and labels..."
    )

    copied_images = 0
    copied_labels = 0

    for image in images:

        destination = TARGET_IMAGES / image.name

        shutil.copy2(
            image,
            destination
        )

        copied_images += 1

    for label in labels:

        destination = TARGET_LABELS / label.name

        shutil.copy2(
            label,
            destination
        )

        copied_labels += 1

    # --------------------------------------------------------
    # 7. Final verification
    # --------------------------------------------------------

    print("\n==============================================")
    print("MERGE COMPLETE")
    print("==============================================")

    print(
        f"Images added : {copied_images}"
    )

    print(
        f"Labels added : {copied_labels}"
    )

    print(
        f"Objects added: {total_objects}"
    )

    print("\nDestination:")

    print(
        f"Images -> {TARGET_IMAGES}"
    )

    print(
        f"Labels -> {TARGET_LABELS}"
    )

    print("\nDataset structure:")

    print(
        "production_yolo_v2/"
    )

    print(
        "├── train/       ← correction data added here"
    )

    print(
        "├── val/         ← unchanged"
    )

    print(
        "└── test/        ← unchanged"
    )

    print(
        "\nMerge completed successfully."
    )

    print(
        "Do NOT retrain until we verify the merged dataset."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()