from pathlib import Path
from collections import Counter


DATASET = Path(
    "data/dataset/production_yolo_v2"
)

TRAIN_IMAGES = DATASET / "train" / "images"
TRAIN_LABELS = DATASET / "train" / "labels"


def main():

    print("\n==============================================")
    print("VERIFYING MERGED DATASET")
    print("==============================================")

    images = list(TRAIN_IMAGES.glob("*.jpg"))
    labels = list(TRAIN_LABELS.glob("*.txt"))

    print(f"\nTraining images : {len(images)}")
    print(f"Training labels : {len(labels)}")

    # --------------------------------------------------------
    # Check image-label matching
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

    print(
        f"\nImages without labels : "
        f"{len(missing_labels)}"
    )

    print(
        f"Labels without images : "
        f"{len(missing_images)}"
    )

    # --------------------------------------------------------
    # Count classes
    # --------------------------------------------------------

    class_counts = Counter()
    object_count = 0

    bad_files = []

    for label_file in labels:

        with open(
            label_file,
            "r",
            encoding="utf-8"
        ) as f:

            for line_number, line in enumerate(
                f,
                start=1
            ):

                line = line.strip()

                if not line:
                    continue

                values = line.split()

                if len(values) != 5:

                    bad_files.append(
                        f"{label_file.name}: "
                        f"invalid number of values"
                    )

                    continue

                try:

                    class_id = int(values[0])

                    x = float(values[1])
                    y = float(values[2])
                    w = float(values[3])
                    h = float(values[4])

                except ValueError:

                    bad_files.append(
                        f"{label_file.name}: "
                        f"non-numeric values"
                    )

                    continue

                if not (
                    0 <= x <= 1
                    and 0 <= y <= 1
                    and 0 < w <= 1
                    and 0 < h <= 1
                ):

                    bad_files.append(
                        f"{label_file.name}: "
                        f"invalid bounding box"
                    )

                    continue

                class_counts[class_id] += 1
                object_count += 1

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\n==============================================")
    print("OBJECT COUNTS")
    print("==============================================")

    names = {
        0: "brake_pad",
        1: "bearing",
        2: "spark_plug",
        3: "gear"
    }

    for class_id in range(4):

        print(
            f"{names[class_id]:12s}: "
            f"{class_counts[class_id]}"
        )

    print(
        f"\nTotal objects: {object_count}"
    )

    # --------------------------------------------------------
    # Errors
    # --------------------------------------------------------

    print("\n==============================================")
    print("VALIDATION")
    print("==============================================")

    if missing_labels:
        print("❌ Missing labels")

    else:
        print("✓ Every image has a label")

    if missing_images:
        print("❌ Labels without images")

    else:
        print("✓ Every label has an image")

    if bad_files:

        print(
            f"❌ Invalid label files: "
            f"{len(bad_files)}"
        )

        for error in bad_files[:20]:
            print(f"  {error}")

    else:
        print("✓ All YOLO annotations are valid")

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    if (
        not missing_labels
        and not missing_images
        and not bad_files
    ):

        print(
            "\n=============================================="
        )

        print(
            "DATASET VERIFICATION PASSED ✓"
        )

        print(
            "=============================================="
        )

        print(
            "\nThe dataset is ready for the next step."
        )

    else:

        print(
            "\nDATASET VERIFICATION FAILED."
        )

        print(
            "Do NOT train until the problems are fixed."
        )


if __name__ == "__main__":
    main()