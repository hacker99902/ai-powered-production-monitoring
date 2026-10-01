from pathlib import Path
from collections import Counter


# ============================================================
# DATASET PATH
# ============================================================

ROOT = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "dataset"
    / "production_yolo_v2"
)


# ============================================================
# EXPECTED CLASSES
# ============================================================

CLASS_NAMES = {
    0: "brake_pad",
    1: "bearing",
    2: "spark_plug",
    3: "gear",
}


# ============================================================
# AUDIT
# ============================================================

def main():

    print("=" * 70)
    print("PRODUCTION YOLO V2 DATASET AUDIT")
    print("=" * 70)

    print("\nDataset:")
    print(ROOT)

    if not ROOT.exists():
        print("\nERROR: Dataset does not exist.")
        return

    total_objects = Counter()
    total_images = Counter()

    empty_labels = []
    bad_lines = []
    invalid_classes = []
    invalid_values = []

    # ========================================================
    # CHECK EACH SPLIT
    # ========================================================

    for split in ["train", "val", "test"]:

        # IMPORTANT:
        # Your dataset uses:
        #
        # production_yolo_v2/
        # ├── train/
        # │   ├── images/
        # │   └── labels/
        #
        # Therefore the paths are:
        label_dir = ROOT / split / "labels"
        image_dir = ROOT / split / "images"

        print("\n" + "-" * 70)
        print(split.upper())
        print("-" * 70)

        if not label_dir.exists():
            print("ERROR: Label directory missing:")
            print(label_dir)
            continue

        if not image_dir.exists():
            print("ERROR: Image directory missing:")
            print(image_dir)
            continue

        label_files = list(label_dir.glob("*.txt"))

        print("Label files:", len(label_files))

        split_objects = Counter()
        split_images = Counter()

        # ====================================================
        # READ LABEL FILES
        # ====================================================

        for label_file in label_files:

            text = label_file.read_text(
                encoding="utf-8"
            ).strip()

            # ------------------------------------------------
            # Empty label
            # ------------------------------------------------

            if not text:

                empty_labels.append(
                    str(label_file)
                )

                continue

            image_classes = set()

            # ------------------------------------------------
            # Read each annotation
            # ------------------------------------------------

            for line_number, line in enumerate(
                text.splitlines(),
                start=1
            ):

                parts = line.strip().split()

                # YOLO detection format:
                #
                # class
                # x_center
                # y_center
                # width
                # height

                if len(parts) != 5:

                    bad_lines.append(
                        (
                            str(label_file),
                            line_number,
                            line
                        )
                    )

                    continue

                # ------------------------------------------------
                # Convert values
                # ------------------------------------------------

                try:

                    class_id = int(parts[0])

                    x = float(parts[1])
                    y = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])

                except ValueError:

                    invalid_values.append(
                        (
                            str(label_file),
                            line_number,
                            line
                        )
                    )

                    continue

                # ------------------------------------------------
                # Check class ID
                # ------------------------------------------------

                if class_id not in CLASS_NAMES:

                    invalid_classes.append(
                        (
                            str(label_file),
                            line_number,
                            class_id
                        )
                    )

                    continue

                # ------------------------------------------------
                # Check bounding box
                # ------------------------------------------------

                values = [x, y, w, h]

                if any(
                    value < 0 or value > 1
                    for value in values
                ):

                    invalid_values.append(
                        (
                            str(label_file),
                            line_number,
                            line
                        )
                    )

                    continue

                if w <= 0 or h <= 0:

                    invalid_values.append(
                        (
                            str(label_file),
                            line_number,
                            line
                        )
                    )

                    continue

                # ------------------------------------------------
                # Valid object
                # ------------------------------------------------

                split_objects[class_id] += 1
                total_objects[class_id] += 1

                image_classes.add(class_id)

            # ------------------------------------------------
            # Count images containing each class
            # ------------------------------------------------

            for class_id in image_classes:

                split_images[class_id] += 1
                total_images[class_id] += 1

        # ====================================================
        # SPLIT RESULTS
        # ====================================================

        print("\nObjects:")

        for class_id, class_name in CLASS_NAMES.items():

            print(
                f"  {class_name:12s}: "
                f"{split_objects[class_id]}"
            )

        print("\nImages containing class:")

        for class_id, class_name in CLASS_NAMES.items():

            print(
                f"  {class_name:12s}: "
                f"{split_images[class_id]}"
            )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n")
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print("\nTotal objects:")

    for class_id, class_name in CLASS_NAMES.items():

        print(
            f"  {class_id} - {class_name:12s}: "
            f"{total_objects[class_id]}"
        )

    print("\nTotal images containing each class:")

    for class_id, class_name in CLASS_NAMES.items():

        print(
            f"  {class_id} - {class_name:12s}: "
            f"{total_images[class_id]}"
        )

    # ========================================================
    # DATA QUALITY
    # ========================================================

    print("\n")
    print("=" * 70)
    print("DATA QUALITY CHECK")
    print("=" * 70)

    print(
        f"\nEmpty label files   : {len(empty_labels)}"
    )

    print(
        f"Bad label lines     : {len(bad_lines)}"
    )

    print(
        f"Invalid class IDs   : {len(invalid_classes)}"
    )

    print(
        f"Invalid box values  : {len(invalid_values)}"
    )

    # ========================================================
    # RESULT
    # ========================================================

    if (
        len(empty_labels) == 0
        and len(bad_lines) == 0
        and len(invalid_classes) == 0
        and len(invalid_values) == 0
    ):

        print(
            "\nRESULT: DATASET PASSED BASIC VALIDATION"
        )

    else:

        print(
            "\nRESULT: DATASET HAS PROBLEMS"
        )

        print(
            "DO NOT TRAIN YET."
        )

    print("\n" + "=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()