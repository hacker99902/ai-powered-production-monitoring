from pathlib import Path
from collections import Counter


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPARK_PLUG_ROOT = (
    PROJECT_ROOT
    / "data"
    / "dataset"
    / "raw"
    / "spark_plugs"
)

# ============================================================
# DATASETS
# ============================================================
DATASETS = {
    "SPARK PLUG": (
        PROJECT_ROOT
        / "data"
        / "dataset"
        / "raw"
        / "spark_plugs"
        / "Spark plug images.v1i.yolov8"
    ),

    "BRAKE PAD": (
        PROJECT_ROOT
        / "data"
        / "dataset"
        / "raw"
        / "brake_pads"
        / "Brake Pads x.v2i.yolov8"
    ),
}

# ============================================================
# INSPECT ONE DATASET
# ============================================================

def inspect_dataset(name, root):

    print("\n")
    print("=" * 70)
    print(name)
    print("=" * 70)

    class_counts = Counter()

    invalid_lines = []
    missing_label_files = []

    total_lines = 0
    valid_objects = 0

    # --------------------------------------------------------
    # Process splits
    # --------------------------------------------------------

    for split in ["train", "valid", "test", "val"]:

        label_dir = root / split / "labels"

        if not label_dir.exists():
            continue

        label_files = list(
            label_dir.glob("*.txt")
        )

        print(
            f"\nScanning {split}: "
            f"{len(label_files)} files"
        )

        for index, label_file in enumerate(
            label_files,
            start=1
        ):

            if index % 100 == 0:

                print(
                    f"  {index}/{len(label_files)}",
                    end="\r"
                )

            # ------------------------------------------------
            # File may have disappeared / be inaccessible
            # ------------------------------------------------

            if not label_file.exists():

                missing_label_files.append(
                    str(label_file)
                )

                continue

            try:

                with open(
                    label_file,
                    "r",
                    encoding="utf-8"
                ) as f:

                    lines = f.readlines()

            except FileNotFoundError:

                missing_label_files.append(
                    str(label_file)
                )

                continue

            except OSError as e:

                print(
                    f"\nCould not read:"
                    f"\n{label_file}"
                    f"\nReason: {e}"
                )

                continue

            # ------------------------------------------------
            # Process annotations
            # ------------------------------------------------

            for line_number, line in enumerate(
                lines,
                start=1
            ):

                line = line.strip()

                if not line:
                    continue

                total_lines += 1

                parts = line.split()

                # --------------------------------------------
                # YOLO format must have 5 values
                # --------------------------------------------

                if len(parts) != 5:

                    invalid_lines.append(
                        (
                            str(label_file),
                            line_number,
                            line
                        )
                    )

                    continue

                try:

                    class_id = int(parts[0])

                    x = float(parts[1])
                    y = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])

                except ValueError:

                    invalid_lines.append(
                        (
                            str(label_file),
                            line_number,
                            line
                        )
                    )

                    continue

                # --------------------------------------------
                # Check YOLO coordinates
                # --------------------------------------------

                if not (
                    0 <= x <= 1
                    and 0 <= y <= 1
                    and 0 < w <= 1
                    and 0 < h <= 1
                ):

                    invalid_lines.append(
                        (
                            str(label_file),
                            line_number,
                            line
                        )
                    )

                    continue

                # --------------------------------------------
                # Valid object
                # --------------------------------------------

                class_counts[class_id] += 1

                valid_objects += 1

    # ========================================================
    # RESULTS
    # ========================================================

    print("\n")

    print("-" * 70)
    print("CLASS DISTRIBUTION")
    print("-" * 70)

    if class_counts:

        for class_id in sorted(class_counts):

            print(
                f"Class {class_id:>2}: "
                f"{class_counts[class_id]} objects"
            )

    else:

        print("No valid objects found.")

    print("\nTOTAL VALID OBJECTS:")
    print(valid_objects)

    print("\nTOTAL LABEL LINES:")
    print(total_lines)

    print("\nINVALID ANNOTATION LINES:")
    print(len(invalid_lines))

    print("\nMISSING / UNREADABLE LABEL FILES:")
    print(len(missing_label_files))

    # ========================================================
    # SHOW INVALID LINES
    # ========================================================

    if invalid_lines:

        print("\n")
        print("-" * 70)
        print("FIRST 20 INVALID ANNOTATIONS")
        print("-" * 70)

        for filename, line_number, line in (
            invalid_lines[:20]
        ):

            print(
                f"\n{filename}:{line_number}"
            )

            print(
                f"  {line}"
            )

    # ========================================================
    # SHOW MISSING FILES
    # ========================================================

    if missing_label_files:

        print("\n")
        print("-" * 70)
        print("MISSING / UNREADABLE FILES")
        print("-" * 70)

        for filename in missing_label_files[:20]:

            print(filename)

        if len(missing_label_files) > 20:

            print(
                f"\n... and "
                f"{len(missing_label_files) - 20}"
                f" more."
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("DATASET CLASS DIAGNOSTIC")
    print("=" * 70)

    print(
        "\nThis script ONLY READS the datasets."
    )

    print(
        "It does NOT modify, delete, or merge anything."
    )

    for name, root in DATASETS.items():

        inspect_dataset(
            name,
            root
        )

    print("\n")
    print("=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()