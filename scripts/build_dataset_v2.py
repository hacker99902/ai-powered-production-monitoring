from pathlib import Path
import shutil
import random


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# SOURCE DATASETS
# ============================================================

OLD_DATASET = (
    PROJECT_ROOT
    / "data"
    / "dataset"
    / "production_yolo_clean"
)

BRAKE_DATASET = (
    PROJECT_ROOT
    / "data"
    / "dataset"
    / "processed"
    / "brake_pads_clean"
)

SPARK_DATASET = (
    PROJECT_ROOT
    / "data"
    / "dataset"
    / "raw"
    / "spark_plugs"
    / "Spark plug images.v1i.yolov8"
)

OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "dataset"
    / "production_yolo_v2"
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_SEED = 42

TARGETS = {
    "brake_pad": 302,
    "bearing": 728,
    "spark_plug": 728,
    "gear": 616,
}


# Final project class IDs

CLASS_IDS = {
    "brake_pad": 0,
    "bearing": 1,
    "spark_plug": 2,
    "gear": 3,
}


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# GET SAMPLES
# ============================================================

def get_samples(root, class_id):
    """
    Collect images containing the requested class.

    Supports both dataset structures.

    Structure A:

        root/
        ├── images/
        │   ├── train/
        │   ├── val/
        │   └── test/
        │
        └── labels/
            ├── train/
            ├── val/
            └── test/

    Structure B:

        root/
        ├── train/
        │   ├── images/
        │   └── labels/
        │
        ├── valid/
        │   ├── images/
        │   └── labels/
        │
        └── test/
            ├── images/
            └── labels/
    """

    samples = []

    # --------------------------------------------------------
    # Structure A
    # --------------------------------------------------------

    structure_a = [
        (
            "train",
            root / "images" / "train",
            root / "labels" / "train",
        ),
        (
            "val",
            root / "images" / "val",
            root / "labels" / "val",
        ),
        (
            "test",
            root / "images" / "test",
            root / "labels" / "test",
        ),
    ]

    # --------------------------------------------------------
    # Structure B
    # --------------------------------------------------------

    structure_b = [
        (
            "train",
            root / "train" / "images",
            root / "train" / "labels",
        ),
        (
            "valid",
            root / "valid" / "images",
            root / "valid" / "labels",
        ),
        (
            "test",
            root / "test" / "images",
            root / "test" / "labels",
        ),
    ]

    # --------------------------------------------------------
    # Detect structure
    # --------------------------------------------------------

    if any(
        image_dir.exists()
        for _, image_dir, _ in structure_a
    ):
        structures = structure_a
    else:
        structures = structure_b

    # --------------------------------------------------------
    # Scan
    # --------------------------------------------------------

    for split, image_dir, label_dir in structures:

        if not image_dir.exists():
            continue

        if not label_dir.exists():
            continue

        for image_path in image_dir.iterdir():

            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            label_path = (
                label_dir
                / f"{image_path.stem}.txt"
            )

            if not label_path.exists():
                continue

            has_class = False

            try:

                lines = label_path.read_text(
                    encoding="utf-8"
                ).splitlines()

            except Exception:
                continue

            for line in lines:

                parts = line.strip().split()

                if not parts:
                    continue

                try:
                    cls = int(parts[0])
                except ValueError:
                    continue

                if cls == class_id:

                    has_class = True
                    break

            if has_class:

                samples.append(
                    (
                        image_path,
                        label_path,
                    )
                )

    return samples


# ============================================================
# SAMPLE DATA
# ============================================================

def sample_data(samples, target_count):
    """
    Randomly select up to target_count samples.
    """

    samples = samples.copy()

    random.shuffle(samples)

    if len(samples) <= target_count:
        return samples

    return samples[:target_count]


# ============================================================
# REMAP LABELS
# ============================================================

def remap_label(
    label_path,
    source_class,
    target_class,
):
    """
    Keep only the requested source class and
    convert it to the final project class.
    """

    output_lines = []

    try:

        lines = label_path.read_text(
            encoding="utf-8"
        ).splitlines()

    except Exception:

        return output_lines

    for line in lines:

        parts = line.strip().split()

        # Standard YOLO detection format
        if len(parts) != 5:
            continue

        try:
            cls = int(parts[0])
        except ValueError:
            continue

        if cls != source_class:
            continue

        output_lines.append(
            " ".join(
                [
                    str(target_class),
                    *parts[1:],
                ]
            )
        )

    return output_lines


# ============================================================
# ADD DATASET
# ============================================================

def add_dataset(
    samples,
    source_class,
    target_class,
    split,
    prefix,
):
    """
    Copy images and remapped labels into the
    final dataset.
    """

    image_dir = (
        OUTPUT
        / split
        / "images"
    )

    label_dir = (
        OUTPUT
        / split
        / "labels"
    )

    image_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    label_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    added = 0

    for index, (
        image_path,
        label_path,
    ) in enumerate(samples):

        new_stem = (
            f"{prefix}_{index:06d}"
        )

        new_image = (
            image_dir
            / f"{new_stem}{image_path.suffix.lower()}"
        )

        new_label = (
            label_dir
            / f"{new_stem}.txt"
        )

        # ----------------------------------------------------
        # Remap labels first
        # ----------------------------------------------------

        lines = remap_label(
            label_path,
            source_class,
            target_class,
        )

        # If no valid target annotations remain,
        # don't copy the image.
        if not lines:
            continue

        # ----------------------------------------------------
        # Copy image
        # ----------------------------------------------------

        shutil.copy2(
            image_path,
            new_image,
        )

        # ----------------------------------------------------
        # Write label
        # ----------------------------------------------------

        new_label.write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )

        added += 1

    return added


# ============================================================
# SPLIT DATASET
# ============================================================

def split_samples(samples):

    samples = samples.copy()

    random.shuffle(samples)

    n = len(samples)

    train_end = int(n * 0.70)

    val_end = int(n * 0.85)

    return {
        "train": samples[:train_end],
        "val": samples[train_end:val_end],
        "test": samples[val_end:],
    }


# ============================================================
# WRITE DATA YAML
# ============================================================

def write_yaml():

    yaml_text = f"""path: {OUTPUT.as_posix()}

train: train/images
val: val/images
test: test/images

nc: 4

names:
  0: brake_pad
  1: bearing
  2: spark_plug
  3: gear
"""

    yaml_path = (
        OUTPUT
        / "data.yaml"
    )

    yaml_path.write_text(
        yaml_text,
        encoding="utf-8",
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("BUILDING PRODUCTION DATASET V2")
    print("=" * 70)

    print("\nSources:")

    print(
        "Bearing/Gear:",
        OLD_DATASET,
    )

    print(
        "Brake Pad:",
        BRAKE_DATASET,
    )

    print(
        "Spark Plug:",
        SPARK_DATASET,
    )

    print("\nOutput:")

    print(OUTPUT)

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if OUTPUT.exists():

        print(
            "\nERROR: production_yolo_v2 already exists."
        )

        print(
            "Delete it manually before rebuilding."
        )

        return

    # --------------------------------------------------------
    # Random seed
    # --------------------------------------------------------

    random.seed(RANDOM_SEED)

    # --------------------------------------------------------
    # Collect source samples
    # --------------------------------------------------------

    brake_samples = get_samples(
        BRAKE_DATASET,
        class_id=0,
    )

    bearing_samples = get_samples(
        OLD_DATASET,
        class_id=1,
    )

    spark_samples = get_samples(
        SPARK_DATASET,
        class_id=0,
    )

    gear_samples = get_samples(
        OLD_DATASET,
        class_id=3,
    )

    # --------------------------------------------------------
    # Available images
    # --------------------------------------------------------

    print("\nAvailable images:")

    print(
        "Brake Pad:",
        len(brake_samples),
    )

    print(
        "Bearing:",
        len(bearing_samples),
    )

    print(
        "Spark Plug:",
        len(spark_samples),
    )

    print(
        "Gear:",
        len(gear_samples),
    )

    # --------------------------------------------------------
    # Select samples
    # --------------------------------------------------------

    brake_samples = sample_data(
        brake_samples,
        TARGETS["brake_pad"],
    )

    bearing_samples = sample_data(
        bearing_samples,
        TARGETS["bearing"],
    )

    spark_samples = sample_data(
        spark_samples,
        TARGETS["spark_plug"],
    )

    gear_samples = sample_data(
        gear_samples,
        TARGETS["gear"],
    )

    print("\nSelected images:")

    print(
        "Brake Pad:",
        len(brake_samples),
    )

    print(
        "Bearing:",
        len(bearing_samples),
    )

    print(
        "Spark Plug:",
        len(spark_samples),
    )

    print(
        "Gear:",
        len(gear_samples),
    )

    # --------------------------------------------------------
    # Dataset configuration
    # --------------------------------------------------------

    datasets = {

        "brake_pad": (
            brake_samples,
            0,      # source class
            0,      # target class
            "brake",
        ),

        "bearing": (
            bearing_samples,
            1,
            1,
            "bearing",
        ),

        "spark_plug": (
            spark_samples,
            0,
            2,
            "spark",
        ),

        "gear": (
            gear_samples,
            3,
            3,
            "gear",
        ),
    }

    totals = {
        "train": 0,
        "val": 0,
        "test": 0,
    }

    # --------------------------------------------------------
    # Split and copy
    # --------------------------------------------------------

    for name, (
        samples,
        source_class,
        target_class,
        prefix,
    ) in datasets.items():

        splits = split_samples(samples)

        print(f"\n{name}:")

        for split, split_samples_list in splits.items():

            count = add_dataset(
                split_samples_list,
                source_class,
                target_class,
                split,
                prefix,
            )

            totals[split] += count

            print(
                f"  {split}: {count}"
            )

    # --------------------------------------------------------
    # Write YAML
    # --------------------------------------------------------

    write_yaml()

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DATASET V2 CREATED")
    print("=" * 70)

    print("\nTotal images:")

    print(
        "Train:",
        totals["train"],
    )

    print(
        "Val:",
        totals["val"],
    )

    print(
        "Test:",
        totals["test"],
    )

    print(
        "\nOutput:",
        OUTPUT,
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()