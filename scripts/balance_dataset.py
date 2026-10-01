from pathlib import Path
import shutil
import hashlib
import random
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# IMPORTANT:
# This is the dataset we already selected earlier.
SOURCE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "dataset"
    / "production_yolo_final"
)

# New cleaned dataset
OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "dataset"
    / "production_yolo_clean"
)

RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# Final class mapping
CLASS_NAMES = [
    "brake_pad",
    "bearing",
    "spark_plug",
    "gear",
]


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
# FULL-IMAGE BOX DETECTION
# ============================================================

def is_full_image_box(parts):
    """
    Detect a bounding box covering the complete image.

    YOLO format:
    class x_center y_center width height

    Example:
    2 0.5 0.5 1.0 1.0
    """

    try:
        width = float(parts[3])
        height = float(parts[4])
    except (ValueError, IndexError):
        return False

    return (
        abs(width - 1.0) < 1e-6
        and abs(height - 1.0) < 1e-6
    )


# ============================================================
# VALID YOLO BOX
# ============================================================

def valid_yolo_box(parts):

    if len(parts) != 5:
        return False

    try:
        class_id = int(parts[0])

        x = float(parts[1])
        y = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])

    except ValueError:
        return False

    if not 0 <= class_id < len(CLASS_NAMES):
        return False

    if not 0 <= x <= 1:
        return False

    if not 0 <= y <= 1:
        return False

    if not 0 < width <= 1:
        return False

    if not 0 < height <= 1:
        return False

    return True


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
# CHECK IMAGE
# ============================================================

def valid_image(image_path):

    try:

        with Image.open(image_path) as img:
            img.verify()

        return True

    except Exception:

        return False


# ============================================================
# HASH IMAGE
# ============================================================

def image_hash(image_path):

    sha = hashlib.sha256()

    with open(image_path, "rb") as f:

        while True:

            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            sha.update(chunk)

    return sha.hexdigest()


# ============================================================
# COLLECT CLEAN SAMPLES
# ============================================================

def collect_samples():

    samples = []

    total_labels = 0
    total_images = 0

    removed_full_image = {
        "brake_pad": 0,
        "bearing": 0,
        "spark_plug": 0,
        "gear": 0,
    }

    removed_empty_images = 0
    invalid_images = 0

    seen_hashes = set()

    # --------------------------------------------------------
    # Read all existing images from production_yolo_final
    # --------------------------------------------------------

    for split in ["train", "val", "test"]:

        image_dir = (
            SOURCE_ROOT
            / "images"
            / split
        )

        label_dir = (
            SOURCE_ROOT
            / "labels"
            / split
        )

        if not label_dir.exists():
            continue

        label_files = list(
            label_dir.glob("*.txt")
        )

        print(
            f"\nScanning {split}: "
            f"{len(label_files)} labels"
        )

        for index, label_path in enumerate(
            label_files,
            start=1
        ):

            total_labels += 1

            if index % 100 == 0:

                print(
                    f"  {index}/{len(label_files)}",
                    end="\r"
                )

            image_path = find_image(
                label_path,
                image_dir
            )

            if image_path is None:
                continue

            if not valid_image(image_path):

                invalid_images += 1

                continue

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

            cleaned_labels = []

            for line in lines:

                parts = line.split()

                if not valid_yolo_box(parts):
                    continue

                class_id = int(parts[0])

                class_name = CLASS_NAMES[class_id]

                # ------------------------------------------------
                # IMPORTANT CLEANING RULE
                #
                # Only remove full-image boxes for:
                #
                # 0 = brake_pad
                # 2 = spark_plug
                #
                # Bearing and gear full-image boxes are retained.
                # ------------------------------------------------

                if (
                    class_id in [0, 2]
                    and is_full_image_box(parts)
                ):

                    removed_full_image[
                        class_name
                    ] += 1

                    continue

                cleaned_labels.append(
                    " ".join(
                        [
                            str(class_id),
                            f"{float(parts[1]):.6f}",
                            f"{float(parts[2]):.6f}",
                            f"{float(parts[3]):.6f}",
                            f"{float(parts[4]):.6f}",
                        ]
                    )
                )

            # ------------------------------------------------
            # If removing bad annotations left the image empty,
            # don't use the image.
            # ------------------------------------------------

            if not cleaned_labels:

                removed_empty_images += 1

                continue

            # ------------------------------------------------
            # Remove exact duplicate images
            # ------------------------------------------------

            try:

                file_hash = image_hash(
                    image_path
                )

            except Exception:

                continue

            if file_hash in seen_hashes:
                continue

            seen_hashes.add(file_hash)

            # ------------------------------------------------
            # Save sample information
            # ------------------------------------------------

            samples.append(
                {
                    "image": image_path,
                    "labels": cleaned_labels,
                }
            )

            total_images += 1

    print("\n")

    print("=" * 70)
    print("CLEANING RESULTS")
    print("=" * 70)

    print(
        f"\nOriginal label files scanned: "
        f"{total_labels}"
    )

    print(
        f"Usable unique images: "
        f"{total_images}"
    )

    print(
        f"Images removed because no labels remained: "
        f"{removed_empty_images}"
    )

    print(
        f"Invalid images: "
        f"{invalid_images}"
    )

    print("\nRemoved full-image boxes:")

    for class_name in CLASS_NAMES:

        print(
            f"  {class_name}: "
            f"{removed_full_image[class_name]}"
        )

    return samples


# ============================================================
# SPLIT DATASET
# ============================================================

def split_dataset(samples):

    random.seed(RANDOM_SEED)

    random.shuffle(samples)

    total = len(samples)

    train_end = int(
        total * TRAIN_RATIO
    )

    val_end = (
        train_end
        + int(total * VAL_RATIO)
    )

    train = samples[:train_end]

    val = samples[
        train_end:val_end
    ]

    test = samples[val_end:]

    return train, val, test


# ============================================================
# COUNT OBJECTS
# ============================================================

def count_objects(samples):

    counts = {
        class_name: 0
        for class_name in CLASS_NAMES
    }

    image_counts = {
        class_name: 0
        for class_name in CLASS_NAMES
    }

    for sample in samples:

        classes_in_image = set()

        for line in sample["labels"]:

            class_id = int(
                line.split()[0]
            )

            class_name = CLASS_NAMES[
                class_id
            ]

            counts[class_name] += 1

            classes_in_image.add(
                class_name
            )

        for class_name in classes_in_image:

            image_counts[class_name] += 1

    return counts, image_counts


# ============================================================
# WRITE SPLIT
# ============================================================

def write_split(
    samples,
    split_name
):

    image_dir = (
        OUTPUT_ROOT
        / "images"
        / split_name
    )

    label_dir = (
        OUTPUT_ROOT
        / "labels"
        / split_name
    )

    image_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    label_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for index, sample in enumerate(
        samples
    ):

        source_image = sample["image"]

        extension = (
            source_image
            .suffix
            .lower()
        )

        filename = (
            f"production_{index:06d}"
            f"{extension}"
        )

        destination_image = (
            image_dir
            / filename
        )

        destination_label = (
            label_dir
            / f"production_{index:06d}.txt"
        )

        shutil.copy2(
            source_image,
            destination_image
        )

        with open(
            destination_label,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(
                "\n".join(
                    sample["labels"]
                )
            )


# ============================================================
# WRITE DATA.YAML
# ============================================================

def write_yaml():

    yaml_content = f"""path: {OUTPUT_ROOT.resolve()}
train: images/train
val: images/val
test: images/test
nc: 4
names:
- brake_pad
- bearing
- spark_plug
- gear
"""

    yaml_path = (
        OUTPUT_ROOT
        / "data.yaml"
    )

    with open(
        yaml_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(yaml_content)

    return yaml_path


# ============================================================
# VALIDATE OUTPUT
# ============================================================

def validate_output():

    print("\n" + "=" * 70)
    print("FINAL DATASET VALIDATION")
    print("=" * 70)

    total_images = 0
    total_labels = 0

    for split in [
        "train",
        "val",
        "test",
    ]:

        image_dir = (
            OUTPUT_ROOT
            / "images"
            / split
        )

        label_dir = (
            OUTPUT_ROOT
            / "labels"
            / split
        )

        images = []

        if image_dir.exists():

            images = [
                p
                for p in image_dir.iterdir()
                if p.suffix.lower()
                in IMAGE_EXTENSIONS
            ]

        labels = list(
            label_dir.glob("*.txt")
        ) if label_dir.exists() else []

        total_images += len(images)
        total_labels += len(labels)

        counts = {
            class_name: 0
            for class_name in CLASS_NAMES
        }

        for label_file in labels:

            with open(
                label_file,
                "r",
                encoding="utf-8"
            ) as f:

                for line in f:

                    parts = line.split()

                    if len(parts) != 5:
                        continue

                    class_id = int(parts[0])

                    if 0 <= class_id < 4:

                        counts[
                            CLASS_NAMES[class_id]
                        ] += 1

        print(f"\n{split.upper()}")

        print(
            f"  Images: {len(images)}"
        )

        print(
            f"  Labels: {len(labels)}"
        )

        for class_name in CLASS_NAMES:

            print(
                f"  {class_name}: "
                f"{counts[class_name]}"
            )

    print("\n" + "-" * 50)

    print(
        f"TOTAL IMAGES: {total_images}"
    )

    print(
        f"TOTAL LABEL FILES: {total_labels}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CLEAN PRODUCTION DATASET BUILDER")
    print("=" * 70)

    print(
        "\nSOURCE:"
        f"\n{SOURCE_ROOT}"
    )

    print(
        "\nOUTPUT:"
        f"\n{OUTPUT_ROOT}"
    )

    print("\nClasses:")

    for index, name in enumerate(
        CLASS_NAMES
    ):

        print(
            f"  {index}: {name}"
        )

    print(
        "\nCleaning rule:"
    )

    print(
        "  Remove full-image brake_pad boxes"
    )

    print(
        "  Remove full-image spark_plug boxes"
    )

    print(
        "  Keep normal bounding boxes"
    )

    # --------------------------------------------------------
    # SAFETY CHECK
    # --------------------------------------------------------

    if not SOURCE_ROOT.exists():

        raise FileNotFoundError(
            f"\nSource dataset not found:\n"
            f"{SOURCE_ROOT}"
        )

    # --------------------------------------------------------
    # Delete ONLY the new clean dataset
    # --------------------------------------------------------

    if OUTPUT_ROOT.exists():

        print(
            "\nRemoving previous clean dataset..."
        )

        shutil.rmtree(
            OUTPUT_ROOT
        )

    # --------------------------------------------------------
    # Collect
    # --------------------------------------------------------

    samples = collect_samples()

    if not samples:

        raise RuntimeError(
            "No usable samples found."
        )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    train, val, test = split_dataset(
        samples
    )

    print("\n" + "=" * 70)
    print("DATASET SPLIT")
    print("=" * 70)

    print(
        f"\nTrain: {len(train)}"
    )

    print(
        f"Val:   {len(val)}"
    )

    print(
        f"Test:  {len(test)}"
    )

    # --------------------------------------------------------
    # Object counts
    # --------------------------------------------------------

    train_counts, train_images = (
        count_objects(train)
    )

    val_counts, val_images = (
        count_objects(val)
    )

    test_counts, test_images = (
        count_objects(test)
    )

    print("\n" + "=" * 70)
    print("OBJECT COUNTS")
    print("=" * 70)

    print(
        f"\n{'Class':<15}"
        f"{'Train':>10}"
        f"{'Val':>10}"
        f"{'Test':>10}"
        f"{'Total':>10}"
    )

    print("-" * 55)

    for class_name in CLASS_NAMES:

        total = (
            train_counts[class_name]
            + val_counts[class_name]
            + test_counts[class_name]
        )

        print(
            f"{class_name:<15}"
            f"{train_counts[class_name]:>10}"
            f"{val_counts[class_name]:>10}"
            f"{test_counts[class_name]:>10}"
            f"{total:>10}"
        )

    print("\nImages containing each class:")

    print(
        f"\n{'Class':<15}"
        f"{'Train':>10}"
        f"{'Val':>10}"
        f"{'Test':>10}"
    )

    print("-" * 45)

    for class_name in CLASS_NAMES:

        print(
            f"{class_name:<15}"
            f"{train_images[class_name]:>10}"
            f"{val_images[class_name]:>10}"
            f"{test_images[class_name]:>10}"
        )

    # --------------------------------------------------------
    # Write
    # --------------------------------------------------------

    print("\nWriting dataset...")

    write_split(
        train,
        "train"
    )

    write_split(
        val,
        "val"
    )

    write_split(
        test,
        "test"
    )

    yaml_path = write_yaml()

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_output()

    # --------------------------------------------------------
    # Finish
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DATASET CREATED SUCCESSFULLY")
    print("=" * 70)

    print(
        f"\nClean dataset:"
        f"\n{OUTPUT_ROOT}"
    )

    print(
        f"\nYAML:"
        f"\n{yaml_path}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "production_yolo_final was NOT modified."
    )


if __name__ == "__main__":
    main()