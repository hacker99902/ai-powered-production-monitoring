from pathlib import Path
import shutil


SOURCE = Path(
    "data/dataset/raw/conveyor_spark_plug_generated_60"
)

SOURCE_IMAGES = SOURCE / "images"
SOURCE_LABELS = SOURCE / "labels"

TARGET = Path(
    "data/dataset/production_yolo_v2"
)

TARGET_IMAGES = TARGET / "train" / "images"
TARGET_LABELS = TARGET / "train" / "labels"


def main():

    print("=" * 50)
    print("MERGING GENERATED SPARK-PLUG DATASET")
    print("=" * 50)

    source_images = list(SOURCE_IMAGES.glob("*.jpg"))
    source_labels = list(SOURCE_LABELS.glob("*.txt"))

    print(f"Source images : {len(source_images)}")
    print(f"Source labels : {len(source_labels)}")

    # Check image/label counts
    if len(source_images) != len(source_labels):
        raise ValueError(
            "Image and label counts do not match!"
        )

    # Check every image has a label
    for image in source_images:

        label = SOURCE_LABELS / f"{image.stem}.txt"

        if not label.exists():
            raise FileNotFoundError(
                f"Missing label for {image.name}"
            )

    # Validate labels
    object_count = 0

    for label_file in source_labels:

        with open(label_file, "r", encoding="utf-8") as f:

            for line_number, line in enumerate(f, start=1):

                line = line.strip()

                if not line:
                    continue

                parts = line.split()

                # YOLO format:
                # class x_center y_center width height
                if len(parts) != 5:
                    raise ValueError(
                        f"Invalid YOLO line in "
                        f"{label_file.name}:{line_number}"
                    )

                class_id = int(parts[0])

                # Spark plug must be class 2
                if class_id != 2:
                    raise ValueError(
                        f"Unexpected class {class_id} "
                        f"in {label_file.name}"
                    )

                values = list(map(float, parts[1:]))

                for value in values:

                    if not 0 <= value <= 1:
                        raise ValueError(
                            f"Invalid normalized value "
                            f"in {label_file.name}"
                        )

                object_count += 1

    print(f"Spark-plug objects found: {object_count}")

    # Check duplicate filenames
    for image in source_images:

        target = TARGET_IMAGES / image.name

        if target.exists():
            raise FileExistsError(
                f"Duplicate image already exists: {image.name}"
            )

    print("\nCopying files...")

    for image in source_images:

        label = SOURCE_LABELS / f"{image.stem}.txt"

        shutil.copy2(
            image,
            TARGET_IMAGES / image.name
        )

        shutil.copy2(
            label,
            TARGET_LABELS / label.name
        )

    print("\nMERGE COMPLETE")
    print("-" * 50)
    print(f"Images added : {len(source_images)}")
    print(f"Labels added : {len(source_labels)}")
    print(f"Objects added: {object_count}")

    print("\nDestination:")
    print(f"Images -> {TARGET_IMAGES}")
    print(f"Labels -> {TARGET_LABELS}")


if __name__ == "__main__":
    main()