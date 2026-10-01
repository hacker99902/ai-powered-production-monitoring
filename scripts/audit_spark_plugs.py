from pathlib import Path
from collections import Counter


DATASET = Path(
    "data/dataset/raw/spark_plugs/"
    "Spark plug images.v1i.yolov8"
)


def analyze_split(split):

    label_dir = DATASET / split / "labels"

    if not label_dir.exists():
        print(f"\nNo labels directory: {label_dir}")
        return

    object_counts = Counter()
    images_with_multiple = []

    label_files = list(label_dir.glob("*.txt"))

    for label_file in label_files:

        boxes = []

        with open(label_file, "r") as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                values = line.split()

                # YOLO:
                # class x_center y_center width height

                if len(values) != 5:
                    continue

                class_id = int(values[0])

                if class_id != 0:
                    continue

                boxes.append(
                    [
                        float(values[1]),
                        float(values[2]),
                        float(values[3]),
                        float(values[4]),
                    ]
                )

        count = len(boxes)

        object_counts[count] += 1

        if count >= 2:

            images_with_multiple.append(
                (
                    label_file,
                    count,
                    boxes
                )
            )

    print("\n======================================")
    print(f"SPARK PLUG AUDIT: {split}")
    print("======================================")

    print(
        f"Images: {len(label_files)}"
    )

    print("\nNumber of spark plugs per image:")

    for count in sorted(object_counts):

        print(
            f"  {count} object(s): "
            f"{object_counts[count]} images"
        )

    print(
        f"\nImages containing 2+ spark plugs: "
        f"{len(images_with_multiple)}"
    )

    print("\nExamples:")

    for item in images_with_multiple[:20]:

        label_file, count, boxes = item

        print(
            f"\n{label_file.name}"
        )

        print(
            f"  Objects: {count}"
        )

        for i, box in enumerate(boxes):

            xc, yc, w, h = box

            print(
                f"  Box {i+1}: "
                f"x={xc:.3f}, "
                f"y={yc:.3f}, "
                f"w={w:.3f}, "
                f"h={h:.3f}"
            )


def main():

    for split in [
        "train",
        "valid",
        "test"
    ]:

        analyze_split(split)


if __name__ == "__main__":
    main()