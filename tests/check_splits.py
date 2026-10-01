from pathlib import Path
from collections import Counter


ROOT = Path("data/dataset/production_yolo_v2")

CLASS_NAMES = {
    0: "brake_pad",
    1: "bearing",
    2: "spark_plug",
    3: "gear",
}


def check_split(split):

    image_dir = ROOT / split / "images"
    label_dir = ROOT / split / "labels"

    images = list(image_dir.glob("*.jpg"))
    labels = list(label_dir.glob("*.txt"))

    counts = Counter()

    for label_file in labels:

        with open(label_file, "r", encoding="utf-8") as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                class_id = int(line.split()[0])

                counts[class_id] += 1

    print(f"\n{split.upper()}")
    print("-" * 40)

    print(f"Images : {len(images)}")
    print(f"Labels : {len(labels)}")

    for class_id in range(4):

        print(
            f"{CLASS_NAMES[class_id]:12s}: "
            f"{counts[class_id]}"
        )

    print(
        f"Total objects: {sum(counts.values())}"
    )


def main():

    print("\n==============================================")
    print("PRODUCTION DATASET SPLIT CHECK")
    print("==============================================")

    for split in ["train", "val", "test"]:
        check_split(split)

    print("\n==============================================")
    print("CHECK COMPLETE")
    print("==============================================")


if __name__ == "__main__":
    main()