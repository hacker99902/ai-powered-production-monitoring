import os
import shutil
import yaml
from pathlib import Path
from collections import Counter


# ============================================================
# FINAL CLASS MAPPING
# ============================================================

FINAL_CLASSES = {
    "brake pad": 0,
    "bearing": 1,
    "spark plug": 2,
    "gear": 3,
}


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "dataset" / "raw"

ROBOFLOW_DIR = (
    RAW_DIR
    / "roboflow"
    / "Machine Parts.v2-raw.yolov8"
)

ZENODO_DIR = (
    RAW_DIR
    / "zenodo"
    / "Mechanical Parts Dataset"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "dataset"
    / "production_yolo"
)


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

for split in ["train", "val", "test"]:

    (OUTPUT_DIR / "images" / split).mkdir(
        parents=True,
        exist_ok=True
    )

    (OUTPUT_DIR / "labels" / split).mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# NORMALIZE CLASS NAME
# ============================================================

def normalize_name(name):

    return (
        name.lower()
        .strip()
        .replace("_", " ")
        .replace("-", " ")
    )


# ============================================================
# READ YAML
# ============================================================

def read_yaml(yaml_path):

    print("\nReading:")
    print(yaml_path)

    with open(
        yaml_path,
        "r",
        encoding="utf-8"
    ) as f:

        data = yaml.safe_load(f)

    names = data["names"]

    if isinstance(names, dict):

        names = [
            names[i]
            for i in sorted(names.keys())
        ]

    print("\nSource classes:")

    for i, name in enumerate(names):

        print(f"{i:2d} -> {name}")

    return names


# ============================================================
# BUILD CLASS MAPPING
# ============================================================

def build_mapping(names):

    mapping = {}

    print("\nClass mapping:")

    for source_id, source_name in enumerate(names):

        normalized = normalize_name(
            source_name
        )

        if normalized in FINAL_CLASSES:

            final_id = FINAL_CLASSES[
                normalized
            ]

            mapping[source_id] = final_id

            print(
                f"KEEP   {source_id:2d} "
                f"{source_name:15s} "
                f"-> {final_id}"
            )

        else:

            print(
                f"IGNORE {source_id:2d} "
                f"{source_name}"
            )

    return mapping


# ============================================================
# FIND IMAGE
# ============================================================

def find_image(image_dir, stem):

    extensions = [
        ".jpg",
        ".jpeg",
        ".png",
        ".JPG",
        ".JPEG",
        ".PNG",
        ".webp",
        ".WEBP"
    ]

    for ext in extensions:

        image_path = image_dir / (
            stem + ext
        )

        if image_path.exists():

            return image_path

    return None


# ============================================================
# PROCESS ONE DATASET
# ============================================================

def process_dataset(
    dataset_dir,
    dataset_name
):

    print("\n")
    print("=" * 70)
    print(f"PROCESSING {dataset_name.upper()}")
    print("=" * 70)

    # --------------------------------------------------------
    # Find YAML
    # --------------------------------------------------------

    yaml_files = list(
        dataset_dir.glob("*.yaml")
    )

    if not yaml_files:

        yaml_files = list(
            dataset_dir.glob("*.yml")
        )

    if not yaml_files:

        raise FileNotFoundError(
            f"No YAML found in {dataset_dir}"
        )

    yaml_path = yaml_files[0]

    # --------------------------------------------------------
    # Read classes
    # --------------------------------------------------------

    names = read_yaml(yaml_path)

    source_to_final = build_mapping(names)

    statistics = Counter()

    # --------------------------------------------------------
    # Source split names
    # --------------------------------------------------------

    possible_splits = {
        "train": "train",
        "val": "val",
        "valid": "val",
        "test": "test"
    }

    for source_split, final_split in possible_splits.items():

        split_dir = (
            dataset_dir
            / source_split
        )

        if not split_dir.exists():

            continue

        image_dir = (
            split_dir
            / "images"
        )

        label_dir = (
            split_dir
            / "labels"
        )

        if not image_dir.exists():

            print(
                f"\nSkipping {source_split}: "
                f"images folder missing"
            )

            continue

        if not label_dir.exists():

            print(
                f"\nSkipping {source_split}: "
                f"labels folder missing"
            )

            continue

        print(
            f"\nProcessing "
            f"{dataset_name} -> "
            f"{source_split} "
            f"-> {final_split}"
        )

        label_files = list(
            label_dir.glob("*.txt")
        )

        print(
            f"Found {len(label_files)} "
            f"label files"
        )

        # ----------------------------------------------------
        # Process every label
        # ----------------------------------------------------

        for label_file in label_files:

            output_lines = []

            with open(
                label_file,
                "r",
                encoding="utf-8"
            ) as f:

                lines = f.readlines()

            for line in lines:

                parts = line.strip().split()

                if len(parts) != 5:

                    continue

                try:

                    source_id = int(
                        parts[0]
                    )

                except ValueError:

                    continue

                # Ignore unwanted classes

                if source_id not in source_to_final:

                    continue

                final_id = source_to_final[
                    source_id
                ]

                # --------------------------------------------
                # Validate YOLO coordinates
                # --------------------------------------------

                try:

                    x = float(parts[1])
                    y = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])

                except ValueError:

                    continue

                if not (
                    0 <= x <= 1
                    and
                    0 <= y <= 1
                    and
                    0 < w <= 1
                    and
                    0 < h <= 1
                ):

                    print(
                        "WARNING: invalid bbox:",
                        label_file.name
                    )

                    continue

                # --------------------------------------------
                # Create final YOLO annotation
                # --------------------------------------------

                output_lines.append(
                    f"{final_id} "
                    f"{x} "
                    f"{y} "
                    f"{w} "
                    f"{h}\n"
                )

                statistics[final_id] += 1

            # ------------------------------------------------
            # No required objects
            # ------------------------------------------------

            if not output_lines:

                continue

            # ------------------------------------------------
            # Find corresponding image
            # ------------------------------------------------

            image_path = find_image(
                image_dir,
                label_file.stem
            )

            if image_path is None:

                print(
                    "WARNING: image missing:",
                    label_file.name
                )

                continue

            # ------------------------------------------------
            # Avoid filename collision
            # ------------------------------------------------

            new_stem = (
                f"{dataset_name}_"
                f"{label_file.stem}"
            )

            output_image = (
                OUTPUT_DIR
                / "images"
                / final_split
                / (
                    new_stem
                    + image_path.suffix
                )
            )

            output_label = (
                OUTPUT_DIR
                / "labels"
                / final_split
                / (
                    new_stem
                    + ".txt"
                )
            )

            # ------------------------------------------------
            # Copy image
            # ------------------------------------------------

            shutil.copy2(
                image_path,
                output_image
            )

            # ------------------------------------------------
            # Write converted label
            # ------------------------------------------------

            with open(
                output_label,
                "w",
                encoding="utf-8"
            ) as f:

                f.writelines(
                    output_lines
                )

    return statistics


# ============================================================
# CREATE DATA.YAML
# ============================================================

def create_data_yaml():

    yaml_path = (
        OUTPUT_DIR
        / "data.yaml"
    )

    data = {
        "path": str(
            OUTPUT_DIR.resolve()
        ),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 4,
        "names": [
            "brake_pad",
            "bearing",
            "spark_plug",
            "gear"
        ]
    }

    with open(
        yaml_path,
        "w",
        encoding="utf-8"
    ) as f:

        yaml.dump(
            data,
            f,
            sort_keys=False
        )

    print("\nCreated:")
    print(yaml_path)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SMART FACTORY - DATASET PREPARATION")
    print("=" * 70)

    print("\nFinal classes:")

    print("0 -> brake_pad")
    print("1 -> bearing")
    print("2 -> spark_plug")
    print("3 -> gear")

    # --------------------------------------------------------
    # Process Roboflow
    # --------------------------------------------------------

    roboflow_stats = process_dataset(
        ROBOFLOW_DIR,
        "roboflow"
    )

    # --------------------------------------------------------
    # Process Zenodo
    # --------------------------------------------------------

    zenodo_stats = process_dataset(
        ZENODO_DIR,
        "zenodo"
    )

    # --------------------------------------------------------
    # Combine statistics
    # --------------------------------------------------------

    total_stats = (
        roboflow_stats
        + zenodo_stats
    )

    # --------------------------------------------------------
    # Create final YAML
    # --------------------------------------------------------

    create_data_yaml()

    # --------------------------------------------------------
    # Print final statistics
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("FINAL OBJECT COUNTS")
    print("=" * 70)

    class_names = {
        0: "brake_pad",
        1: "bearing",
        2: "spark_plug",
        3: "gear"
    }

    for class_id in range(4):

        print(
            f"{class_names[class_id]:15s}"
            f" : "
            f"{total_stats[class_id]}"
        )

    print("\nFinal dataset:")
    print(OUTPUT_DIR)

    print("\nDone! ✅")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()