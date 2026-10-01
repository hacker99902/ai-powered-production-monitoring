from roboflow import Roboflow
import os

API_KEY = "roboflow api key "

WORKSPACE = "penguin-wxcxp"
PROJECT = "machine-parts-bwqjl"

# Change this after checking the version you want
VERSION = 2

OUTPUT_DIR = "../data/dataset/raw/roboflow"

os.makedirs(OUTPUT_DIR, exist_ok=True)

rf = Roboflow(api_key=API_KEY)

project = rf.workspace(WORKSPACE).project(PROJECT)

dataset = project.version(VERSION).download(
    "yolov8",
    location=OUTPUT_DIR
)

print("\nDataset downloaded successfully!")
print("Location:", OUTPUT_DIR)