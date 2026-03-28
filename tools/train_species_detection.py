"""Simple Ultralytics training script for cedr/pinus seedling detection."""

from ultralytics import YOLO


model = YOLO("models/bestCrop.pt")

results = model.train(
    data="dataset/datasetV1_species/data.yaml",
    imgsz=896,
    epochs=80,
    batch=-1,
    device="0",
    workers=2,
    patience=25,
    project="results",
    name="cedr_pinus_detect",
)
