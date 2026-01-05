from ultralytics import YOLO

model = YOLO("runs/pose/train/weights/best.pt")
model.predict(
    source="RekTNet_Full_Test_Imgs",  # directory of images
    imgsz=128,
    save=True,
    conf=0.25
)
