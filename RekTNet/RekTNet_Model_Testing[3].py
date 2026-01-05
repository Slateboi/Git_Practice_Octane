from ultralytics import YOLO

model = YOLO("runs/pose/train/weights/best.pt")
model.predict(
    source="yolo_pose_dataset/test",  # directory of images
    imgsz=128,
    save=True,
    conf=0.25
)

''' Test_Img_Size :20 ScreenShoots'''