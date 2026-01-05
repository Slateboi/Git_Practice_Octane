from ultralytics import YOLO

model = YOLO("runs/pose/train/weights/best.pt")

model.val(
    data="config.yaml",
    imgsz=128,
    device="cpu"
)


''' OUTPUT:
Ultralytics 8.3.246 🚀 Python-3.10.12 torch-2.9.1+cpu CPU (13th Gen Intel Core i5-1335U)
YOLOv8s-pose summary (fused): 81 layers, 11,414,235 parameters, 0 gradients, 29.4 GFLOPs
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 186.8±54.0 MB/s, size: 1.2 KB)
val: Scanning /home/soni/Desktop/Octane/RekTNet/yolo_pose_dataset/val/labels.cache... 20 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 20/20
val: Scanning /home/soni/Desktop/Octane/RekTNet/yolo_pose_dataset/val/labels.cache... 20 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 20/20 603.5Kit/s 0.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 50% ━━━━━━──
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 2/2 5.7it/s 0.3s
                   all         20         20          1          1      0.995      0.995          1          1      0.995      0.965
Speed: 0.0ms preprocess, 16.0ms inference, 0.0ms loss, 0.1ms postprocess per image
Results saved to /home/soni/Desktop/Octane/RekTNet/runs/pose/val
'''