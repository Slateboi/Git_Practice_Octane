# pip3 list | grep -E "torch|torchvision|torchaudio" --> For Check the Pytoch Lib Versions(all CPU)
from ultralytics import YOLO
import torch

def train_pose():
    model = YOLO("yolov8s-pose.pt")

    model.train(
        data="config.yaml",
        epochs=10,
        imgsz=128,
        batch=2,                 # SAFE for CPU
        workers=0,               # workers=0 tells YOLO to load data in the main process only (no parallel workers).
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

if __name__ == "__main__":
    print("CUDA:", torch.cuda.is_available())
    train_pose()


''' OUTPUT:
      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
      10/10         0G     0.1401     0.2187     0.2968     0.1383     0.8951          1        128: 100% ━━━━━━━━━━━━ 200/200 8.9it/s 22.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 40% ━━━━╸───
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 80% ━━━━━━━━
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 5/5 17.4it/s 0.3s
                   all         20         20          1          1      0.995      0.995          1          1      0.995      0.954
'''