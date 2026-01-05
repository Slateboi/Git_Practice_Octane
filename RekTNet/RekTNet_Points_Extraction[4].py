import cv2
import numpy as np
from ultralytics import YOLO

# ---------------- CONFIG ----------------
MODEL_PATH = "runs/pose/train/weights/best.pt"
IMAGE_PATH = "yolo_pose_dataset/test/Screenshot from 2026-01-01 14-04-34.png"  # path to single test image 
IMG_SIZE = 128
# ----------------------------------------

# Load model
model = YOLO(MODEL_PATH)

# Run inference
results = model.predict(
    source=IMAGE_PATH,
    imgsz=IMG_SIZE,
    conf=0.25,
    device="cpu",
    verbose=False
)

# Load original image
img = cv2.imread(IMAGE_PATH)
h, w = img.shape[:2]

# Process detections
for r in results:
    if r.keypoints is None:
        continue

    # r.keypoints.xy → shape: (num_objects, num_keypoints, 2)
    keypoints = r.keypoints.xy.cpu().numpy()
    print("Type of keypoints:", type(keypoints))
    for kp_set in keypoints:
        print(f'Keypoints for one object: {kp_set}')
        for (x, y) in kp_set:
            x = int(x)
            y = int(y)

            # Draw keypoint
            cv2.circle(img, (x, y), 4, (0, 0, 255), -1)

# Show result
cv2.imshow("Keypoints Overlay", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

