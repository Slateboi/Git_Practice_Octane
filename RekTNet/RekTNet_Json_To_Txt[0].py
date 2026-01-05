import json
import cv2
import os
from glob import glob

# ===================== CONFIG =====================
ANN_DIR = "archive/supervisely/Dataset/ann"   # folder with .json files
IMG_DIR = "archive/supervisely/Dataset/img"     # folder with images

OUT_DIR = "yolo_pose_dataset"
OUT_IMG_DIR = os.path.join(OUT_DIR, "images")
OUT_LBL_DIR = os.path.join(OUT_DIR, "labels")

CLASS_ID = 0  # cone
# =================================================

os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_LBL_DIR, exist_ok=True)

json_files = sorted(glob(os.path.join(ANN_DIR, "*.json")))
sample_id = 1

for jf in json_files:
    with open(jf, "r") as f:
        data = json.load(f)

    # Skip empty annotations
    if "objects" not in data or len(data["objects"]) == 0:
        continue

    obj = data["objects"][0]
    # graph --> Cone pose
    if obj.get("geometryType") != "graph": #  .get() tries to get 'geometryType' if it doesn't exist, returns None
        continue

    nodes = obj.get("nodes", {})  # tries to get 'nodes' dict if it doesn't exist, returns empty dict {}
    if len(nodes) == 0:
        continue

    # Extract keypoints
    keypoints = [node["loc"] for node in nodes.values()]
    print(f"Keypoints: {keypoints}") #debugging

    # Corresponding image (keep exact name)

    #jf --> Dataset/ann/amz_00026_class_0_1401_659_1410_670.jpg.json
    # os.path.basename(jf) --> only stores amz_00026_class_0_1401_659_1410_670.jpg.json
    # os.path.splitext(...) --> only stores amz_00026_class_0_1401_659_1410_670.jpg

    base_name = os.path.splitext(os.path.basename(jf))[0]
    img_path = os.path.join(IMG_DIR, base_name)

    image = cv2.imread(img_path)
    if image is None:
        continue

    img_h, img_w = image.shape[:2]

    # ================= BOUNDING BOX =================
    # FULL IMAGE AS BOUNDING BOX
    x_center = 0.5
    y_center = 0.5
    w_norm = 1.0
    h_norm = 1.0

    label = [CLASS_ID, x_center, y_center, w_norm, h_norm]

    # ================= KEYPOINTS ====================
    # Normalize keypoints relative to FULL IMAGE
    for x_kp, y_kp in keypoints:
        x_norm = x_kp / img_w
        y_norm = y_kp / img_h

        x_norm = max(0.0, min(1.0, x_norm))
        y_norm = max(0.0, min(1.0, y_norm))

        label.extend([x_norm, y_norm, 2])  # v=2 (visible)

    # Save image (sequential naming)
    out_name = f"{sample_id:05d}"
    cv2.imwrite(os.path.join(OUT_IMG_DIR, out_name + ".jpg"), image)

    # Save label
    with open(os.path.join(OUT_LBL_DIR, out_name + ".txt"), "w") as f:
        f.write(" ".join(f"{v:.6f}" for v in label))

    sample_id += 1

print("\n✅ Conversion complete")
print(f"Annotated samples written: {sample_id - 1}")
print(f"Saved to: {OUT_DIR}")



    


''' skeleton of .json file
{
  "description": "",
  "tags": [],
  "size": {
    "height": 38,
    "width": 32
  },
  "objects": [
    0: {
      "id": 1573923869,
      "classId": 13210152,
      "objectId": null,
      "classTitle": "Cone pose",
      "description": "",
      "geometryType": "graph",
      "labelerLogin": "jivap",
      "createdAt": "2024-07-01T21:42:22.212Z",
      "updatedAt": "2024-07-01T21:42:42.077Z",
      "tags": [],
      "nodes": {
        "node_uuid_1": {
          "loc": [x1, y1]
        },
        "node_uuid_2": {
          "loc": [x2, y2]
        },
        "node_uuid_3": {
          "loc": [x3, y3]
        }
        // ...
        // total 8 nodes = 8 keypoints
      }
    }
  ]
}

'''