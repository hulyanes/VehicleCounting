import cv2
import os
import torch
import numpy as np
from ultralytics import YOLO
from torchvision import transforms, models
import torch.nn as nn

# =========================
# PATHS
# =========================
YOLO_MODEL_PATH = "VehicleDetectionFinal3.pt"
CLASSIFIER_PATH = "classifier_model.pt"

TEST_VIDEOS_DIR = "testvids"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# =========================
# DEVICE
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# LOAD CLASSIFIER
# =========================
ckpt = torch.load(CLASSIFIER_PATH, map_location=DEVICE)
CLASS_NAMES = ckpt["class_names"]

classifier = models.resnet18(weights=None)
classifier.fc = nn.Linear(classifier.fc.in_features, len(CLASS_NAMES))
classifier.load_state_dict(ckpt["model_state"])
classifier = classifier.to(DEVICE)
classifier.eval()

# =========================
# TRANSFORM
# =========================
clf_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# LOAD YOLO
# =========================
yolo = YOLO(YOLO_MODEL_PATH)

# =========================
# MAIN LOOP
# =========================
for video in os.listdir(TEST_VIDEOS_DIR):
    if not video.lower().endswith(".mp4"):
        continue

    # =========================
    # COUNTERS
    # =========================
    COUNTERS = {c: 0 for c in CLASS_NAMES}
    counted_ids = set()

    cap = cv2.VideoCapture(os.path.join(TEST_VIDEOS_DIR, video))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(
        os.path.join(RESULTS_DIR, video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.3,
            iou=0.5,
            verbose=False
        )

        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                if box.id is None:
                    continue

                track_id = int(box.id[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                from PIL import Image

                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crop_pil = Image.fromarray(crop_rgb)

                crop_tensor = clf_tf(crop_pil).unsqueeze(0).to(DEVICE)


                with torch.no_grad():
                    logits = classifier(crop_tensor)
                    cls_id = logits.argmax(dim=1).item()
                    cls_name = CLASS_NAMES[cls_id]

                if track_id not in counted_ids:
                    COUNTERS[cls_name] += 1
                    counted_ids.add(track_id) 

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{cls_name} ID:{track_id}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2
                )

        # UI COUNTERS
        y0 = 70
        for cls, cnt in COUNTERS.items():
            cv2.putText(frame,f"{cls}: {cnt}",
                (20, y0),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                5
            )
            cv2.putText(
                frame,
                f"{cls}: {cnt}",
                (20, y0),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            y0 += 22

        out.write(frame)
        cv2.imshow("Two-Stage Vehicle Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

print("✅ Two-stage inference complete")
