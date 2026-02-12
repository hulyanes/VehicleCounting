# *!! This is just a test app.py for model and ui connection for uploaded videos. Not guaranteed to work for lower GPU capacity.

import os
import uuid
import shutil
import cv2
import torch
import numpy as np

from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image

# ==========================================
# CONFIG
# ==========================================

YOLO_MODEL_PATH = "yolo11n.pt"
CLASSIFIER_PATH = "classifier_model.pt"

UPLOAD_DIR = "uploads"
RESULTS_DIR = "results"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔥 Using device: {DEVICE}")

# ==========================================
# LOAD MODELS (LOAD ONCE)
# ==========================================

print("Loading YOLO...")
yolo = YOLO(YOLO_MODEL_PATH)

print("Loading classifier...")
ckpt = torch.load(CLASSIFIER_PATH, map_location=DEVICE)
CLASS_NAMES = ckpt["class_names"]

classifier = models.resnet18(weights=None)
classifier.fc = nn.Linear(classifier.fc.in_features, len(CLASS_NAMES))
classifier.load_state_dict(ckpt["model_state"])
classifier = classifier.to(DEVICE)
classifier.eval()

clf_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

print("✅ Models loaded successfully")

# ==========================================
# FASTAPI INIT
# ==========================================

app = FastAPI()

# Serve processed videos
app.mount("/results", StaticFiles(directory=RESULTS_DIR), name="results")

# ==========================================
# TWO-STAGE PROCESS FUNCTION
# ==========================================

def run_two_stage_detection(input_path, output_path):

    COUNTERS = {c: 0 for c in CLASS_NAMES}
    counted_ids = set()

    cap = cv2.VideoCapture(input_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps == 0:
        fps = 25

    out = cv2.VideoWriter(
        output_path,
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

                # Draw box
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

        # Draw counters on screen
        y0 = 50
        for cls, cnt in COUNTERS.items():
            cv2.putText(
                frame,
                f"{cls}: {cnt}",
                (20, y0),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            y0 += 30

        out.write(frame)

    cap.release()
    out.release()

    return COUNTERS

# ==========================================
# API ENDPOINT
# ==========================================

@app.post("/process-video/")
async def process_video(file: UploadFile = File(...)):

    input_filename = f"{uuid.uuid4()}.mp4"
    output_filename = f"{uuid.uuid4()}_processed.mp4"

    input_path = os.path.join(UPLOAD_DIR, input_filename)
    output_path = os.path.join(RESULTS_DIR, output_filename)

    # Save uploaded file
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run detection
    counters = run_two_stage_detection(input_path, output_path)

    return {
        "video_url": f"/results/{output_filename}",
        "counters": counters
    }

# ==========================================
# RUN (FOR WINDOWS SAFETY)
# ==========================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
