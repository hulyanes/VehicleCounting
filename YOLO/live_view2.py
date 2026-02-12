import os
import time
import cv2
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO
from torchvision import models, transforms
from datetime import datetime
import pandas as pd

HOURLY_FILE = "vehicle_hourly_summary.csv"
DAILY_FILE = "vehicle_daily_summary.csv"

hourly_totals = defaultdict(int)
daily_totals = defaultdict(int)

current_hour = datetime.now().hour
current_date = datetime.now().date()


# =========================
# FORCE RTSP TCP
# =========================
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# =========================
# CONFIG
# =========================
CCTV_URL = "rtsp://ICTRTSP:Rtsp80051@172.31.255.56:554/Streaming/Channels/101"

DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

TARGET_FPS = 5
FRAME_DELAY = int(1000 / TARGET_FPS)

YOLO_MODEL_PATH = "yolo11n.pt"
CLASSIFIER_MODEL_PATH = "classifier_model.pt"

IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = ['Bus', 'Car', 'Jeep', 'Motor', 'Tricycle', 'Truck', 'Van']
NUM_CLASSES = len(CLASS_NAMES)

CONF_THRES = 0.25
CLASSIFIER_CONF_THRES = 0.6
SMOOTHING_WINDOW = 7
ID_TIMEOUT = 3.0  # seconds before removing stale track

# =========================
# BUILD CLASSIFIER
# =========================
def build_classifier():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model

# =========================
# LOAD MODELS
# =========================
print("🚀 Loading YOLO detector...")
detector = YOLO(YOLO_MODEL_PATH)

print("🚀 Loading classifier...")
classifier = build_classifier().to(DEVICE)

ckpt = torch.load(CLASSIFIER_MODEL_PATH, map_location=DEVICE)
classifier.load_state_dict(ckpt["model_state"])
classifier.eval()

# Safety check
assert ckpt["class_names"] == CLASS_NAMES, \
    "❌ Class name mismatch between checkpoint and config!"

print("✅ Models loaded successfully")

# =========================
# TRANSFORMS
# =========================
clf_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# STREAM HANDLING
# =========================
def open_stream(url):
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap

print("🔌 Connecting to CCTV...")
cap = open_stream(CCTV_URL)

while not cap.isOpened():
    print("⏳ Waiting for CCTV stream...")
    time.sleep(2)
    cap.release()
    cap = open_stream(CCTV_URL)

print("✅ CCTV stream connected")

# =========================
# TRACK STATE
# =========================
track_label_hist = defaultdict(lambda: deque(maxlen=SMOOTHING_WINDOW))
track_final_label = {}
track_last_seen = {}
vehicle_counter = defaultdict(int)
counted_ids = set()

# =========================
# MAIN LOOP
# =========================
while True:
    start_loop = time.time()

    # Drop stale frames
    for _ in range(2):
        cap.grab()

    ret, frame = cap.read()

    if not ret:
        print("⚠️ Stream lost — reconnecting...")
        cap.release()
        time.sleep(2)
        cap = open_stream(CCTV_URL)
        continue

    current_time = time.time()

    results = detector.track(
        frame,
        persist=True,
        conf=CONF_THRES,
        tracker="bytetrack.yaml",
        verbose=False
    )

    if results and results[0].boxes is not None:
        for box in results[0].boxes:

            if box.id is None:
                continue

            track_id = int(box.id[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Clamp bounding box
            h, w = frame.shape[:2]
            x1 = max(0, min(x1, w))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h))
            y2 = max(0, min(y2, h))

            if x2 <= x1 or y2 <= y1:
                continue

            track_last_seen[track_id] = current_time

            # If already stable, skip reclassification
            if track_id not in track_final_label:

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crop_tensor = clf_tf(crop_rgb).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    logits = classifier(crop_tensor)
                    probs = torch.softmax(logits, dim=1)[0]
                    conf, pred = torch.max(probs, dim=0)

                if conf < CLASSIFIER_CONF_THRES:
                    continue

                label = CLASS_NAMES[pred.item()]
                track_label_hist[track_id].append(label)

                # Stabilize label
                if len(track_label_hist[track_id]) == SMOOTHING_WINDOW:
                    stable_label = max(
                        set(track_label_hist[track_id]),
                        key=track_label_hist[track_id].count
                    )
                    track_final_label[track_id] = stable_label

                    if track_id not in counted_ids:
                        vehicle_counter[stable_label] += 1
                        hourly_totals[stable_label] += 1
                        daily_totals[stable_label] += 1
                        counted_ids.add(track_id)

            # Draw result
            if track_id in track_final_label:
                label = track_final_label[track_id]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{label} ID:{track_id}",
                    (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2
                )

    # =========================
    # CLEAN STALE TRACKS
    # =========================
    expired_ids = [
        tid for tid, last in track_last_seen.items()
        if current_time - last > ID_TIMEOUT
    ]

    for tid in expired_ids:
        track_last_seen.pop(tid, None)
        track_label_hist.pop(tid, None)
        track_final_label.pop(tid, None)

    # =========================
    # DRAW COUNTERS
    # =========================
    y0 = 30
    cv2.rectangle(frame, (10, 10), (260, y0 + 25 * len(CLASS_NAMES)), (0, 0, 0), -1)
    cv2.putText(frame, "Vehicle Count", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    for i, cls in enumerate(CLASS_NAMES):
        cv2.putText(
            frame,
            f"{cls}: {vehicle_counter[cls]}",
            (20, y0 + 25 * i),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )

    # Resize display
    display_frame = cv2.resize(
        frame,
        (DISPLAY_WIDTH, DISPLAY_HEIGHT),
        interpolation=cv2.INTER_AREA
    )

    cv2.imshow("CCTV Vehicle Detection", display_frame)

    # FPS control
    elapsed = time.time() - start_loop
    delay = max(1, FRAME_DELAY - int(elapsed * 1000))

    now = datetime.now()

    # -------------------------
    # HOURLY CHECK
    # -------------------------
    if now.hour != current_hour:
        save_hourly_summary(current_hour, current_date)
        current_hour = now.hour

    # -------------------------
    # MIDNIGHT CHECK
    # -------------------------
    if now.date() != current_date:
        save_daily_summary(current_date)
        current_date = now.date()

    if cv2.waitKey(delay) & 0xFF == ord("q"):
        break


def save_hourly_summary(hour, date):
    global hourly_totals

    row = {
        "date": str(date),
        "hour": f"{hour:02d}:00",
        **hourly_totals
    }

    df = pd.DataFrame([row])

    if os.path.exists(HOURLY_FILE):
        df.to_csv(HOURLY_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(HOURLY_FILE, index=False)

    print(f"📊 Hourly summary saved for {hour:02d}:00")

    # Reset hourly totals
    hourly_totals = defaultdict(int)


def save_daily_summary(date):
    global daily_totals

    row = {
        "date": str(date),
        **daily_totals
    }

    df = pd.DataFrame([row])

    if os.path.exists(DAILY_FILE):
        df.to_csv(DAILY_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(DAILY_FILE, index=False)

    print(f"🗓 Daily summary saved for {date}")

    # Reset daily totals
    daily_totals = defaultdict(int)


# =========================
# CLEANUP
# =========================
cap.release()
cv2.destroyAllWindows()
