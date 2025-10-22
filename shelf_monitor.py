import cv2
import numpy as np
from ultralytics import YOLO
import time

# ---------------------------
# SETTINGS
# ---------------------------
CAMERA_INDEX = 1   # 0 = laptop cam, 1 = external webcam
MODEL_PATH = 'yolov8n.pt'
ROWS, COLS = 3, 3  # 3x3 grid = 9 zones
FRAME_SKIP = 5     # process every 5th frame
DELAY_SEC = 0.4    # delay for smoother output
# ---------------------------

# Load YOLO model
model = YOLO(MODEL_PATH)

# Start webcam
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise Exception(f"Camera {CAMERA_INDEX} not detected. Try another index (0 or 1).")

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    frame_height, frame_width, _ = frame.shape
    rack_h = frame_height // ROWS
    rack_w = frame_width // COLS

    # Grid label matrix (store detected item names)
    label_matrix = [["Empty" for _ in range(COLS)] for _ in range(ROWS)]

    # Only run YOLO every few frames
    if frame_count % FRAME_SKIP == 0:
        results = model(frame, stream=True)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = model.names[cls]

                if conf > 0.5:
                    # Draw detection
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

                    # Determine which grid zone this item belongs to
                    cx, cy = (x1+x2)//2, (y1+y2)//2
                    col_idx = min(cx // rack_w, COLS-1)
                    row_idx = min(cy // rack_h, ROWS-1)

                    # Store the detected item name in that zone
                    label_matrix[row_idx][col_idx] = label

    # --- Draw grid overlay ---
    for r in range(1, ROWS):
        cv2.line(frame, (0, r*rack_h), (frame_width, r*rack_h), (255,255,0), 2)
    for c in range(1, COLS):
        cv2.line(frame, (c*rack_w, 0), (c*rack_w, frame_height), (255,255,0), 2)

    # --- Display item names in each cell ---
    for r in range(ROWS):
        for c in range(COLS):
            item = label_matrix[r][c]
            color = (0,255,0) if item != "Empty" else (0,0,255)
            text = item
            cv2.putText(frame, text, ((c*rack_w)+40, (r*rack_h)+80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    # Show window
    cv2.imshow("🧠 SmartStock Shelf Monitor (3x3 with Object Names)", frame)

    # Print current rack status in console
    if frame_count % FRAME_SKIP == 0:
        print("\n--- Rack Status ---")
        for r in range(ROWS):
            row_status = " | ".join(label_matrix[r])
            print(f"Rack {r+1}: {row_status}")

    # Delay for smoother speed
    time.sleep(DELAY_SEC)

    # Exit when 'q' pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
