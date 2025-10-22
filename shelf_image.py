import cv2
import numpy as np
from ultralytics import YOLO

# ---------------------------
IMAGE_PATH = "images/2.png"   # put your image path here
MODEL_PATH = "yolov8n.pt"
ROWS, COLS = 3, 3           # 3x3 grid
# ---------------------------

# Load model
model = YOLO(MODEL_PATH)

# Read image
frame = cv2.imread(IMAGE_PATH)
if frame is None:
    raise Exception("Image not found. Check IMAGE_PATH.")

frame_height, frame_width, _ = frame.shape
rack_h = frame_height // ROWS
rack_w = frame_width // COLS

# Run YOLO detection
results = model(frame, stream=True)
label_matrix = [["Empty" for _ in range(COLS)] for _ in range(ROWS)]

# Process detections
for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = model.names[cls]

        if conf > 0.5:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"{label}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            # Find grid zone
            cx, cy = (x1+x2)//2, (y1+y2)//2
            col_idx = min(cx // rack_w, COLS-1)
            row_idx = min(cy // rack_h, ROWS-1)
            label_matrix[row_idx][col_idx] = label

# Draw grid lines
for r in range(1, ROWS):
    cv2.line(frame, (0, r*rack_h), (frame_width, r*rack_h), (255,255,0), 2)
for c in range(1, COLS):
    cv2.line(frame, (c*rack_w, 0), (c*rack_w, frame_height), (255,255,0), 2)

# Display item names per grid
for r in range(ROWS):
    for c in range(COLS):
        item = label_matrix[r][c]
        color = (0,255,0) if item != "Empty" else (0,0,255)
        cv2.putText(frame, item, ((c*rack_w)+40, (r*rack_h)+80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

# Print grid info
print("\n--- Rack Status ---")
for r in range(ROWS):
    print(f"Rack {r+1}: {' | '.join(label_matrix[r])}")

# Show image with overlays
cv2.imshow("Shelf Detection (Image Mode)", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
