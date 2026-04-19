import cv2
import json
import numpy as np
from ultralytics import YOLO
from ml.parking_logic import check_parking_status, find_nearest_free_slot

# Initialize YOLOv8 Nano
model = YOLO('ml/yolov8n.pt') 
TARGET_CLASSES = [2, 3, 5, 7, 15, 16]

# Define where your entry gate is on the image (X, Y coordinates)
# You will need to tweak these numbers based on your specific photo!
ENTRY_POINT = (50, 500) 

def load_zones(filepath='data/parking_zones.json'):
    try:
        with open(filepath, 'r') as f:
            # Check if the file is completely empty before trying to load it
            content = f.read().strip()
            if not content:
                print("Warning: parking_zones.json is empty. Run roi_selector.py")
                return {}
            return json.loads(content)
    except FileNotFoundError:
        print("Warning: parking_zones.json not found. Run roi_selector.py")
        return {}
    except json.JSONDecodeError:
        print("Warning: parking_zones.json is corrupted. Run roi_selector.py")
        return {}

def process_static_image(image_path="data/sample_parking.png"):
    zones = load_zones()
    frame = cv2.imread(image_path)
    
    if frame is None:
        print(f"Error: Could not load image at {image_path}")
        return None

    # 1. Run YOLO inference
    results = model(frame, classes=TARGET_CLASSES, verbose=False)[0]
    
    detections = []
    for box in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = box
        detections.append([int(x1), int(y1), int(x2), int(y2), int(class_id)])

    # 2. Determine status of each slot
    slot_status = check_parking_status(detections, zones)

    # 3. Find the nearest free slot to the Entry Gate
    nearest_slot, target_center = find_nearest_free_slot(slot_status, zones, ENTRY_POINT)

    # 4. Draw the dashboard overlays
    for slot_id, pts in zones.items():
        pts_array = np.array(pts, np.int32)
        status = slot_status[slot_id]
        
        if status == "Free":
            color = (0, 255, 0) # Green
        elif status == "Occupied":
            color = (0, 0, 255) # Red
        else:
            color = (0, 255, 255) # Yellow

        cv2.polylines(frame, [pts_array], isClosed=True, color=color, thickness=2)
        
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts_array], color)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        cv2.putText(frame, f"{slot_id}", tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    # 5. Draw Entry Point and Path to Nearest Slot
    cv2.circle(frame, ENTRY_POINT, 10, (255, 0, 255), -1) # Purple dot for Entry
    cv2.putText(frame, "ENTRY", (ENTRY_POINT[0] - 20, ENTRY_POINT[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

    if nearest_slot:
        # Draw a line from Entry to the nearest free slot
        cv2.arrowedLine(frame, ENTRY_POINT, target_center, (255, 255, 0), 4, tipLength=0.05)
        cv2.putText(frame, f"Directing to {nearest_slot}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
    else:
        cv2.putText(frame, "PARKING FULL", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # Encode image to send to the Flask web dashboard
    ret, buffer = cv2.imencode('.jpg', frame)
    return buffer.tobytes()