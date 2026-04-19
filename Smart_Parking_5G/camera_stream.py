import cv2
import json
import numpy as np
from ultralytics import YOLO
from ml.parking_logic import check_parking_status

# Initialize YOLOv8 Nano
model = YOLO('ml/yolov8n.pt') # Will auto-download the first time

# Define target classes: 2:Car, 3:Motorcycle, 5:Bus, 7:Truck, 15:Cat, 16:Dog
TARGET_CLASSES = [2, 3, 5, 7, 15, 16]

def load_zones(filepath='data/parking_zones.json'):
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Zones not found. Run utils/roi_selector.py first.")
        return {}

def generate_frames():
    zones = load_zones()
    # Replace with your RTSP feed
    cap = cv2.VideoCapture("rtsp://admin:password@192.168.1.100:554/stream")

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Run YOLO inference
        results = model(frame, classes=TARGET_CLASSES, verbose=False)[0]
        
        detections = []
        for box in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = box
            detections.append([int(x1), int(y1), int(x2), int(y2), int(class_id)])

        # Determine status of each slot
        slot_status = check_parking_status(detections, zones)

        # Draw the dashboard overlays
        for slot_id, pts in zones.items():
            pts_array = np.array(pts, np.int32)
            status = slot_status[slot_id]
            
            # Colors: BGR format
            if status == "Free":
                color = (0, 255, 0) # Green
            elif status == "Occupied":
                color = (0, 0, 255) # Red
            else:
                color = (0, 255, 255) # Yellow (Obstacle/Animal)

            cv2.polylines(frame, [pts_array], isClosed=True, color=color, thickness=2)
            
            # Semi-transparent overlay
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts_array], color)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

            # Draw label
            cv2.putText(frame, f"{slot_id}", tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # Encode frame for web streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')