import cv2
import json
import os

os.makedirs('data', exist_ok=True)

VIDEO_SOURCE = "data/sample_parking.png" 
JSON_PATH = 'data/parking_zones.json'

points = []
slots = {}
slot_count = 1

def draw_polygon(event, x, y, flags, param):
    global points, slots, slot_count
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
    elif event == cv2.EVENT_RBUTTONDOWN: 
        if points: points.pop()

cap = cv2.VideoCapture(VIDEO_SOURCE)
ret, frame = cap.read()
if not ret:
    print("Failed to grab frame. Check video source.")
    exit()

cv2.namedWindow("Select ROI - Press 's' to save a slot, 'q' to quit")
cv2.setMouseCallback("Select ROI - Press 's' to save a slot, 'q' to quit", draw_polygon)

while True:
    display_frame = frame.copy()
    
    for slot_id, pts in slots.items():
        cv2.polylines(display_frame, [cv2.UMat(pts).get()], isClosed=True, color=(255, 0, 0), thickness=2)
        cv2.putText(display_frame, slot_id, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

    for pt in points:
        cv2.circle(display_frame, tuple(pt), 4, (0, 0, 255), -1)
    if len(points) > 1:
        cv2.polylines(display_frame, [cv2.UMat(points).get()], isClosed=False, color=(0, 0, 255), thickness=2)

    cv2.imshow("Select ROI - Press 's' to save a slot, 'q' to quit", display_frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s') and len(points) == 4:
        slots[f"Slot_{slot_count}"] = points.copy()
        print(f"Saved Slot_{slot_count}")
        slot_count += 1
        points = [] 
    elif key == ord('q'):
        with open(JSON_PATH, 'w') as f:
            json.dump(slots, f, indent=4)
        print("Saved to parking_zones.json")
        break

cap.release()
cv2.destroyAllWindows()