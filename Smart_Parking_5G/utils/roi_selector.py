import cv2
import json
import os
import numpy as np

os.makedirs('data', exist_ok=True)

IMAGE_SOURCE = "data/baseline.png" 
JSON_PATH = 'data/parking_zones.json'

points = []
slots = {}
slot_count = 1

def draw_polygon(event, x, y, flags, param):
    global points, slots
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append([x, y])
    elif event == cv2.EVENT_RBUTTONDOWN: 
        if points: points.pop()

frame = cv2.imread(IMAGE_SOURCE)

if frame is None:
    print(f"Error: Could not read the image exactly at '{IMAGE_SOURCE}'.")
    exit()

window_name = "Smart Parking - ROI Selector GUI"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1280, 720)
cv2.setMouseCallback(window_name, draw_polygon)

while True:
    display_frame = frame.copy()
    
    # --- 1. DRAW THE INSTRUCTION GUI PANEL ---
    overlay = display_frame.copy()
    cv2.rectangle(overlay, (0, 0), (display_frame.shape[1], 140), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)
    
    cv2.putText(display_frame, "PARKING MAPPING TOOL", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(display_frame, "MOUSE: [Left Click] Add corner dot  |  [Right Click] Delete last dot", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    cv2.putText(display_frame, "KEYBOARD: Press 'S' -> Save the current 4 dots as a parking slot", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(display_frame, "KEYBOARD: Press 'U' -> Undo/Delete the LAST saved slot  |  Press 'Q' -> Save All & Quit", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
    
    # --- 2. DRAW PREVIOUSLY SAVED SLOTS ---
    for slot_id, pts in slots.items():
        pts_array = np.array(pts, np.int32) 
        cv2.polylines(display_frame, [pts_array], isClosed=True, color=(255, 0, 0), thickness=2)
        
        M = cv2.moments(pts_array)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            cv2.putText(display_frame, slot_id, (cx-25, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # --- 3. DRAW THE CURRENT POINTS ---
    for pt in points:
        cv2.circle(display_frame, tuple(pt), 6, (0, 255, 255), -1) 
        
    if len(points) > 0:
        pts_array = np.array(points, np.int32)
        if len(points) == 4:
            cv2.polylines(display_frame, [pts_array], isClosed=True, color=(0, 255, 0), thickness=3)
        elif len(points) > 1:
            cv2.polylines(display_frame, [pts_array], isClosed=False, color=(0, 0, 255), thickness=2)

    cv2.imshow(window_name, display_frame)
    key = cv2.waitKey(1) & 0xFF

    # --- NEW: CHECK IF WINDOW WAS CLOSED VIA THE 'X' BUTTON ---
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        with open(JSON_PATH, 'w') as f:
            json.dump(slots, f, indent=4)
        print("Window closed. Successfully saved to parking_zones.json")
        break

    # --- 4. KEYBOARD SHORTCUTS ---
    if key == ord('s') and len(points) == 4:
        slots[f"Slot_{slot_count}"] = points.copy()
        print(f"Saved Slot_{slot_count}")
        slot_count += 1
        points = [] 
        
    elif key == ord('u'):
        if slot_count > 1:
            slot_count -= 1
            deleted_slot = f"Slot_{slot_count}"
            slots.pop(deleted_slot)
            print(f"DELETED {deleted_slot}")
            
    elif key == ord('q'):
        with open(JSON_PATH, 'w') as f:
            json.dump(slots, f, indent=4)
        print("Successfully saved all coordinates to parking_zones.json")
        break

cv2.destroyAllWindows()