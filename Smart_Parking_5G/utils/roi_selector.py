import cv2
import json
import os
import numpy as np
import math

# --- MODE SWITCH ---
# 0 = Map Static Data (Reads from data/baseline.png)
# 1 = Map Live Data (Reads from live_data/latest.jpg)
TARGET_MODE = 1 

DATA_DIR = "live_data" if TARGET_MODE == 1 else "data"
os.makedirs(DATA_DIR, exist_ok=True)

IMAGE_SOURCE = os.path.join(DATA_DIR, "latest.jpg" if TARGET_MODE == 1 else "baseline.png") 
JSON_SLOTS = os.path.join(DATA_DIR, 'parking_zones.json')
JSON_ROADS = os.path.join(DATA_DIR, 'road_network.json')
# --- Application State ---
mode = "SLOTS" # Starts in SLOTS mode

# Slot Data
points = []
slots = {}
slot_count = 1



# Road Network Data
nodes = {}
edges = []
node_count = 1
active_node = None

# --- NEW: UI State Variables ---
mouse_x, mouse_y = 0, 0
hover_node = None

# --- Load Existing Data ---
if os.path.exists(JSON_SLOTS):
    try:
        with open(JSON_SLOTS, 'r') as f:
            slots = json.load(f)
            if slots:
                slot_nums = [int(k.split('_')[1]) for k in slots.keys() if '_' in k]
                slot_count = max(slot_nums) + 1 if slot_nums else 1
    except: pass

if os.path.exists(JSON_ROADS):
    try:
        with open(JSON_ROADS, 'r') as f:
            road_data = json.load(f)
            nodes = road_data.get("nodes", {})
            edges = road_data.get("edges", [])
            if nodes:
                node_nums = [int(k.split('_')[1]) for k in nodes.keys() if '_' in k]
                node_count = max(node_nums) + 1 if node_nums else 1
    except: pass

def get_nearest_node(x, y, threshold=15):
    """Finds if the user clicked near an existing road node."""
    for n_id, pt in nodes.items():
        if math.hypot(pt[0]-x, pt[1]-y) < threshold:
            return n_id
    return None

def mouse_callback(event, x, y, flags, param):
    global points, slots, slot_count
    global nodes, edges, node_count, active_node
    global mouse_x, mouse_y, hover_node

    # Track mouse movement for hover effects and preview lines
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = x, y
        hover_node = get_nearest_node(x, y)

    # --- SLOTS LOGIC ---
    if mode == "SLOTS":
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append([x, y])
        elif event == cv2.EVENT_RBUTTONDOWN:
            if points: 
                points.pop()
            else:
                slot_to_delete = None
                for slot_id, pts in slots.items():
                    if cv2.pointPolygonTest(np.array(pts, np.int32), (x, y), False) >= 0:
                        slot_to_delete = slot_id
                        break
                if slot_to_delete:
                    del slots[slot_to_delete]

    # --- ROADS LOGIC ---
    elif mode == "ROADS":
        if event == cv2.EVENT_LBUTTONDOWN:
            if hover_node:
                # Clicked existing node
                if active_node and active_node != hover_node:
                    edge = sorted([active_node, hover_node])
                    if edge not in edges:
                        edges.append(edge)
                active_node = hover_node
            else:
                # Clicked empty space
                new_node_id = f"N_{node_count}"
                nodes[new_node_id] = [x, y]
                node_count += 1
                if active_node:
                    edges.append(sorted([active_node, new_node_id]))
                active_node = new_node_id

        elif event == cv2.EVENT_RBUTTONDOWN:
            if active_node:
                active_node = None # Break line
            elif hover_node:
                # Delete node and its edges
                del nodes[hover_node]
                edges = [e for e in edges if hover_node not in e]
                hover_node = None

frame = cv2.imread(IMAGE_SOURCE)
if frame is None:
    print(f"Error: Could not read '{IMAGE_SOURCE}'.")
    exit()

window_name = "Smart Parking - Mapping Tool"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1280, 720)
cv2.setMouseCallback(window_name, mouse_callback)

while True:
    display_frame = frame.copy()
    
    # --- 1. GUI OVERLAY ---
    # Made the black box taller (180px) to give the text more breathing room
    cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], 180), (0, 0, 0), -1)
    
    # Line 1: Mode indicator (Added cv2.LINE_AA to all text to make it smooth and readable)
    cv2.putText(display_frame, f"MODE: {mode}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(display_frame, "(Press 'M' to toggle between SLOTS and ROADS)", (260, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)
    
    # Line 2 & 3: Mode specific instructions
    if mode == "SLOTS":
        # UPDATED: Added "or Delete Slot" to the Right Click instructions
        cv2.putText(display_frame, "MOUSE: [Left Click] Add dot  |  [Right Click] Delete dot or Delete Slot", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(display_frame, "KEYBOARD: 'S' -> Save 4 dots as slot  |  'U' -> Undo last slot", (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
    else:
        # UPDATED: Added "Delete Node" and "Clear All ('C')" instructions
        cv2.putText(display_frame, "MOUSE: [Left] Drop/Connect  |  [Right] Break Line OR Delete Node", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        
        active_text = f"Active: {active_node}" if active_node else "Active: None (Click to start line)"
        cv2.putText(display_frame, active_text, (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA)
        cv2.putText(display_frame, "KEYBOARD: 'C' -> Clear all roads", (450, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)

    # Line 4: Quit instruction
    cv2.putText(display_frame, "KEYBOARD: Press 'Q' -> Save Everything & Quit", (20, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)

    
   # --- 2. DRAW SLOTS (Always visible) ---
    for slot_id, pts in slots.items():
        pts_array = np.array(pts, np.int32) 
        
        # FIXED: Changed color from dark gray to Bright Neon Green (0, 255, 0)
        cv2.polylines(display_frame, [pts_array], isClosed=True, color=(0, 255, 0), thickness=2)
        
        cx = int(np.mean([p[0] for p in pts]))
        cy = int(np.mean([p[1] for p in pts]))
        
        # FIXED: Changed text to bright White so you can read the slot names clearly
        cv2.putText(display_frame, slot_id, (cx-20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    # Draw active slot drawing points (the ones you are currently clicking)
    for pt in points:
        cv2.circle(display_frame, tuple(pt), 6, (0, 255, 255), -1) 
    if len(points) > 0:
        pts_array = np.array(points, np.int32)
        # Active drawing lines are Red until 4 dots are placed, then turn Green
        cv2.polylines(display_frame, [pts_array], isClosed=(len(points)==4), color=(0, 255, 0) if len(points)==4 else (0,0,255), thickness=2)

    # --- 3. DRAW ROADS (Always visible) ---
    for edge in edges:
        pt1 = tuple(nodes[edge[0]])
        pt2 = tuple(nodes[edge[1]])
        cv2.line(display_frame, pt1, pt2, (255, 255, 0), 2, cv2.LINE_AA) # Yellow roads
        
    # NEW: Draw a Live Preview Line to the mouse
    if mode == "ROADS" and active_node:
        start_pt = tuple(nodes[active_node])
        cv2.line(display_frame, start_pt, (mouse_x, mouse_y), (255, 255, 255), 1, cv2.LINE_AA)
        
    for n_id, pt in nodes.items():
        # Nodes turn Green when hovered, Red when active, Light Blue otherwise
        if n_id == hover_node:
            color, radius = (0, 255, 0), 8
        elif n_id == active_node:
            color, radius = (0, 0, 255), 8
        else:
            color, radius = (255, 255, 0), 5
        cv2.circle(display_frame, tuple(pt), radius, color, -1)

    cv2.imshow(window_name, display_frame)
    key = cv2.waitKey(1) & 0xFF

    # Save and Exit safely if 'X' button is clicked
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

    # Keyboard Controls
    if key == ord('m'):
        mode = "ROADS" if mode == "SLOTS" else "SLOTS"
        points = [] # Clear slot drawing points when switching
        active_node = None # Clear active node when switching
        
    elif key == ord('s') and mode == "SLOTS" and len(points) == 4:
        slots[f"Slot_{slot_count}"] = points.copy()
        slot_count += 1
        points = [] 
        
    elif key == ord('u') and mode == "SLOTS":
        if slot_count > 1:
            slot_count -= 1
            slots.pop(f"Slot_{slot_count}", None)
            
    # NEW: Clear all roads if 'C' is pressed while in ROADS mode
    elif key == ord('c') and mode == "ROADS":
        nodes.clear()
        edges.clear()
        active_node = None
        print("Cleared all road networks.")
            
    elif key == ord('q'):
        break

# Save JSON files
with open(JSON_SLOTS, 'w') as f:
    json.dump(slots, f, indent=4)
with open(JSON_ROADS, 'w') as f:
    json.dump({"nodes": nodes, "edges": edges}, f, indent=4)

print("\nSuccessfully saved data/parking_zones.json")
print("Successfully saved data/road_network.json")
cv2.destroyAllWindows()