import cv2
import numpy as np
import math

def check_parking_status(detections, parking_zones):
    """Evaluates which parking zones are occupied based on YOLO detections."""
    status = {slot_id: "Free" for slot_id in parking_zones.keys()}
    
    for box in detections:
        x1, y1, x2, y2, class_id = box
        center_x = int((x1 + x2) / 2)
        center_y = int(y2 - ((y2 - y1) * 0.1)) 
        
        for slot_id, points in parking_zones.items():
            pts_array = np.array(points, np.int32)
            is_inside = cv2.pointPolygonTest(pts_array, (center_x, center_y), False)
            
            if is_inside >= 0:
                if class_id in [15, 16]: # Cat, Dog
                    status[slot_id] = "Obstacle"
                else:
                    status[slot_id] = "Occupied"
                break 

    return status

def find_nearest_free_slot(parking_status, parking_zones, entry_point):
    """Finds the closest free slot to the designated entry coordinates."""
    min_dist = float('inf')
    nearest_slot = None
    nearest_center = None

    for slot_id, status in parking_status.items():
        if status == "Free":
            # Calculate the centroid (center) of the parking polygon
            pts = parking_zones[slot_id]
            center_x = int(sum([p[0] for p in pts]) / len(pts))
            center_y = int(sum([p[1] for p in pts]) / len(pts))
            
            # Calculate Euclidean distance
            dist = math.sqrt((center_x - entry_point[0])**2 + (center_y - entry_point[1])**2)
            
            if dist < min_dist:
                min_dist = dist
                nearest_slot = slot_id
                nearest_center = (center_x, center_y)

    return nearest_slot, nearest_center