import os
import time
import threading
import subprocess
import cv2
import glob
import shutil


os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
# ==========================================
# ⚙️ THE MASTER SWITCH
# ==========================================
# 0 = Static Mode (Uses original data/ folder)
# 1 = Live Polling Mode (Captures from camera to live_data/ folder)
MODE = 1

# Live Capture Configuration
# Change '0' to your RTSP URL (either hardcoded or via environment variable)
CAMERA_SOURCE = os.getenv("PARKING_RTSP_URL", "rtsp://admin:admin123@12.0.0.84:554/avstream/channel=1/stream=1.sdp") 


CAPTURE_INTERVAL = 10   # Seconds between frame captures
MAX_FRAMES = 10         # Rolling buffer size
LIVE_DIR = "live_data"
DATA_DIR = "data"   

def setup_live_environment():
    """Creates live_data folder and safely copies mapping files so they are isolated."""
    os.makedirs(LIVE_DIR, exist_ok=True)
    mappings = ['parking_zones.json', 'road_network.json', 'runtime_points.json']
    for file in mappings:
        src = os.path.join(DATA_DIR, file)
        dst = os.path.join(LIVE_DIR, file)
        # If live_data mappings don't exist yet, seed them from the static folder
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy(src, dst)

def capture_loop():
    """Runs in the background: Captures, saves, prunes frames, and handles disconnects."""
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    frame_counter = 1
    
    while True:
        ret, frame = cap.read()
        if ret:
            filename = os.path.join(LIVE_DIR, f"live_{frame_counter:05d}.jpg")
            cv2.imwrite(filename, frame)
            print(f"[CAMERA] Captured {filename}")
            
            cv2.imwrite(os.path.join(LIVE_DIR, "latest.jpg"), frame)
            
            saved_frames = sorted(glob.glob(os.path.join(LIVE_DIR, "live_*.jpg")))
            if len(saved_frames) > MAX_FRAMES:
                for old_frame in saved_frames[:-MAX_FRAMES]:
                    os.remove(old_frame)
                    print(f"[CLEANUP] Deleted {old_frame}")
            
            frame_counter += 1
            time.sleep(CAPTURE_INTERVAL)
        else:
            # SAFETY FIX: If the camera drops, release it, wait, and reconnect
            print("[CAMERA] Warning: Stream dropped or timed out. Reconnecting in 3s...")
            cap.release()
            time.sleep(3)
            cap = cv2.VideoCapture(CAMERA_SOURCE)

if __name__ == "__main__":
    if MODE == 0:
        print(">>> BOOTING: STATIC MODE (0) <<<")
        os.environ["PARKING_STREAM_MODE"] = "static"
        os.environ["PARKING_DATA_DIR"] = DATA_DIR
        os.environ["PARKING_ZONE_PATH"] = f"{DATA_DIR}/parking_zones.json"
        os.environ["PARKING_ROAD_PATH"] = f"{DATA_DIR}/road_network.json"
        os.environ["PARKING_MANUAL_POINTS_PATH"] = f"{DATA_DIR}/runtime_points.json"
        subprocess.run(["python", "app.py"])
        
    elif MODE == 1:
        print(">>> BOOTING: LIVE POLLING MODE (1) <<<")
        setup_live_environment()
        
        # Start the background camera thread
        capture_thread = threading.Thread(target=capture_loop, daemon=True)
        capture_thread.start()
        
        # Route the engine's environment variables to the new live folder
        os.environ["PARKING_STREAM_MODE"] = "static" # We still process them as static images!
        os.environ["PARKING_DATA_DIR"] = LIVE_DIR
        os.environ["PARKING_ZONE_PATH"] = f"{LIVE_DIR}/parking_zones.json"
        os.environ["PARKING_ROAD_PATH"] = f"{LIVE_DIR}/road_network.json"
        os.environ["PARKING_MANUAL_POINTS_PATH"] = f"{LIVE_DIR}/runtime_points.json"
        
        subprocess.run(["python", "app.py"])