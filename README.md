
# 5G Smart Parking & Autonomous Routing System

### Edge-Computed Computer Vision Pipeline for Real-Time Vehicle Tracking and Topological Pathfinding

This project is a full-stack, edge-to-cloud IoT application designed to monitor parking lot occupancy and dynamically route moving vehicles to the nearest available slot (or exit) in real time. It leverages optimized Convolutional Neural Networks (CNNs) and graph traversal algorithms to run efficiently on constrained edge hardware over wireless networks.

---

## Core Engineering Features

### 1. Tri-Modal Edge Architecture
The system uses a decoupled Master Orchestrator (`run_system.py`) to isolate the OpenCV capture loop from the Flask web server, eliminating UI freezing and buffer overflows.

Supported modes:
- **Static Mode**: Simulates historical data for offline testing
- **RTSP Mode**: Streams from IP cameras (see Known Issues)
- **Phone Edge Mode**: Uses a smartphone as a wireless MJPEG camera node

---

### 2. Parallax-Corrected Spatial Detection (YOLOv8)
Standard bounding boxes fail in angled views due to parallax.

This system implements a **70/30 Tire-Anchor Algorithm**:
- Extracts anchor points from the bottom 15% of bounding boxes
- Combines:
  - Anchor-polygon intersection (70%)
  - IoU overlap (30%)

This prevents false slot occupancy from overlapping vehicle roofs.

---

### 3. Topological Routing Using A* Algorithm
Instead of straight-line routing, vehicles are mapped onto a predefined graph (`road_network.json`).

- Uses A* (A-Star) with Euclidean heuristic
- Computes:
  - Path to nearest available slot
  - Path to exit simultaneously

Includes **Historical Spawn Intent** to infer whether a vehicle is entering or leaving.

---

### 4. Digital Debounce Filtering (State Hysteresis)
To stabilize noisy AI predictions:

- 3 consecutive frames → mark as Occupied
- 5 consecutive frames → mark as Free
- 20-frame occlusion → mark as Unknown

Implemented via a custom `SlotStateSmoother` state machine.

---

## Technology Stack

**Backend**
- Python 3.9+
- Flask (REST APIs, WebSockets)

**Computer Vision**
- OpenCV (cv2)
- NumPy

**Machine Learning**
- YOLOv8n (Ultralytics)

**Frontend**
- HTML5, CSS3, Vanilla JavaScript

**Algorithms**
- A* Pathfinding (`heapq`)
- Euclidean heuristics

---

## Installation and Setup

### Prerequisites
- Python 3.9+
- Git
- Smartphone with IP Webcam app (for Phone Edge mode)

---

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/Smart_Parking_5G.git
cd Smart_Parking_5G
````

---

### Step 2: Create Virtual Environment (Recommended)

**Windows**

```cmd
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS**

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

Note: YOLOv8 model weights will auto-download on first run.

---

## Execution and Usage

### Start the System

```bash
python run_system.py
```

Open dashboard:

```
http://localhost:5000
```

---

## Operating Modes

### Static Mode

* Uses `/data/baseline.png`
* Best for testing routing logic

---

### RTSP Mode

* Connects to IP camera

Set environment variable:

```bash
export PARKING_RTSP_URL="rtsp://admin:password@192.168.x.x:554/stream"
```

---

### Phone Edge Mode (Recommended)

Use smartphone as camera:

1. Connect phone and laptop to same Wi-Fi
2. Open IP Webcam app and start server
3. Update camera source in `run_system.py`:

   ```
   http://192.168.x.x:8080/video
   ```
4. Select "Phone Edge" from dashboard

---

## System Calibration

1. Click "Calibrate" on dashboard
2. Use OpenCV window controls:

   * Draw parking slots (Green)
   * Draw roads (Yellow)
   * Entry point (Magenta)
   * Exit point (Orange)

Controls:

* Left-click: draw/place
* Right-click: delete
* M: toggle mode
* E: entry point
* X: exit point
* Q: save and exit

Changes reload automatically.

---

## Known Issues

### RTSP Instability

RTSP mode may face packet drops or connection issues.

Workaround: Use Phone Edge mode for stable performance.

---

### Firewall Blocking

If phone stream is not detected:

* Allow port 8080 in firewall

---

### OpenCV Crash on Movement

Ensure all coordinates are cast to integers before drawing.

---

## Developed By

Aryan Patel

Shreyash Bharati

Yash Varshney

ECE Department


