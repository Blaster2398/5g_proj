from flask import Flask, render_template, Response, request, jsonify
import os
from stream_engine import StreamEngine, RuntimeConfig

app = Flask(__name__)
config = RuntimeConfig()
stream_engine = StreamEngine(config=config)


# Feature 20: Optional API key protection for operational endpoints
def _require_api_key():
    required_key = os.getenv("PARKING_API_KEY", "")
    if not required_key:
        return None
    provided_key = request.headers.get("X-API-Key", "")
    if provided_key != required_key:
        return jsonify({"error": "Unauthorized"}), 401
    return None

@app.route('/')
def index():
    return render_template('index.html')

# Feature 3: Static image endpoint now uses unified engine pipeline
@app.route('/process_image')
def process_image():
    # Get the image filename from the URL (default to baseline)
    image_name = request.args.get('img', 'baseline.png')
    image_path = os.path.join('data', image_name)
    
    image_bytes = stream_engine.process_image_file(image_path)
    if image_bytes:
        return Response(image_bytes, mimetype='image/jpeg')
    else:
        return f"Error: Could not process {image_path}", 404

# Feature 13: Real RTSP MJPEG endpoint wired to production engine
@app.route('/video_feed')
def video_feed():
    return Response(
        stream_engine.generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame',
    )

# Feature 14: Health + readiness + status API
@app.route('/health')
def health():
    unauthorized = _require_api_key()
    if unauthorized:
        return unauthorized
    health_data = stream_engine.get_health()
    code = 200 if health_data.get("status") in ("running", "starting", "reconnecting") else 503
    return jsonify(health_data), code

@app.route('/ready')
def ready():
    unauthorized = _require_api_key()
    if unauthorized:
        return unauthorized
    zones_loaded = bool(stream_engine.zones)
    return jsonify({"ready": zones_loaded}), (200 if zones_loaded else 503)

# Feature 15: Slot summary API for dashboard integration
@app.route('/api/slot_summary')
def slot_summary():
    unauthorized = _require_api_key()
    if unauthorized:
        return unauthorized
    # Feature 32: Auto-warm static mode so dashboard summary is never stuck at 'starting'
    health_data = stream_engine.get_health()
    if health_data.get("status") == "starting" and config.stream_mode == "static":
        stream_engine.process_image_file(config.static_image)
        health_data = stream_engine.get_health()
    return jsonify(
        {
            "free_slots": health_data.get("free_slots", 0),
            "occupied_slots": health_data.get("occupied_slots", 0),
            "unknown_slots": health_data.get("unknown_slots", 0),
            "fps": health_data.get("fps", 0.0),
            "status": health_data.get("status", "unknown"),
        }
    )

# Feature 25: API for manual entry/exit updates from dashboard clicks
@app.route('/api/manual_points', methods=['GET', 'POST'])
def manual_points():
    unauthorized = _require_api_key()
    if unauthorized:
        return unauthorized
    if request.method == 'GET':
        return jsonify(
            {
                "entry_point": list(stream_engine.config.entry_point),
                "exit_point": list(stream_engine.config.exit_point),
                "entry_line": [list(stream_engine.config.entry_line[0]), list(stream_engine.config.entry_line[1])],
                "exit_line": [list(stream_engine.config.exit_line[0]), list(stream_engine.config.exit_line[1])],
            }
        )
    payload = request.get_json(silent=True) or {}
    updated = stream_engine.update_manual_points(payload)
    return jsonify(
        {
            "entry_point": list(updated["entry_point"]),
            "exit_point": list(updated["exit_point"]),
            "entry_line": [list(updated["entry_line"][0]), list(updated["entry_line"][1])],
            "exit_line": [list(updated["exit_line"][0]), list(updated["exit_line"][1])],
        }
    )

# Feature 26: API to return detailed route instructions and vehicle dimensions
@app.route('/api/path_details')
def path_details():
    unauthorized = _require_api_key()
    if unauthorized:
        return unauthorized
    return jsonify({"paths": stream_engine.get_latest_paths()})

if __name__ == "__main__":
    port = int(os.getenv("PARKING_APP_PORT", "5000"))
    debug = os.getenv("PARKING_DEBUG", "1") == "1"
    app.run(host='0.0.0.0', port=port, debug=debug)