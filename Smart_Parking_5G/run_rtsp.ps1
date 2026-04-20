param(
    [string]$RtspUrl = "rtsp://admin:password@192.168.1.100:554/stream"
)

$env:PARKING_STREAM_MODE = "rtsp"
$env:PARKING_RTSP_URL = $RtspUrl
$env:PARKING_APP_PORT = "5001"
$env:PARKING_DEBUG = "0"

python app.py
