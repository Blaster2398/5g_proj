from flask import Flask, render_template, Response
from image_processor import process_static_image

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed') # Keeping the same route name so index.html doesn't break
def video_feed():
    image_bytes = process_static_image()
    if image_bytes:
        # Serve it as a single static JPEG rather than a multipart stream
        return Response(image_bytes, mimetype='image/jpeg')
    else:
        return "Image processing failed. Check data/sample_parking.jpg", 404

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)