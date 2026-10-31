
import cv2
import numpy as np
import onnxruntime as ort
import time

from flask import Flask, Response
from picamera2 import Picamera2

app = Flask(__name__)

# Model settings
MODEL_PATH = "leaf_detector.onnx"
CONFIDENCE = 0.40
INPUT_SIZE = 320

# Camera settings
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Load ONNX model
print("Loading AI model...")

options = ort.SessionOptions()
options.intra_op_num_threads = 2
options.inter_op_num_threads = 1
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

session = ort.InferenceSession(
    MODEL_PATH,
    sess_options=options,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

print("AI model loaded successfully")

# Start Raspberry Pi camera
print("Starting camera...")

camera = Picamera2()

config = camera.create_video_configuration(
    main={
        "size": (CAMERA_WIDTH, CAMERA_HEIGHT),
        "format": "RGB888"
    },
    buffer_count=4
)

camera.configure(config)
camera.start()

# Automatic exposure and white balance
camera.set_controls({
    "AeEnable": True,
    "AwbEnable": True
})

time.sleep(3)

print("Camera started successfully")


def generate_frames():

    previous_time = time.time()

    while True:

        # Capture camera image
        frame = camera.capture_array()

        height, width = frame.shape[:2]

        # Picamera2 RGB888 data is BGR-ordered
        # Convert BGR to RGB for AI
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Resize image
        resized = cv2.resize(
            rgb_frame,
            (INPUT_SIZE, INPUT_SIZE),
            interpolation=cv2.INTER_LINEAR
        )

        # Prepare AI input
        image = resized.astype(np.float32) / 255.0

        image = np.transpose(
            image,
            (2, 0, 1)
        )

        image = np.expand_dims(
            image,
            axis=0
        )

        # Run AI detection
        boxes, scores, labels = session.run(
            None,
            {input_name: image}
        )

        leaf_count = 0

        scale_x = width / INPUT_SIZE
        scale_y = height / INPUT_SIZE

        # Process detections
        for box, score, label in zip(
            boxes, scores, labels
        ):

            score = float(score)
            label = int(label)

            if score < CONFIDENCE:
                continue

            if label != 1:
                continue

            x1, y1, x2, y2 = box

            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)

            # Keep boxes inside the image
            x1 = max(0, min(width - 1, x1))
            y1 = max(0, min(height - 1, y1))
            x2 = max(0, min(width - 1, x2))
            y2 = max(0, min(height - 1, y2))

            if x2 <= x1 or y2 <= y1:
                continue

            leaf_count += 1

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Show confidence
            cv2.putText(
                frame,
                f"Leaf {score:.2f}",
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        # Calculate FPS
        current_time = time.time()

        fps = 1.0 / max(
            current_time - previous_time,
            0.001
        )

        previous_time = current_time

        # Show detection information
        cv2.putText(
            frame,
            f"Leaves: {leaf_count}  FPS: {fps:.1f}",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        # Convert BGR frame to RGB for browser JPEG
        display = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        success, buffer = cv2.imencode(
            ".jpg",
            display,
            [cv2.IMWRITE_JPEG_QUALITY, 85]
        )

        if not success:
            continue

        jpeg = buffer.tobytes()

        # Send video frame
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + jpeg +
            b"\r\n"
        )


@app.route("/")
def home():

    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>PlantCare AI</title>

        <meta name="viewport"
              content="width=device-width, initial-scale=1">

        <style>
            body {
                background: #202020;
                color: white;
                text-align: center;
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
            }

            h1 {
                color: #35d05b;
            }

            img {
                width: 100%;
                max-width: 900px;
                border: 3px solid #35d05b;
                border-radius: 10px;
            }

            p {
                color: #cccccc;
            }
        </style>
    </head>

    <body>
        <h1>PlantCare AI</h1>
        <h3>Live Leaf Detection</h3>

        <img src="/video_feed">

        <p>Raspberry Pi 5 - ONNX Runtime</p>
    </body>
    </html>
    """


@app.route("/video_feed")
def video_feed():

    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":

    print("--------------------------------")
    print("PlantCare AI Camera Server")
    print("http://172.20.10.2:5000")
    print("--------------------------------")

    app.run(
        host="0.0.0.0",
        port=5000,
        threaded=True,
        debug=False
    )
