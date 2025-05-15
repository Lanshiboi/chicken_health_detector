import sys
import os
import base64
import tempfile
import cv2
import numpy as np
from PyQt5 import QtCore, QtWidgets, QtWebChannel
from PyQt5.QtWebEngineWidgets import QWebEngineView
from tensorflow.keras.models import load_model
from ultralytics import YOLO

# Constants
IMG_SIZE = 224
TEMP_MIN = 30.0
TEMP_MAX = 45.0
THRESHOLD = 0.7

# Load models once
model = load_model("hybrid_mobilenetv2_temperature_model.h5", compile=False)
yolo_model = YOLO("yolov8n.pt")

from thermal_utils import extract_temperature_flir
import tempfile
import os

def estimate_temperature(img_crop):
    """
    Estimate temperature from a thermal image crop using flirpy extraction.
    """
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
        temp_path = tmp_file.name
        cv2.imwrite(temp_path, img_crop)

    temp = extract_temperature_flir(temp_path)

    try:
        os.remove(temp_path)
    except Exception:
        pass

    return temp

def detect_and_classify(image):
    """
    Detect chickens in the image using YOLO and classify each detected region using the hybrid model.
    Draw bounding boxes and labels on the output image.

    Args:
        image (numpy.ndarray): Input image.

    Returns:
        output (numpy.ndarray): Annotated image with detection and classification results.
        detection_count (int): Number of detected chickens.
        labels (list of str): List of labels for each detected chicken ("Healthy" or "Infected").
        temperatures (list of float): List of estimated temperatures for each detected chicken.
    """
    results = yolo_model(image)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    output = image.copy()
    labels = []
    temperatures = []

    if len(boxes) == 0:
        # No detections, treat whole image as single chicken
        temp = estimate_temperature(image)
        temp_norm = (temp - TEMP_MIN) / (TEMP_MAX - TEMP_MIN)
        resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE)) / 255.0
        resized = np.expand_dims(resized, axis=0)
        temp_input = np.array([[temp_norm]])

        pred = model.predict({"image_input": resized, "temp_input": temp_input})[0][0]
        # Override label to Healthy if temperature is in healthy range regardless of prediction
        if 39.0 <= temp <= 43.0:
            label = "Healthy"
        else:
            label = "Infected" if pred >= THRESHOLD else "Healthy"
        color = (0, 0, 255) if label == "Infected" else (0, 255, 0)

        # Draw results
        cv2.rectangle(output, (0, 0), (output.shape[1], output.shape[0]), color, 3)
        text = f"{label} ({temp:.1f}°C)"
        font_scale = 0.8
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        # Semi-transparent rectangle
        overlay = output.copy()
        alpha = 0.5
        cv2.rectangle(overlay, (0, 0), (text_width + 10, text_height + baseline + 10), color, -1)
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
        cv2.putText(output, text, (5, text_height + 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

        labels.append(label)
        temperatures.append(temp)
        detection_count = 1
        return output, detection_count, labels, temperatures

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        crop = image[y1:y2, x1:x2]

        temp = estimate_temperature(crop)
        temp_norm = (temp - TEMP_MIN) / (TEMP_MAX - TEMP_MIN)
        resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE)) / 255.0
        resized = np.expand_dims(resized, axis=0)
        temp_input = np.array([[temp_norm]])

        pred = model.predict({"image_input": resized, "temp_input": temp_input})[0][0]
        label = "Infected" if pred >= THRESHOLD else "Healthy"
        color = (0, 0, 255) if label == "Infected" else (0, 255, 0)

        # Draw results
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 3)
        text = f"{label} ({temp:.1f}°C)"
        font_scale = 0.8
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        # Semi-transparent rectangle
        overlay = output.copy()
        alpha = 0.5
        cv2.rectangle(overlay, (x1, y1 - text_height - baseline - 10), (x1 + text_width + 10, y1), color, -1)
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
        cv2.putText(output, text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

        labels.append(label)
        temperatures.append(temp)

    detection_count = len(boxes)
    return output, detection_count, labels, temperatures

class BackendBridge(QtCore.QObject):
    @QtCore.pyqtSlot(str, result='QVariant')
    def analyze_image(self, image_data_url):
        try:
            # Decode base64 image
            header, encoded = image_data_url.split(',', 1)
            image_bytes = base64.b64decode(encoded)

            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                temp_file.write(image_bytes)
                temp_path = temp_file.name

            # Load image using OpenCV
            img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Image could not be decoded.")

            # Use detect_and_classify function for detection and classification
            results = yolo_model(img)
            boxes = results[0].boxes
            classes = results[0].boxes.cls.cpu().numpy() if boxes is not None else []
            scores = results[0].boxes.conf.cpu().numpy() if boxes is not None else []

            # Check if any detected object is chicken class with confidence above threshold
            chicken_class_id = 0  # Assuming class 0 is chicken in YOLO model
            confidence_threshold = 0.5
            chicken_detections = [(cls, conf) for cls, conf in zip(classes, scores)
                                  if int(cls) == chicken_class_id and conf >= confidence_threshold]

            if len(chicken_detections) == 0:
                return {'error': 'No chickens detected with sufficient confidence. Please upload proper thermal chicken images.'}

            # Proceed with classification as before
            output_img, count, labels, temps = detect_and_classify(img)

            # Compute average temperature
            avg_temp = float(np.mean(temps)) if temps else None

            # Determine overall result label
            if not labels:
                overall_result = "Unknown"
            elif "Infected" in labels:
                overall_result = "Infected"
            elif "Monitoring" in labels:
                overall_result = "Monitoring"
            else:
                overall_result = "Healthy"

            # Encode output image to base64 to send back to frontend
            _, buffer = cv2.imencode('.png', output_img)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            img_data_url = f"data:image/png;base64,{img_base64}"

            return {
                'result': overall_result,
                'count': count,
                'average_temperature': avg_temp,
                'image': img_data_url
            }

        except Exception as e:
            return {'error': f"Error analyzing image: {str(e)}"}

class AnalyzeApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('BCHD - Analyze - Poultry Disease Detector')
        self.setGeometry(100, 100, 1200, 800)

        # Load the frontend (HTML)
        self.browser = QWebEngineView()
        html_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'analyze.html'))

        if os.path.exists(html_path):
            # Fix for Windows path: convert to QUrl with forward slashes
            url = QtCore.QUrl.fromLocalFile(html_path.replace("\\", "/"))
            self.browser.setUrl(url)
        else:
            QtWidgets.QMessageBox.critical(self, "Error", "analyze.html file not found!")

        self.setCentralWidget(self.browser)

        # Connect backend to JS via WebChannel
        self.channel = QtWebChannel.QWebChannel()
        self.backend = BackendBridge()
        self.channel.registerObject('backend', self.backend)
        self.browser.page().setWebChannel(self.channel)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = AnalyzeApp()
    window.show()
    sys.exit(app.exec_())
