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
THRESHOLD = 0.5

from thermal_utils import extract_temperature_flir

# Load models
model = load_model("hybrid_mobilenetv2_temperature_model.h5")
yolo_model = YOLO("yolov8n.pt")

def estimate_temperature(img_crop):
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
        temp_path = tmp_file.name
        cv2.imwrite(temp_path, img_crop)

    temp = extract_temperature_flir(temp_path)

    try:
        os.remove(temp_path)
    except Exception:
        pass

    return max(temp, TEMP_MIN)

def get_robust_max_temperature(img_crop):
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
        temp_path = tmp_file.name
        cv2.imwrite(temp_path, img_crop)

    img_gray = cv2.imread(temp_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        try:
            os.remove(temp_path)
        except Exception:
            pass
        return TEMP_MIN

    norm = img_gray.astype(np.float32) / 255.0
    temp_map = norm * (TEMP_MAX - TEMP_MIN) + TEMP_MIN

    try:
        os.remove(temp_path)
    except Exception:
        pass

    robust_max_temp = np.percentile(temp_map, 95)
    return max(robust_max_temp, TEMP_MIN)

def detect_and_classify(image):
    results = yolo_model(image)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    output = image.copy()
    labels = []
    temperatures = []

    if len(boxes) == 0:
        temp = estimate_temperature(image)
        temp_norm = (temp - TEMP_MIN) / (TEMP_MAX - TEMP_MIN)
        resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE)) / 255.0
        resized = np.expand_dims(resized, axis=0)
        temp_input = np.array([[temp_norm]])

        pred = model.predict({"image_input": resized, "temp_input": temp_input})[0]

        if 35.0 <= temp <= 41.0:
            label = "Healthy"
        elif 41.0 < temp <= 42.0:
            label = "Possible Fever / Early Illness"
        elif 42.0 <= temp < 43.0:
            label = "Possible Bird Flu (High Risk)"
        elif temp >= 43.0:
            label = "Infected"
        else:
            label = "Infected" if pred[3] >= THRESHOLD else "Healthy"

        color = (0, 0, 255) if "Infected" in label else (0, 255, 0)
        cv2.rectangle(output, (0, 0), (output.shape[1], output.shape[0]), color, 3)
        text = f"{label} ({temp:.1f}°C)"
        overlay = output.copy()
        font_scale, thickness = 0.8, 2
        (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(overlay, (0, 0), (tw + 10, th + bl + 10), color, -1)
        cv2.addWeighted(overlay, 0.5, output, 0.5, 0, output)
        cv2.putText(output, text, (5, th + 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

        labels.append(label)
        temperatures.append(temp)
        return output, 1, labels, temperatures

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        x1 = max(0, min(x1, image.shape[1] - 1))
        x2 = max(0, min(x2, image.shape[1] - 1))
        y1 = max(0, min(y1, image.shape[0] - 1))
        y2 = max(0, min(y2, image.shape[0] - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        max_temp = get_robust_max_temperature(crop)
        temp_norm = (max_temp - TEMP_MIN) / (TEMP_MAX - TEMP_MIN)
        resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE)) / 255.0
        resized = np.expand_dims(resized, axis=0)
        temp_input = np.array([[temp_norm]])

        pred = model.predict({"image_input": resized, "temp_input": temp_input})[0]

        pred_label_index = np.argmax(pred)
        pred_confidence = pred[pred_label_index]

        if 35.0 <= max_temp <= 40.5:
            label = "Healthy"
        elif 40.5 < max_temp <= 42.0:
            label = "Possible Fever / Early Illness"
        elif 42.0 <= max_temp < 43.0:
            label = "Possible Bird Flu (High Risk)"
        elif max_temp >= 43.0:
            label = "Infected"
        else:
            if pred_confidence >= THRESHOLD:
                label = ["Healthy", "Possible Fever / Early Illness", "Possible Bird Flu (High Risk)", "Infected"][pred_label_index]
            else:
                label = "Healthy"

        color = (0, 0, 255) if label == "Infected" else (0, 255, 0)

        cv2.rectangle(output, (x1, y1), (x2, y2), color, 3)
        text = f"{label} ({max_temp:.1f}°C)"
        overlay = output.copy()
        font_scale, thickness = 0.8, 2
        (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(overlay, (x1, y1 - th - bl - 10), (x1 + tw + 10, y1), color, -1)
        cv2.addWeighted(overlay, 0.5, output, 0.5, 0, output)
        cv2.putText(output, text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

        labels.append(label)
        temperatures.append(max_temp)

    valid_temps = [t for t in temperatures if t >= TEMP_MIN]
    return output, len(labels), labels, valid_temps

class BackendBridge(QtCore.QObject):
    @QtCore.pyqtSlot(str, str, result='QVariant')
    def login(self, email, password):
        valid_users = {
            "user@example.com": "Password123",
            "admin@example.com": "AdminPass123"
        }
        if email in valid_users and valid_users[email] == password:
            return {"success": True, "message": "Login successful"}
        else:
            return {"success": False, "message": "Invalid email or password"}

    @QtCore.pyqtSlot(str, result='QVariant')
    def analyze_image(self, image_data_url):
        try:
            _, encoded = image_data_url.split(',', 1)
            image_bytes = base64.b64decode(encoded)
            img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Image could not be decoded.")

            output_img, count, labels, temps = detect_and_classify(img)
            max_temp = float(np.max(temps)) if temps else None

            if not labels:
                overall_result = "Unknown"
            elif "Infected" in labels:
                overall_result = "Infected"
            elif "Possible Bird Flu (High Risk)" in labels or "Possible Fever / Early Illness" in labels:
                overall_result = "At Risk"
            elif "Monitoring" in labels:
                overall_result = "Monitoring"
            else:
                overall_result = "Healthy"

            _, buffer = cv2.imencode('.png', output_img)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            img_data_url = f"data:image/png;base64,{img_base64}"

            return {
                'result': overall_result,
                'count': count,
                'average_temperature': max_temp,
                'image': img_data_url
            }

        except Exception as e:
            return {'error': f"Error analyzing image: {str(e)}"}

    @QtCore.pyqtSlot(result='QVariant')
    def get_dashboard_data(self):
        healthy_count = 10
        possible_fever_count = 3
        possible_bird_flu_count = 2
        infected_count = 3
        at_risk_count = possible_fever_count + possible_bird_flu_count

        return {
            'statistics': {
                'healthy': healthy_count,
                'atRisk': at_risk_count,
                'infected': infected_count,
                'total': healthy_count + at_risk_count + infected_count
            },
            'recentAlerts': [
                {'date': '2024-06-01', 'chickenId': 'CHK_001', 'status': 'Infected', 'id': '1'},
                {'date': '2024-06-02', 'chickenId': 'CHK_002', 'status': 'At Risk', 'id': '2'},
                {'date': '2024-06-03', 'chickenId': 'CHK_003', 'status': 'Infected', 'id': '3'}
            ],
            'healthOverview': {
                'labels': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                'healthy': [2, 3, 2, 4, 3, 2, 1],
                'infected': [0, 1, 0, 1, 0, 1, 0]
            },
            'distribution': {
                'healthy': healthy_count,
                'atRisk': at_risk_count,
                'infected': infected_count
            }
        }

    @QtCore.pyqtSlot(str, result='QVariant')
    def show_details(self, alert_id):
        details = {
            '1': {'info': 'Details for alert 1'},
            '2': {'info': 'Details for alert 2'},
            '3': {'info': 'Details for alert 3'}
        }
        return details.get(alert_id, {'info': 'No details found'})

    @QtCore.pyqtSlot(result='QVariant')
    def get_reports_data(self):
        return [
            {'id': 'R001', 'date': '2024-06-01', 'summary': 'Report 1 summary', 'status': 'Completed'},
            {'id': 'R002', 'date': '2024-06-05', 'summary': 'Report 2 summary', 'status': 'Pending'},
            {'id': 'R003', 'date': '2024-06-10', 'summary': 'Report 3 summary', 'status': 'Completed'}
        ]

    @QtCore.pyqtSlot(result='QVariant')
    def get_calendar_events(self):
        return [
            {'id': 'E001', 'title': 'Vaccination', 'date': '2024-06-15', 'description': 'Vaccination for flock A'},
            {'id': 'E002', 'title': 'Inspection', 'date': '2024-06-20', 'description': 'Health inspection for barn 3'},
            {'id': 'E003', 'title': 'Feed Delivery', 'date': '2024-06-25', 'description': 'Delivery of feed supplies'}
        ]

class MainApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('BCHD - Poultry Disease Monitoring')
        self.setGeometry(100, 100, 1200, 800)

        self.browser = QWebEngineView()
        self.current_page = 'dashboard'  # Start with dashboard.html

        # Clear cache on startup
        self.browser.page().profile().clearHttpCache()

        self.load_page(self.current_page)
        self.setCentralWidget(self.browser)

        self.channel = QtWebChannel.QWebChannel()
        self.backend = BackendBridge()
        self.channel.registerObject('backend', self.backend)
        self.browser.page().setWebChannel(self.channel)

    def load_page(self, page_name):
        html_file = f"{page_name}.html"
        html_path = os.path.abspath(os.path.join(os.path.dirname(__file__), html_file))
        if os.path.exists(html_path):
            # Clear cache before loading page
            self.browser.page().profile().clearHttpCache()
            self.browser.setUrl(QtCore.QUrl.fromLocalFile(html_path))
            self.current_page = page_name
        else:
            QtWidgets.QMessageBox.critical(self, "Error", f"{html_file} file not found!")

    @QtCore.pyqtSlot()
    def load_dashboard(self):
        self.load_page('dashboard')

    @QtCore.pyqtSlot()
    def load_auth(self):
        self.load_page('auth')

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())