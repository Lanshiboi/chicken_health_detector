import cv2
import numpy as np
from tensorflow.keras.models import load_model
from ultralytics import YOLO

import numpy as np
import cv2
from tensorflow.keras.models import load_model
from ultralytics import YOLO
from thermal_utils import extract_temperature_flir

# Constants
IMG_SIZE = 224
TEMP_MIN = 30.0
TEMP_MAX = 45.0
THRESHOLD = 0.7

# Load models
yolo_model = YOLO("yolov8n.pt")  # Use default pretrained YOLOv8n weights
hybrid_model = load_model("hybrid_mobilenetv2_temperature_model.h5")

def estimate_temperature(img_crop):
    """
    Estimate temperature from a thermal image crop using flirpy extraction.
    """
    # Save crop temporarily to disk to use extract_temperature_flir
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
        temp_path = tmp_file.name
        cv2.imwrite(temp_path, img_crop)

    temp = extract_temperature_flir(temp_path)

    try:
        os.remove(temp_path)
    except Exception:
        pass

    return temp

def detect_and_classify(img_path):
    """
    Detect chickens in the image using YOLO and classify each detected region using the hybrid model.
    Draw bounding boxes and labels on the output image.

    Args:
        img_path (str): Path to the input image.

    Returns:
        output (numpy.ndarray): Annotated image with detection and classification results.
        detection_count (int): Number of detected chickens.
        labels (list of str): List of labels for each detected chicken ("Healthy" or "Infected").
        temperatures (list of float): List of estimated temperatures for each detected chicken.
    """
    image = cv2.imread(img_path)
    if image is None:
        raise ValueError(f"Image at path {img_path} could not be loaded.")

    results = yolo_model(image)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    output = image.copy()
    labels = []
    temperatures = []

    if len(boxes) == 0:
        # No detections, treat whole image as single chicken
        temp = extract_temperature_flir(img_path)
        temp_norm = (temp - TEMP_MIN) / (TEMP_MAX - TEMP_MIN)
        resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE)) / 255.0
        resized = np.expand_dims(resized, axis=0)
        temp_input = np.array([[temp_norm]])

        pred = hybrid_model.predict({"image_input": resized, "temp_input": temp_input})[0][0]
        # Apply temperature range based labeling similar to MobileNetV2
        if 38.0 <= temp <= 40.5:
            label = "Healthy"
        elif 40.5 < temp <= 42.0:
            label = "Possible Fever / Early Illness"
        elif 42.0 <= temp < 43.0:
            label = "Possible Bird Flu (High Risk)"
        elif temp >= 43.0:
            label = "Infected"
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
        # Clamp box coordinates to image dimensions
        x1, y1, x2, y2 = map(int, box)
        x1 = max(0, min(x1, image.shape[1] - 1))
        x2 = max(0, min(x2, image.shape[1] - 1))
        y1 = max(0, min(y1, image.shape[0] - 1))
        y2 = max(0, min(y2, image.shape[0] - 1))

        # Skip invalid boxes
        if x2 <= x1 or y2 <= y1:
            print(f"Skipping invalid box with coordinates: {(x1, y1, x2, y2)}")
            continue

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            print(f"Skipping empty crop for box: {(x1, y1, x2, y2)}")
            continue

        temp = estimate_temperature(crop)
        temp_norm = (temp - TEMP_MIN) / (TEMP_MAX - TEMP_MIN)
        resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE)) / 255.0
        resized = np.expand_dims(resized, axis=0)
        temp_input = np.array([[temp_norm]])

        pred = hybrid_model.predict({"image_input": resized, "temp_input": temp_input})[0]
        pred_class = np.argmax(pred)
        label_map = {
            0: "Healthy",
            1: "Possible Fever / Early Illness",
            2: "Possible Bird Flu (High Risk)",
            3: "Infected"
        }
        label = label_map.get(pred_class, "Infected")

        # Override label based on temperature ranges
        if 38.0 <= temp <= 40.5:
            label = "Healthy"
        elif 40.5 < temp <= 42.0:
            label = "Possible Fever / Early Illness"
        elif 42.0 <= temp < 43.0:
            label = "Possible Bird Flu (High Risk)"
        elif temp >= 43.0:
            label = "Infected"
        else:
            # Use prediction confidence threshold to decide label
            if pred[pred_class] >= THRESHOLD:
                label = label_map.get(pred_class, "Infected")
            else:
                label = "Healthy"

        print(f"Box: {(x1, y1, x2, y2)}, Label: {label}, Temp: {temp:.2f}")

        color = (0, 0, 255) if label == "Infected" else (0, 255, 0)

        # Draw results
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 3)
        text = f"{label} ({temp:.1f}°C)"
        font_scale = 0.8
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        # Semi-transparent rectangle
        overlay = output.copy()
        cv2.rectangle(overlay, (x1, y1 - text_height - baseline - 10), (x1 + text_width + 10, y1), color, -1)
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
        cv2.putText(output, text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

        labels.append(label)
        temperatures.append(temp)

    detection_count = len(boxes)
    return output, detection_count, labels, temperatures
