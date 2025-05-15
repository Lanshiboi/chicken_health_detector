import os
import shutil
from thermal_utils import extract_temperature_flir

BASE_DIR = 'thermal_dataset'
# Adjusted to use the base directory directly for images
ALL_IMAGES_DIR = BASE_DIR

CLASS_FOLDERS = {
    'Healthy': os.path.join(BASE_DIR, 'Healthy'),
    'Possible_Fever': os.path.join(BASE_DIR, 'Possible_Fever'),
    'Possible_Bird_Flu': os.path.join(BASE_DIR, 'Possible_Bird_Flu'),
    'Infected': os.path.join(BASE_DIR, 'Infected'),
}

# Temperature thresholds for classification
def classify_temp(temp):
    # Adjusted temperature ranges to start Healthy from 35.0 as requested
    if 35.0 <= temp <= 41.0:
        return 'Healthy'
    elif 41.0 < temp <= 42.5:
        return 'Possible_Fever'
    elif 42.5 < temp < 43.5:
        return 'Possible_Bird_Flu'
    elif temp >= 43.5:
        return 'Infected'
    else:
        return None  # Temperature out of expected range

def ensure_dirs():
    for folder in CLASS_FOLDERS.values():
        if not os.path.exists(folder):
            os.makedirs(folder)

def rearrange_dataset():
    ensure_dirs()
    # Recursively find all images in base directory and subdirectories
    images = []
    for root, dirs, files in os.walk(ALL_IMAGES_DIR):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                images.append(os.path.join(root, file))

    for img_path in images:
        img_name = os.path.basename(img_path)
        temp = extract_temperature_flir(img_path)
        if temp is None:
            print(f"Warning: Could not extract temperature from {img_name}. Skipping.")
            continue

        class_name = classify_temp(temp)
        if class_name is None:
            print(f"Warning: Temperature {temp} for {img_name} out of expected range. Skipping.")
            continue

        dest_folder = CLASS_FOLDERS[class_name]
        dest_img_path = os.path.join(dest_folder, img_name)

        # Move the image to the appropriate class folder
        shutil.move(img_path, dest_img_path)

        # Also move the corresponding label file if it exists
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_src_path = os.path.join(os.path.dirname(img_path), label_name)
        label_dest_path = os.path.join(dest_folder, label_name)
        if os.path.exists(label_src_path):
            shutil.move(label_src_path, label_dest_path)

        print(f"Moved {img_name} and label (if exists) to {class_name} folder based on temperature {temp:.2f}")

if __name__ == "__main__":
    rearrange_dataset()
    print("Thermal dataset rearrangement based on temperature complete.")
