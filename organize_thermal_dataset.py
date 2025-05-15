import os
import random
import shutil

BASE_DIR = 'thermal_dataset'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VAL_DIR = os.path.join(BASE_DIR, 'val')

CLASS_FOLDERS = ['Healthy', 'Possible_Fever', 'Possible_Bird_Flu', 'Infected']

def ensure_dirs():
    for base in [TRAIN_DIR, VAL_DIR]:
        for cls in CLASS_FOLDERS:
            path = os.path.join(base, 'images', cls)
            os.makedirs(path, exist_ok=True)
            label_path = os.path.join(base, 'labels', cls)
            os.makedirs(label_path, exist_ok=True)

def split_dataset(train_ratio=0.8):
    ensure_dirs()
    for cls in CLASS_FOLDERS:
        class_folder = os.path.join(BASE_DIR, cls)
        images = [f for f in os.listdir(class_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        random.shuffle(images)
        train_count = int(len(images) * train_ratio)
        train_images = images[:train_count]
        val_images = images[train_count:]

        for img_name in train_images:
            src_img = os.path.join(class_folder, img_name)
            dst_img = os.path.join(TRAIN_DIR, 'images', cls, img_name)
            shutil.copy2(src_img, dst_img)

            label_name = os.path.splitext(img_name)[0] + '.txt'
            src_label = os.path.join(class_folder, label_name)
            if os.path.exists(src_label):
                dst_label = os.path.join(TRAIN_DIR, 'labels', cls, label_name)
                shutil.copy2(src_label, dst_label)

        for img_name in val_images:
            src_img = os.path.join(class_folder, img_name)
            dst_img = os.path.join(VAL_DIR, 'images', cls, img_name)
            shutil.copy2(src_img, dst_img)

            label_name = os.path.splitext(img_name)[0] + '.txt'
            src_label = os.path.join(class_folder, label_name)
            if os.path.exists(src_label):
                dst_label = os.path.join(VAL_DIR, 'labels', cls, label_name)
                shutil.copy2(src_label, dst_label)

if __name__ == "__main__":
    split_dataset()
    print("Dataset split into train and val completed.")
