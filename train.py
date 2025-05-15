import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input, Concatenate
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# === Constants ===
IMG_SIZE = 224
BATCH_SIZE = 32
TEMP_MIN = 30.0
TEMP_MAX = 45.0
BASE_DIR = 'thermal_dataset'

from thermal_utils import extract_temperature_flir

# === Data augmentation generator for Infected class and training images ===
datagen_infected = ImageDataGenerator(
    rotation_range=20,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)

datagen_train = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)

# === Load dataset with filtering for healthy, infected, and possible infected temperature ranges ===
def load_data_with_temp(base_dir):
    images, temps, labels = [], [], []
    classes = ['Healthy', 'Possible_Fever', 'Possible_Bird_Flu', 'Infected']

    for label_name in classes:
        if label_name == 'Healthy':
            label = 0
        elif label_name == 'Possible_Fever':
            label = 1
        elif label_name == 'Possible_Bird_Flu':
            label = 2
        else:
            label = 3

        folder = os.path.join(base_dir, label_name)

        for fname in os.listdir(folder):
            if not fname.lower().endswith(('.jpg', '.png', '.bmp', '.jpeg')):
                continue

            path = os.path.join(folder, fname)
            img = cv2.imread(path)
            if img is None:
                continue

            # Convert image to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            temp = extract_temperature_flir(path)
            if temp is None:
                continue

            # Updated temperature filtering based on new classes
            if label == 0:  # Healthy
                if temp < 35.0 or temp > 41.0:
                    continue
            elif label == 1:  # Possible_Fever
                if temp <= 41.0 or temp > 42.5:
                    continue
            elif label == 2:  # Possible_Bird_Flu
                if temp <= 42.5 or temp >= 43.5:
                    continue
            elif label == 3:  # Infected
                if temp < 43.5:
                    continue

            img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0

            # Augment Infected class images
            if label == 3:
                img_resized = datagen_infected.random_transform(img_resized)

            images.append(img_resized)
            temps.append([temp])
            labels.append(label)

    return np.array(images, dtype=np.float32), np.array(temps, dtype=np.float32), np.array(labels, dtype=np.int32)

# === Load and split data with stratify to maintain class distribution ===
print("Loading dataset with temperature filtering and augmentation for Infected class...")
images, temps, labels = load_data_with_temp(BASE_DIR)
print(f" Dataset loaded: {len(images)} samples")

# Print counts for each category with proper label names
label_names = ['Healthy', 'Possible Fever', 'Possible Bird Flu', 'Infected']
unique, counts = np.unique(labels, return_counts=True)
for u, c in zip(unique, counts):
    print(f"{label_names[u]} samples: {c}")

# Merge "Possible_Bird_Flu" class (label 2) into "Possible_Fever" class (label 1) to avoid stratify error
labels = np.where(labels == 2, 1, labels)

# Split data into training and validation sets with stratify
X_img_train, X_img_val, X_temp_train, X_temp_val, y_train, y_val = train_test_split(
    images, temps, labels, test_size=0.2, random_state=42, stratify=labels
)

# Apply data augmentation to training images
augmented_images = []
augmented_temps = []
augmented_labels = []

for img, temp, label in zip(X_img_train, X_temp_train, y_train):
    img_aug = datagen_train.random_transform(img)
    augmented_images.append(img_aug)
    augmented_temps.append(temp)
    augmented_labels.append(label)

# Combine original and augmented training data
X_img_train = np.concatenate((X_img_train, np.array(augmented_images)), axis=0)
X_temp_train = np.concatenate((X_temp_train, np.array(augmented_temps)), axis=0)
y_train = np.concatenate((y_train, np.array(augmented_labels)), axis=0)

# Compute class weights to address imbalance
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))
print("Class weights:", class_weight_dict)

# === Define CNN + Temp hybrid model ===
img_input = Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image_input")
temp_input = Input(shape=(1,), name="temp_input")

# MobileNetV2 Base Model
base_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze initially (for transfer learning)

x = base_model(img_input, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)

# Merge image + temperature features
combined = Concatenate()([x, temp_input])
combined = Dense(128, activation='relu')(combined)
combined = Dropout(0.3)(combined)
output = Dense(4, activation='softmax')(combined)

# Build Model
model = Model(inputs=[img_input, temp_input], outputs=output)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# === Train model with class weights ===
print("Starting model training with class weights...")
history = model.fit(
    {"image_input": X_img_train, "temp_input": X_temp_train},
    y_train,
    validation_data=({"image_input": X_img_val, "temp_input": X_temp_val}, y_val),
    epochs=10,
    batch_size=BATCH_SIZE,
    class_weight=class_weight_dict
)

# === Save model ===
model.save('hybrid_mobilenetv2_temperature_model.h5')
print("Model training complete and saved as 'hybrid_mobilenetv2_temperature_model.h5'")

# === Plot training history ===
print("Plotting training history...")
history_df = pd.DataFrame(history.history)

# Loss Plot
plt.figure(figsize=(8, 6))
history_df[['loss', 'val_loss']].plot()
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

# Accuracy Plot
plt.figure(figsize=(8, 6))
history_df[['accuracy', 'val_accuracy']].plot()
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()

# === Generate confusion matrix on validation set with formatted output ===
print("Generating confusion matrix on validation set...")
y_val_pred_probs = model.predict({"image_input": X_img_val, "temp_input": X_temp_val})
y_val_pred = y_val_pred_probs.argmax(axis=1)

cm = confusion_matrix(y_val, y_val_pred, labels=[0,1,2,3])
cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)

print("Confusion Matrix (Validation Set):")
print(cm_df)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix on Validation Set')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# === Generate confusion matrix on training set with formatted output ===
print("Generating confusion matrix on training set...")
y_train_pred_probs = model.predict({"image_input": X_img_train, "temp_input": X_temp_train})
y_train_pred = y_train_pred_probs.argmax(axis=1)

cm_train = confusion_matrix(y_train, y_train_pred, labels=[0,1,2,3])
cm_train_df = pd.DataFrame(cm_train, index=label_names, columns=label_names)

print("Confusion Matrix (Training Set):")
print(cm_train_df)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_train_df, annot=True, fmt='d', cmap='Greens')
plt.title('Confusion Matrix on Training Set')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
