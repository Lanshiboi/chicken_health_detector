import argparse
from ultralytics import YOLO

# Constants for training
YOLO_CONFIG = 'chicken_yolo.yaml'
YOLO_MODEL_NAME = 'yolov8n.pt'
YOLO_EPOCHS = 5
IMG_SIZE = 640

if __name__ == "__main__":
    # Set up argument parsing for customization
    parser = argparse.ArgumentParser(description="Train YOLO model for chicken health detection")
    parser.add_argument('--epochs', type=int, default=YOLO_EPOCHS, help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=IMG_SIZE, help='Image size for training')
    parser.add_argument('--weights', type=str, default=YOLO_MODEL_NAME, help='Pretrained weights to start training from')
    parser.add_argument('--data', type=str, default=YOLO_CONFIG, help='Dataset config YAML file')

    args = parser.parse_args()

    # Initialize the YOLO model with the specified weights
    model = YOLO(args.weights)

    # Print details of the training process
    print(f"Training YOLO model with data={args.data}, epochs={args.epochs}, imgsz={args.imgsz}")

    # Start training the model
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz)

    # Notify that training is finished
    print("YOLO training finished.")

    # Run validation to get metrics including confusion matrix
    print("Running validation to generate confusion matrix...")
    val_results = model.val(data=args.data, imgsz=args.imgsz)

    # Extract confusion matrix from validation results
    # val_results.confusion_matrix is a numpy array if available
    if hasattr(val_results, 'confusion_matrix'):
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay

        cm = val_results.confusion_matrix
        labels = val_results.names if hasattr(val_results, 'names') else None

        plt.figure(figsize=(8, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap=plt.cm.Blues)
        plt.title('YOLO Validation Confusion Matrix')
        plt.show()
    else:
        print("Confusion matrix not available in validation results.")
