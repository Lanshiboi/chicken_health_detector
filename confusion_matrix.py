import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_confusion_matrix(y_true, y_pred, labels=None, normalize=None, title='Confusion Matrix'):
    """
    Plots a confusion matrix using sklearn's ConfusionMatrixDisplay.

    Parameters:
    - y_true: list or array of true labels
    - y_pred: list or array of predicted labels
    - labels: list of label names (optional)
    - normalize: {'true', 'pred', 'all', None}, normalization mode
    - title: title of the plot
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    # Example usage with dummy data
    y_true = ['Healthy', 'Infected', 'Possible Fever', 'Healthy', 'Possible Bird Flu', 'Infected', 'Healthy']
    y_pred = ['Healthy', 'Infected', 'Healthy', 'Healthy', 'Possible Bird Flu', 'Infected', 'Possible Fever']
    labels = ['Healthy', 'Possible Fever', 'Possible Bird Flu', 'Infected']

    plot_confusion_matrix(y_true, y_pred, labels=labels, normalize='true', title='Normalized Confusion Matrix')
