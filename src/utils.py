import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



def plot_confusion_matrix(true_labels, predictions, labels=None):
    """
    Plot a confusion matrix using Seaborn heatmap.

    Args:
        true_labels (list or np.array): The true labels of the dataset.
        predictions (list or np.array): The predicted labels of the dataset.
        labels (list, optional): The list of label names (default is None).
    """
    cm = confusion_matrix(true_labels, predictions, labels=labels)
    
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('True labels')
    plt.ylabel('Predicted labels')
    plt.title('Confusion Matrix')
    plt.show()



def compute_metrics(true_labels, predictions):
    """
    Compute common classification metrics: accuracy, precision, recall, and F1 score.

    Args:
        true_labels (list or np.array): The true labels of the dataset.
        predictions (list or np.array): The predicted labels of the dataset.

    Returns:
        dict: A dictionary containing accuracy, precision, recall, and F1 score.
    """

    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='binary')  # Adjust for multi-class if needed
    recall = recall_score(true_labels, predictions, average='binary')  # Adjust for multi-class if needed
    f1 = f1_score(true_labels, predictions, average='binary')  # Adjust for multi-class if needed
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def training_curves(results):
    """
    Plot training and validation curves for loss and accuracy.

    Args:
        results (dict): Dictionary containing 'train_loss', 'train_acc', 
                         'val_loss', 'val_acc' lists from the trainer.
    """
    epochs = range(1, len(results['train_loss']) + 1)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1) 
    plt.plot(epochs, results['train_loss'], label='Train Loss', color='blue')
    plt.plot(epochs, results['val_loss'], label='Validation Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, results['train_acc'], label='Train Accuracy', color='blue')
    plt.plot(epochs, results['val_acc'], label='Validation Accuracy', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Train and Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

