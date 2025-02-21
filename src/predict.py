import torch
import numpy as np
from models import bert_classifier 
from utils import plot_confusion_matrix, compute_metrics 


def make_predictions(model_path, vocab_path, test_loader):
    """
    Make predictions using a trained model.
    Metrics evaluated: accuracy, precision, recall, and F1 score.
    Confusion matrix plot.
    """
    model = bert_classifier() 
    model.load_state_dict(torch.load(model_path)) 
    model.eval()  

    predictions = []
    true_labels = []

    with torch.no_grad():  
        for inputs, labels in test_loader:
            inputs = inputs.to(model.device)  
            labels = labels.to(model.device)  

            outputs = model(inputs) 
            _, predicted = torch.max(outputs, 1) 
            
            predictions.extend(predicted.cpu().numpy()) 
            true_labels.extend(labels.cpu().numpy()) 

    predictions = np.array(predictions)
    true_labels = np.array(true_labels)

    metrics = compute_metrics(true_labels, predictions)
    for metric, value in metrics.items():
        print(f'{metric.capitalize()}: {value:.4f}')

    plot_confusion_matrix(true_labels, predictions)
