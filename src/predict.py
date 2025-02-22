import torch
import numpy as np
from .models import bert_classifier 
from .utils import plot_confusion_matrix, compute_metrics 


def make_predictions(model, test_loader, device):
    """
    Make predictions using a trained model.
    Metrics evaluated: accuracy, precision, recall, and F1 score.
    Confusion matrix plot.
    """
    model.eval()  

    predictions = []
    true_labels = []

    with torch.no_grad():  
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['label'].to(device) 

            outputs = model(input_ids, attention_mask, token_type_ids)
            _, predicted = torch.max(outputs, 1)
            
            predictions.extend(predicted.cpu().numpy()) 
            true_labels.extend(labels.cpu().numpy()) 

    predictions = np.array(predictions)
    true_labels = np.array(true_labels)

    metrics = compute_metrics(true_labels, predictions)
    for metric, value in metrics.items():
        print(f'{metric.capitalize()}: {value:.4f}')

    plot_confusion_matrix(true_labels, predictions)
