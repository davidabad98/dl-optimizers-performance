import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def evaluate_model(model, data_loader, device, criterion):
    """
    Evaluate the model on a given dataset and compute the average loss and accuracy.

    Args:
        model (torch.nn.Module): The model to evaluate.
        data_loader (torch.utils.data.DataLoader): DataLoader for the evaluation dataset.
        device (torch.device): The device (e.g., 'cpu' or 'cuda') to run the evaluation on.
        criterion (torch.nn.Module): The loss function used to compute the loss.

    Returns:
        tuple: A tuple containing:
            - avg_loss (float): The average loss over the entire dataset.
            - accuracy (float): The accuracy of the model on the dataset.
    """
    model.eval()
    total_loss, correct, total_samples = 0, 0, 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            total_loss += loss.item()  # Accumulate loss for this batch

            _, predicted = torch.max(outputs, 1)  # Get predicted class
            correct += (predicted == labels).sum().item()  # Count correct predictions
            total_samples += labels.size(0)  # Keep track of total samples

    loss /= len(data_loader)  # Compute average loss per batch
    accuracy = correct / total_samples  # Compute accuracy

    return total_loss / len(data_loader), accuracy


class Evaluation:

    def plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()

    def plot_loss_accuracy(self, accuracy, avg_loss):
        metrics = ["Accuracy", "Average Loss"]
        values = [accuracy, avg_loss]

        plt.figure(figsize=(8, 5))
        plt.bar(metrics, values, color=["blue", "red"])
        plt.ylabel("Value")
        plt.title("Model Performance Metrics")
        plt.show()


"""For seperate plotting for the graphs"""

# def plot_accuracy(self, accuracy):
#     plt.figure(figsize=(6, 4))
#     plt.bar(['Accuracy'], [accuracy], color='blue')
#     plt.ylabel("Value")
#     plt.title("Model Accuracy")
#     plt.ylim(0, 1)
#     plt.show()

# def plot_loss(self, avg_loss):
#     plt.figure(figsize=(6, 4))
#     plt.bar(['Average Loss'], [avg_loss], color='red')
#     plt.ylabel("Value")
#     plt.title("Model Loss")
#     plt.show()
