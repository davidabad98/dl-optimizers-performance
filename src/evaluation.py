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

    return loss, accuracy


def plot_train_metrics(results, save_path="results/training_plot.png"):
    """
    Plots the training loss and accuracy of different optimizers.
    """
    plt.figure(figsize=(12, 5))

    # Plot Training Loss
    plt.subplot(1, 2, 1)
    for optimizer_name, res in results.items():
        plt.plot(res["train_loss_final"], label=f"{optimizer_name} (Train)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Across Optimizers")
    plt.legend()

    # Plot Training Accuracy
    plt.subplot(1, 2, 2)
    for optimizer_name, res in results.items():
        plt.plot(res["train_acc_final"], label=f"{optimizer_name} (Train)")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy Across Optimizers")
    plt.legend()
    plt.savefig(save_path)
    print(f"Training over epochs plot saved as {save_path}")
    plt.tight_layout()
    plt.show()


def plot_validation_metrics(results, save_path="results/validation_plot.png"):
    """
    Plots the validation loss and accuracy of different optimizers.
    """
    plt.figure(figsize=(12, 5))

    # Plot Validation Loss
    plt.subplot(1, 2, 1)
    for optimizer_name, res in results.items():
        plt.plot(
            res["val_loss_final"],
            linestyle="dashed",
            label=f"{optimizer_name} (Validation)",
        )
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Validation Loss Across Optimizers")
    plt.legend()

    # Plot Validation Accuracy
    plt.subplot(1, 2, 2)
    for optimizer_name, res in results.items():
        plt.plot(
            res["val_acc_final"],
            linestyle="dashed",
            label=f"{optimizer_name} (Validation)",
        )
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy Across Optimizers")
    plt.legend()
    plt.savefig(save_path)
    print(f"Validation over epochs plot saved as {save_path}")
    plt.tight_layout()
    plt.show()


# Function to plot training time comparison
def plot_training_time(results, save_path="results/training_time.png"):
    """
    Plots a bar chart showing training time for different optimizers.
    """
    optimizer_names = list(results.keys())
    training_times = [res["time"] for res in results.values()]

    plt.figure(figsize=(8, 5))
    plt.bar(optimizer_names, training_times, color=["blue", "orange", "green"])
    plt.xlabel("Optimizers")
    plt.ylabel("Training Time (seconds)")
    plt.title("Training Time Comparison Across Optimizers")
    plt.savefig(save_path)
    print(f"Training Time plot saved as {save_path}")
    plt.show()


def plot_test_accuracy(results, save_path="results/testing_acc.png"):
    """
    Plots test accuracy as a line plot across different optimizers.

    Args:
        results (dict): Dictionary containing performance metrics for each optimizer.
    """
    optimizers = list(results.keys())
    test_accuracies = [results[opt]["test_acc"] for opt in optimizers]

    plt.figure(figsize=(8, 5))
    plt.plot(
        optimizers,
        test_accuracies,
        marker="o",
        linestyle="-",
        color="b",
        linewidth=2,
        markersize=8,
    )

    plt.xlabel("Optimizers")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy Across Optimizers")
    plt.ylim(0, 1)  # Accuracy ranges from 0 to 1
    plt.grid(True, linestyle="--", alpha=0.7)

    for i, acc in enumerate(test_accuracies):
        plt.text(
            i, acc + 0.02, f"{acc:.2f}", ha="center", fontsize=12, fontweight="bold"
        )

    plt.savefig(save_path)
    print(f"Test accuracy saved as {save_path}")
    plt.show()
