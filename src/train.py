import time

import torch
import torch.nn as nn
import torch.optim as optim


def train_model(
    model, device, train_loader, valid_loader, epochs, optimizer, loss_funtion
):
    """
    Trains and evaluates the model.

    Args:
        model (nn.Module): Neural network model.
        device (torch.device): Device to use for training.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        epochs (int): Number of epochs.
        optimizer (torch.optim.Optimizer): Optimizer.
        criterion (nn.Module): Loss function.

    Returns:
        Tuple (list, list, list, list): Training losses, validation losses, train metrics, validation metrics.
    """

    model.to(device)

    # Lists to store training and validation losses/metrics for each epoch.
    train_losses, valid_losses = [], []
    train_metrics, valid_metrics = [], []

    for epoch in range(epochs):
        model.train()  # This puts the model into training mode
        train_loss, correct_train, total_train = (
            0.0,
            0.0,
            0,
        )  # Track loss, accuracy, total samples
        start_time = time.time()

        # Training loop
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move to GPU
            optimizer.zero_grad()  # Clear gradients
            outputs = model(inputs)  # Forward pass
            loss = loss_funtion(outputs, labels)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            train_loss += loss.item()  # Accumulate loss
            _, predicted = outputs.max(1)
            correct_train += predicted.eq(labels).sum().item()  # Accumulates metric
            total_train += labels.size(0)

        # Compute training loss and accuracy for the epoch
        train_loss /= total_train
        train_acc = correct_train / total_train
        train_losses.append(train_loss)
        train_metrics.append(train_acc)

        model.eval()  # Puts the model in evaluation mode (disables dropout, etc.)
        val_loss, correct_val, total_val = 0.0, 0, 0

        with torch.no_grad():
            for inputs, labels in valid_loader:  # loop through validation dataset
                inputs, labels = inputs.to(device), labels.to(device)  # Move to GPU
                outputs = model(inputs)  # Forward pass
                loss = loss_funtion(outputs, labels)  # Compute loss

                # Accumulates validation loss and metric
                val_loss += loss.item()  # Accumulate loss for this batch
                _, predicted = outputs.max(1)  # Get predicted class
                correct_val += (
                    predicted.eq(labels).sum().item()
                )  # Count correct predictions
                total_val += labels.size(0)  # Keep track of total samples

        # Computes the validation loss and metric. Stores them in lists.
        val_loss /= total_val
        val_acc = correct_val / total_val
        valid_losses.append(val_loss)
        valid_metrics.append(val_acc)

        end_time = time.time()

        print(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Time: {end_time-start_time:.2f}s"
        )

    return train_losses, valid_losses, train_metrics, valid_metrics
