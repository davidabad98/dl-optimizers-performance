import time

import torch

from evaluation import evaluate_model
from model import KMNISTModel
from src.utils.logger import Logger
from train import train_model


def train_and_evaluate(models, datasets, device, criterion, epochs=10):
    """Train, validate, and evaluate multiple models on KMNIST."""
    results = {}  # Store results for comparison

    for optimizer_name, (model, optimizer) in models.items():
        print(f"\nTraining model using {optimizer_name.upper()}...\n")
        model.to(device)

        # Unpack datasets
        train_loader, val_loader, test_loader = datasets

        # Training loop
        start_time = time.time()
        train_loss, val_loss, train_acc, val_acc = train_model(
            model, device, train_loader, val_loader, epochs, optimizer, criterion
        )
        test_loss, test_acc = evaluate_model(model, test_loader, device, criterion)
        end_time = time.time()
        training_time = end_time - start_time

        # Store results
        results[optimizer_name] = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "test_loss": test_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "test_acc": test_acc,
            "time": round(training_time, 2),
        }

        print(f"Training completed for {optimizer_name.upper()}.\n")

    return results  # Return performance metrics
