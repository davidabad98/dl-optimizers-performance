import torch
import torch.nn as nn

from data_loader import get_kmnist_dataloaders, load_kmnist
from evaluation import (
    plot_test_accuracy,
    plot_train_metrics,
    plot_training_time,
    plot_validation_metrics,
)
from hyperparameter_tuner import tune_and_store_best_params
from model import KMNISTModel, get_optimizer
from training_pipeline import train_and_evaluate
from utils.logger import Logger
from utils.report_generator import generate_report

if __name__ == "__main__":

    # Device Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define Hyperparams
    EPOCHS = 15

    logger = Logger()  # This will redirect print statements to the log file
    print("Starting training...")

    best_hyperparams = {
        "adam": {
            "lr": 0.001,
            "weight_decay": 0.0001,
            "betas": (0.85, 0.995),
            "eps": 1e-08,
            "amsgrad": False,
        },
        "adamw": {
            "lr": 0.001,
            "weight_decay": 0.001,
            "betas": (0.85, 0.995),
            "eps": 1e-08,
            "amsgrad": True,
        },
        "rmsprop": {
            "lr": 0.0001,
            "momentum": 0.8,
            "alpha": 0.99,
            "eps": 1e-07,
            "centered": False,
            "weight_decay": 1e-05,
        },
    }

    # Initialize Models and Optimizers
    models = {}

    for optimizer_name, params in best_hyperparams.items():
        model = KMNISTModel().to(device)
        optimizer = get_optimizer(model, optimizer_name=optimizer_name, **params)
        models[optimizer_name] = (model, optimizer)

    # Load Train, Validation, Test Data
    train_loader, val_loader, test_loader = get_kmnist_dataloaders(
        batch_size=64, num_workers=2, valid_split=0.2
    )

    # Train and Evaluate Models
    criterion = nn.CrossEntropyLoss()
    results, visualization_results = train_and_evaluate(
        models, (train_loader, val_loader, test_loader), device, criterion, EPOCHS
    )

    # Print Results
    for opt, res in results.items():
        print(f"\nOptimizer: {opt.upper()}")
        print(
            f"Train Accuracy: {res['train_acc']:.4f} | Train Loss: {res['train_loss']:.4f}"
        )
        print(
            f"Validation Accuracy: {res['val_acc']:.4f} | Validation Loss: {res['val_loss']:.4f}"
        )
        print(
            f"Test Accuracy: {res['test_acc']:.4f} | Test Loss: {res['test_loss']:.4f}"
        )
        print(f"Training Time: {res['time']} seconds\n")

    # Generate and Save Report
    generate_report(results)

    # visualize results
    plot_train_metrics(visualization_results)
    plot_validation_metrics(visualization_results)
    plot_training_time(results)
    plot_test_accuracy(results)
