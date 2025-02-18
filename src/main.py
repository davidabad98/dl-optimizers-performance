import torch
import torch.nn as nn

from data_loader import get_kmnist_dataloaders, load_kmnist
from hyperparameter_tuner import tune_and_store_best_params
from model import KMNISTModel, get_optimizer
from training_pipeline import train_and_evaluate
from utils.logger import Logger
from utils.report_generator import generate_report

if __name__ == "__main__":

    # Device Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define Hyperparameter Grid
    hyperparameter_grid = {
        "adamw": {"lr": [0.0001, 0.001, 0.01], "weight_decay": [0.0001, 0.001]},
        "adam": {"lr": [0.0001, 0.001, 0.01], "weight_decay": [0.0001, 0.001]},
        "rmsprop": {
            "lr": [0.0001, 0.001, 0.01],
            "momentum": [0.8, 0.9],
            "alpha": [0.9, 0.99],
        },
    }

    # Define Hyperparams
    # EPOCHS = 10

    # Load Train Data
    train_data = load_kmnist()

    logger = Logger()  # This will redirect print statements to the log file
    print("Starting training...")

    # Tune and store best hyperparameters
    best_hyperparams = tune_and_store_best_params(
        hyperparameter_grid, train_data, k=5, epochs=10
    )

    # Initialize Models and Optimizers
    models = {}

    for optimizer_name, params in best_hyperparams.items():
        model = KMNISTModel().to(device)
        optimizer = get_optimizer(model, optimizer_name=optimizer_name, **params)
        models[optimizer_name] = (model, optimizer)

    # Load Train, Validation, Test Data
    train_loader, val_loader, test_loader = get_kmnist_dataloaders(
        batch_size=64, num_workers=2, valid_split=0.1
    )

    # Train and Evaluate Models
    criterion = nn.CrossEntropyLoss()
    results = train_and_evaluate(
        models, (train_loader, val_loader, test_loader), device, criterion, 10
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
