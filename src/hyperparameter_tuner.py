import itertools

import torch
import torch.nn as nn

from cross_validation import get_k_folds
from model import KMNISTModel, get_optimizer
from train import train_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()


def hyperparameter_tuning(optimizer_name, search_space, train_data, k=5, epochs=10):
    """
    Runs cross-validation for different hyperparameter settings.
    """
    best_config, best_score = None, 0
    param_keys, param_values = zip(*search_space.items())
    param_combinations = [
        dict(zip(param_keys, v)) for v in itertools.product(*param_values)
    ]

    for params in param_combinations:
        print(f"Testing {optimizer_name} with params: {params}")
        total_val_acc = 0

        for fold, (train_idx, val_idx) in enumerate(get_k_folds(train_data, k)):
            print(f"Fold {fold+1}/{k}")
            train_subset = torch.utils.data.Subset(train_data, train_idx)
            val_subset = torch.utils.data.Subset(train_data, val_idx)

            train_loader = torch.utils.data.DataLoader(
                train_subset, batch_size=64, shuffle=True
            )
            val_loader = torch.utils.data.DataLoader(
                val_subset, batch_size=64, shuffle=False
            )

            model = KMNISTModel().to(device)
            optimizer = get_optimizer(model, optimizer_name=optimizer_name, **params)

            _, _, _, valid_metrics = train_model(
                model, device, train_loader, val_loader, epochs, optimizer, criterion
            )
            total_val_acc += max(valid_metrics)

        avg_val_acc = total_val_acc / k
        print(f"Avg Validation Accuracy for {params}: {avg_val_acc:.4f}")
        print("--------------------------------")

        if avg_val_acc > best_score:
            best_score = avg_val_acc
            best_config = params

    print(
        f"Best config for {optimizer_name}: {best_config} with validation accuracy: {best_score:.4f}"
    )
    return best_config


def tune_and_store_best_params(hyperparameter_grid, train_data, k=5, epochs=10):
    """Tune hyperparameters and store the best parameters for each optimizer."""
    best_hyperparams = {}

    for optimizer_name in ["adamw", "adam", "rmsprop"]:
        best_params = hyperparameter_tuning(
            optimizer_name,
            hyperparameter_grid[optimizer_name],
            train_data,
            k=k,
            epochs=epochs,
        )
        best_hyperparams[optimizer_name] = best_params
        print(f"Best parameters for {optimizer_name}: {best_params}")

    return best_hyperparams  # Return dictionary with best params for each optimizer
