from data_loader import load_kmnist
from hyperparameter_tuner import hyperparameter_tuning
from utils.logger import Logger

if __name__ == "__main__":

    hyperparameter_grid = {
        "adamw": {"lr": [0.0001, 0.001, 0.01], "weight_decay": [0.0001, 0.001]},
        "adam": {"lr": [0.0001, 0.001, 0.01], "weight_decay": [0.0001, 0.001]},
        "rmsprop": {
            "lr": [0.0001, 0.001, 0.01],
            "momentum": [0.8, 0.9],
            "alpha": [0.9, 0.99],
        },
    }

    train_data = load_kmnist()

    logger = Logger()  # This will redirect print statements to the log file
    print("Starting training...")

    for optimizer_name in ["adamw", "adam", "rmsprop"]:
        best_params = hyperparameter_tuning(
            optimizer_name,
            hyperparameter_grid[optimizer_name],
            train_data,
            k=5,
            epochs=10,
        )
        print(f"Best parameters for {optimizer_name}: {best_params}")
