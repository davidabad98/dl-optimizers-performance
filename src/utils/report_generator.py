import os

import pandas as pd


def generate_report(results, save_path="results/training_report.csv"):
    """Generate a structured table from training results and save as CSV."""

    # Ensure results directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Convert results dictionary to Pandas DataFrame
    df = pd.DataFrame.from_dict(results, orient="index")

    # Rename columns for readability
    df.rename(
        columns={
            "train_acc": "Train Accuracy",
            "train_loss": "Train Loss",
            "val_acc": "Validation Accuracy",
            "val_loss": "Validation Loss",
            "test_acc": "Test Accuracy",
            "test_loss": "Test Loss",
            "time": "Training Time (s)",
        },
        inplace=True,
    )

    # Print the table
    print("\nTraining Report:")
    print(df.to_markdown())  # Pretty-print table output

    # Save as CSV
    df.to_csv(save_path, index_label="Optimizer")
    print(f"\nReport saved to: {save_path}")
