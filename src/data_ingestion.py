
from pathlib import Path

import pandas as pd

from yaml_loader import load_config


def load_datasets():
    """Load training and test datasets using config.yaml."""

    config = load_config()
    project_root = Path(__file__).resolve().parent.parent

    train_path = project_root / config["data"]["loan_train.csv"]
    test_path = project_root / config["data"]["loan_test.csv"]

    if not train_path.is_file():
        raise FileNotFoundError(
            f"Training dataset not found: {train_path}"
        )

    if not test_path.is_file():
        raise FileNotFoundError(
            f"Test dataset not found: {test_path}"
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if train_df.empty:
        raise ValueError("Training dataset is empty.")

    if test_df.empty:
        raise ValueError("Test dataset is empty.")

    print("Datasets loaded successfully.")
    print(f"Training shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")

    print("\nTraining columns:")
    print(train_df.columns.tolist())

    return train_df, test_df


if __name__ == "__main__":
    train_df, test_df = load_datasets()

    print("\nFirst 5 training rows:")
    print(train_df.head())
