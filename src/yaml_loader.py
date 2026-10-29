from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"

SUPPORTED_ALGORITHMS = {
    "logistic_regression",
    "random_forest",
    "lightgbm",
    "catboost",
}


def validate_config(config: dict) -> None:
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping.")

    for section in ("data", "training", "mlflow", "counterfactual"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing or invalid section: {section}")

    data = config["data"]

    for key in ("train_data", "test_data", "target_column"):
        value = data.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"data.{key} must be a non-empty string.")

    drop_columns = data.get("drop_columns", [])
    if not isinstance(drop_columns, list) or not all(
        isinstance(column, str) for column in drop_columns
    ):
        raise ValueError("data.drop_columns must be a list of strings.")

    training = config["training"]
    algorithms = training.get("algorithms")

    if not isinstance(algorithms, list) or not algorithms:
        raise ValueError("training.algorithms must be a non-empty list.")

    if len(algorithms) != len(set(algorithms)):
        raise ValueError("training.algorithms contains duplicates.")

    unsupported = set(algorithms) - SUPPORTED_ALGORITHMS
    if unsupported:
        raise ValueError(f"Unsupported algorithms: {sorted(unsupported)}")

    random_state = training.get("random_state")
    if (
        isinstance(random_state, bool)
        or not isinstance(random_state, int)
        or random_state < 0
    ):
        raise ValueError("training.random_state must be a non-negative integer.")

    validation_size = training.get("validation_size")
    if (
        isinstance(validation_size, bool)
        or not isinstance(validation_size, (int, float))
        or not 0 < validation_size < 1
    ):
        raise ValueError("training.validation_size must be between 0 and 1.")

    hyperparameters = training.get("hyperparameters", {})
    if not isinstance(hyperparameters, dict):
        raise ValueError("training.hyperparameters must be a mapping.")

    n_estimators = hyperparameters.get("n_estimators", 200)
    if (
        isinstance(n_estimators, bool)
        or not isinstance(n_estimators, int)
        or n_estimators <= 0
    ):
        raise ValueError("n_estimators must be a positive integer.")

    learning_rate = hyperparameters.get("learning_rate", 0.05)
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or not 0 < learning_rate <= 1
    ):
        raise ValueError("learning_rate must be greater than 0 and at most 1.")

    experiment_name = config["mlflow"].get("experiment_name")
    if not isinstance(experiment_name, str) or not experiment_name.strip():
        raise ValueError("mlflow.experiment_name must be a non-empty string.")

    enabled = config["counterfactual"].get("enabled")
    if not isinstance(enabled, bool):
        raise ValueError("counterfactual.enabled must be true or false.")


def load_config() -> dict:
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    validate_config(config)
    return config


if __name__ == "__main__":
    loaded_config = load_config()
    print("Configuration loaded and validated successfully.")
    print(loaded_config)
