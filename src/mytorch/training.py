# main.py

import optuna
import argparse

from mytorch.io.readers import load_config
from mytorch.optimization import objective
from mytorch.setup.pruner import initialize_pruner
from mytorch.io.logging import get_logger
from mytorch.setup.tracking import initialize_mlflow
from mytorch.setup.datamodule import initialize_datamodule


logger = get_logger(__name__)


def run_optimization(config_path):
    config = load_config(config_path)

    dataset_module = initialize_datamodule(config)
    dataset_module.prepare_data()

    # Read n_trials from config
    optuna_config = config.get("optuna", {})
    n_trials = optuna_config.get("n_trials", 100)

    # setup mlflow experiment and tracking uri
    experiment_id = initialize_mlflow(config)

    # Setup pruner
    pruner = initialize_pruner(optuna_config.get("pruner", {}))
    # Start the MLflow UI in a subprocess
    study = optuna.create_study(
        direction=config["optuna"].get("direction", "minimize"),
        pruner=pruner,
        study_name=f"study_{experiment_id}",
    )
    study.optimize(
        lambda trial: objective(trial, config, dataset_module),
        n_trials=n_trials,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter optimization using Optuna and MLFlow."
    )
    parser.add_argument(
        "config",
        type=str,
        default="config.yaml",
        help="Path to the configuration file.",
    )
    args = parser.parse_args()
    run_optimization(args.config)


if __name__ == "__main__":
    main()
