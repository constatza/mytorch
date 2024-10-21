# main.py

import optuna
from mytorch.io.readers import load_config
from mytorch.optimization import objective
import mlflow


def run_optimization(config_path):
    config = load_config(config_path)

    # Read n_trials from config
    optuna_config = config.get("optuna", {})
    n_trials = optuna_config.get("n_trials", 100)

    # Setup pruner
    pruner_config = config.get("pruner", {})
    pruner_name = pruner_config.get("name", "MedianPruner")
    pruner_class = getattr(optuna.pruners, pruner_name, optuna.pruners.MedianPruner)
    pruner_params = {k: v for k, v in pruner_config.items() if k != "name"}
    pruner = pruner_class(**pruner_params)

    # Start MLflow parent run
    with mlflow.start_run(run_name="Optuna_Study"):
        study = optuna.create_study(direction="minimize", pruner=pruner)
        study.optimize(lambda trial: objective(trial, config), n_trials=n_trials)

        print("Best trial:", study.best_trial.params)
        # Log best trial parameters
        mlflow.log_params(study.best_trial.params)
        # Log best trial value
        mlflow.log_metric("best_val_loss", study.best_trial.value)


if __name__ == "__main__":
    run_optimization(
        r"C:\Users\cluster\constantinos\mytorch\studies\bio-surrogate\cae\config.toml"
    )
