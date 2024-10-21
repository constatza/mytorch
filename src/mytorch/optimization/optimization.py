# optimization/objective.py
from wrapt import when_imported

from mytorch.optuna.datamodule import initialize_datamodule
from mytorch.optuna.model import initialize_model
from mytorch.optuna.trainer import initialize_trainer
from mytorch.optuna.optimizer import initialize_optimizer as initialize_optimizer
from mytorch.optuna.trial import suggest
import mlflow
import optuna


def flatten_dict(d, parent_key="", sep="/"):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def objective(trial, config):

    trial_hparams = suggest(trial, config)
    dataset_module = initialize_datamodule(trial, trial_hparams["datamodule"])
    model = initialize_model(trial, trial_hparams["model"], dataset_module)
    optimizer = initialize_optimizer(trial, trial_hparams["optimizer"], model)
    scheduler = initialize_scheduler(trial, trial_hparams["scheduler"], optimizer)

    with mlflow.start_run(nested=True):

        # Combine trial parameters
        trainer, mlflow_logger = initialize_trainer(trial_hparams["trainer"])

        # Log hyperparameters to MLflow
        mlflow_logger.log_hyperparams(trial_hparams)

        # Train the model
        dataset_module.prepare_data()
        dataset_module.setup()
        trainer.fit(model, datamodule=dataset_module)

        # Evaluate and log metrics
        val_loss = trainer.callback_metrics.get("val_loss")
        if val_loss is not None:
            val_loss = val_loss.item()
            # Log metrics to MLflow
            mlflow_logger.log_metrics({"val_loss": val_loss})
            # Report the metric to Optuna
            trial.report(val_loss, step=trainer.current_epoch)
            # Handle pruning based on the intermediate value
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            return val_loss
        else:
            raise ValueError("Validation loss not found in callback metrics.")
