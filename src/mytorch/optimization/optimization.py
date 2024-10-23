# optimization/objective.py
from wrapt import when_imported

import optuna
from mytorch.setup.datamodule import initialize_datamodule
from mytorch.setup.model import initialize_model
from mytorch.setup.trainer import initialize_trainer
from mytorch.setup.trial import suggest
import mlflow


def objective(trial, config, dataset_module):

    trial_hparams = suggest(trial, config)
    model = initialize_model(trial_hparams, dataset_module.shapes)
    trainer = initialize_trainer(config)

    with mlflow.start_run(run_name=f"{trial.number}"):
        mlflow.pytorch.autolog(log_models=config["mlflow"].get("log_models", False))
        mlflow.log_params(trial_hparams)

        dataset_module.setup(stage="fit")
        trainer.fit(model, datamodule=dataset_module)

        val_loss = trainer.callback_metrics.get("val_loss")

        if val_loss is not None:
            # Log metrics to MLflow
            trial.report(val_loss.item(), step=trainer.current_epoch)
            # Handle pruning based on the intermediate value
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            return val_loss
        else:
            raise ValueError("Validation loss not found in callback metrics.")
