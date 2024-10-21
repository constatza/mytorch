from pytorch_lightning import Trainer
from mytorch.optuna.logger import get_mlflow_logger


def initialize_trainer(config):
    trainer_params = config.get("trainer", {}).copy()
    experiment_name = trainer_params.pop("experiment_name", "default_experiment")
    mlflow_logger = get_mlflow_logger(experiment_name=experiment_name)
    trainer = Trainer(
        logger=mlflow_logger, **trainer_params  # Let Trainer handle default parameters
    )
    return trainer, mlflow_logger
