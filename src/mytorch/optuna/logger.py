# utils/logger.py

from pytorch_lightning.loggers import MLFlowLogger
import mlflow


def get_mlflow_logger(experiment_name="default_experiment"):
    mlflow_logger = MLFlowLogger(experiment_name=experiment_name)
    # Ensure the MLflow logger does not create its own run
    mlflow_logger._run_id = mlflow.active_run().info.run_id
    return mlflow_logger
