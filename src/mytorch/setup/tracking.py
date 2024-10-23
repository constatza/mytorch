from polars.dependencies import subprocess

from mytorch.io.logging import get_logger
import mlflow

logger = get_logger(__name__)


def initialize_mlflow(config: dict):
    # Setup MLflow
    experiment_name = str(config["mlflow"]["experiment_name"])
    tracking_uri = str(config["paths"]["tracking_uri"])
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment(experiment_name)
    logger.info(
        f"MLflow experiment name: {experiment_name}, ID: {experiment.experiment_id}"
    )
    logger.info(f"MLflow tracking URI: {tracking_uri}")

    subprocess.Popen(
        ["mlflow", "ui", "--backend-store-uri", tracking_uri, "--host", "localhost"]
    )
    return experiment.experiment_id
