# Importing required libraries
from mytorch.io.logging import get_logger
import mlflow
import subprocess

from mytorch.utils.mlflow_utils import get_create_artifact_dir

logger = get_logger(__name__)


def initialize_mlflow(config: dict):
    # Setup MLflow

    tracking_uri = str(config["paths"]["tracking_uri"])
    artifact_location = str(get_create_artifact_dir(tracking_uri, "artifacts"))

    logger.info(f"MLflow tracking URI: {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)

    experiment_name = str(config["mlflow"]["experiment_name"])
    # Check if the experiment exists
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        # Create the experiment with the desired artifact location
        experiment_id = mlflow.create_experiment(
            name=experiment_name, artifact_location=artifact_location
        )
        logger.info(
            f"Created MLflow experiment '{experiment_name}' with ID: {experiment_id}"
        )
    else:
        experiment_id = experiment.experiment_id
        logger.info(
            f"Found MLflow experiment name '{experiment_name}' with  ID: {experiment_id}"
        )

    mlflow.set_experiment(experiment_name)
    return experiment_id
