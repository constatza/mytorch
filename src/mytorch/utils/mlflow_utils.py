from pathlib import Path

import subprocess


from mytorch.io.logging import get_logger
from mytorch.utils.processing import uri_to_path
from mytorch.utils.system_utils import check_port_available, setup_signal_handlers
from mytorch.utils.defaults import DEFAULT_ARTIFACTS_DIR_NAME


logger = get_logger(__name__)


def get_create_artifact_dir(backend_store_uri: str, artifact_dir_name: str) -> Path:
    uri_path: Path = uri_to_path(backend_store_uri)
    if not uri_path.parent.exists():
        logger.warning(f"Path {backend_store_uri} does not exist. Creating it.")
        uri_path.parent.mkdir(parents=True, exist_ok=True)

    artifact_dir = uri_path.parent / artifact_dir_name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Artifact directory: {artifact_dir}")
    return artifact_dir


def start_mlflow_process(host, port, backend_store_uri, artifact_root):
    command = [
        "mlflow",
        "server",
        "--host",
        host,
        "--port",
        str(port),
        "--backend-store-uri",
        backend_store_uri,
        "--default-artifact-root",
        str(artifact_root),
    ]

    logger.info(f"Starting MLflow server on {host}:{port}")
    logger.info(f"Backend store URI: {backend_store_uri}")
    logger.info(f"Artifact root: {artifact_root}")

    process = subprocess.Popen(command)

    logger.info(f"MLflow server started with PID: {process.pid}")
    return process


def start_mlflow_server(config):
    """
    Starts the MLflow server and ensures cleanup on exit.

    Args:
        config (dict): Configuration dictionary containing:
            - backend_store_uri (str): The backend store URI.
            - host (str): The host to bind the server.
            - port (int): The port to bind the server.
            - default-artifact-root (str, optional): The default artifact root directory.
    """
    backend_store_uri = config["backend_store_uri"]
    host = config["host"]
    port = config["port"]
    artifact_dir_name = DEFAULT_ARTIFACTS_DIR_NAME

    check_port_available(host, port)

    artifact_root = get_create_artifact_dir(backend_store_uri, artifact_dir_name)
    logger.info(f"Artifact directory: {artifact_root}")

    process = start_mlflow_process(host, port, backend_store_uri, artifact_root)

    setup_signal_handlers(process)

    logger.info("MLflow server started. Press Ctrl+C to stop.")
    process.wait()
