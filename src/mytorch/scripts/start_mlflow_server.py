from mytorch.io.logging import get_logger
import os
import signal
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse
import socket
import atexit

from mytorch.io.readers import read_toml

logger = get_logger(__name__)

DEFAULT_ARTIFACTS_DIR_NAME = "artifacts"


def check_port_available(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        try:
            sock.bind((host, port))
        except socket.error:
            logger.error(f"Port {port} is already in use.")
            sys.exit(1)


def uri_to_path(uri: str) -> Path:
    parsed_uri = urlparse(uri)
    if parsed_uri.scheme != "sqlite":
        raise ValueError(f"Invalid URI scheme: {parsed_uri.scheme}. Expected 'sqlite'.")

    path = parsed_uri.path.lstrip("/")
    if parsed_uri.netloc:  # Handle Windows drive letters
        path = f"{parsed_uri.netloc}:{path}"
    return Path(path)


def get_create_artifact_dir(backend_store_uri: str, artifact_dir_name: str) -> Path:
    uri_path: Path = uri_to_path(backend_store_uri)
    if not uri_path.parent.exists():
        logger.warning(f"Path {backend_store_uri} does not exist. Creating it.")
        uri_path.parent.mkdir(parents=True, exist_ok=True)

    artifact_dir = uri_path.parent / artifact_dir_name
    artifact_dir.mkdir(parents=True, exist_ok=True)
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


def terminate_process_tree(pid):
    """
    Terminates a process and its child processes.

    Args:
        pid (int): The PID of the parent process.
    """
    try:
        if os.name == "posix":
            os.killpg(os.getpgid(pid), signal.SIGTERM)  # Terminate process group
            logger.info(f"Terminated process group for PID: {pid}")
        elif os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], check=True)
            logger.info(f"Terminated process tree for PID: {pid}")
    except Exception as e:
        logger.error(f"Error terminating process tree for PID {pid}: {e}")


def setup_signal_handlers(process):
    """
    Sets up signal handlers to ensure graceful termination of the server process.

    Args:
        process (subprocess.Popen): The MLflow server process.
    """

    def cleanup(*args):
        logger.info("Terminating MLflow server...")
        terminate_process_tree(process.pid)
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)  # Handle Ctrl+C
    signal.signal(signal.SIGTERM, cleanup)  # Handle termination signals

    # Ensure cleanup on program exit
    atexit.register(cleanup)


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


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Start MLflow server.")
    parser.add_argument(
        "config",
        type=str,
        default="config.toml",
        help="Path to the configuration file.",
    )
    args = parser.parse_args()
    config = read_toml(args.config)
    server_config = config["mlflow"]["server"]
    server_config.update(**config["paths"])
    start_mlflow_server(server_config)
