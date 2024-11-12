from mytorch.io.logging import get_logger
from mytorch.io.readers import read_toml
from mytorch.utils.mlflow_utils import start_mlflow_server

logger = get_logger(__name__)


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
    logger.info(f"Starting MLflow server with configuration.")
    server_config.update(**config["paths"])
    start_mlflow_server(server_config)
