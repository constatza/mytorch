# training.py

from lightning.pytorch import seed_everything
import mlflow
import mlflow.pytorch

from mytorch.io.readers import load_config
from mytorch.io.logging import get_logger
from mytorch.setup.tracking import initialize_mlflow
from mytorch.setup.datamodule import initialize_datamodule
from mytorch.setup.trainer import initialize_trainer
from mytorch.setup.model import initialize_model

seed_everything(1)


def main():
    # Enable MLflow autologging
    mlflow.pytorch.autolog(
        log_models=False
    )  # Set log_models=False to prevent automatic model logging

    # Initialize data module and model
    config = load_config("config.toml")
    data_module = initialize_datamodule(config)
    experiment_id = initialize_mlflow(config)
    trainer = initialize_trainer(config)
    model = initialize_model(config, data_module.shapes)

    # Start MLflow run
    with mlflow.start_run() as run:
        # Log hyperparameters
        mlflow.autolog(log_model_signatures=True, log_models=True)
        mlflow.log_params(model.hparams)

        # Train the model
        trainer.fit(model, datamodule=data_module)

        # Optionally, log the entire model using mlflow.pytorch.log_model
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            registered_model_name=None,  # Replace with a name if you want to register the model
        )

        # Print the run ID for reference
        run_id = run.info.run_id
        print(f"Training completed. Run ID: {run_id}")


if __name__ == "__main__":
    main()
