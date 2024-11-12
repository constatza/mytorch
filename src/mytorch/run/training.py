# training.py

import pytorch_lightning as pl
import mlflow
import mlflow.pytorch
from models.model import LitModel  # Assuming your model is defined in models/model.py
from data.data_module import (
    DataModule,
)  # Assuming your data module is defined in data/data_module.py


def main():
    # Enable MLflow autologging
    mlflow.pytorch.autolog(
        log_models=False
    )  # Set log_models=False to prevent automatic model logging

    # Initialize data module and model
    data_module = DataModule()  # Replace with your actual data module initialization
    model = LitModel()  # Replace with your actual model initialization

    # Initialize trainer
    trainer = pl.Trainer(max_epochs=5, logger=False)  # Disable default logger

    # Start MLflow run
    with mlflow.start_run() as run:
        # Log hyperparameters
        mlflow.log_params(model.hparams)

        # Train the model
        trainer.fit(model, datamodule=data_module)

        # Manually save the model checkpoint
        checkpoint_path = "model_checkpoint.ckpt"
        trainer.save_checkpoint(checkpoint_path)

        # Log the checkpoint as an artifact to MLflow
        mlflow.log_artifact(checkpoint_path, artifact_path="model_checkpoint")

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
