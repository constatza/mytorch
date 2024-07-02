import shutil
from pathlib import Path

import numpy as np
import optuna
import torch
from optuna.storages import RetryFailedTrialCallback

from mytorch.io.config import TrainingConfig
from mytorch.io.loggers import TrainLogger


# Define the Trainer class
class Trainer:
    """
    A class for training a PyTorch model.

    Attributes:
        config (TrainingConfig): The configuration for the Trainer.
    """

    def __init__(
        self,
        trial: optuna.Trial,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        criterion: torch.nn.modules.loss._Loss,
        optimizer: type(torch.optim.Optimizer),
        device: torch.device,
        learning_rate: float,
        logger: TrainLogger,
        models_dir: Path,
        num_epochs: int,
        delete_old: bool,
    ):
        """
        Initialize the Trainer.

        """
        self.trial = trial
        self.model = model
        self.num_epochs = num_epochs
        self.delete_old = delete_old
        self.model_name = str(model.__class__.__name__)
        self.trial_checkpoint_dir = models_dir
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion()
        self.optimizer = optimizer(self.model.parameters(), lr=learning_rate)
        self.device = device
        self.logger = logger

        self.trial_number = RetryFailedTrialCallback.retried_trial_number(trial)
        if (
            self.delete_old
            and self.trial_number is not None
            and self.checkpoint_path.exists()
        ):
            checkpoint = torch.load(self.checkpoint_path)
            epoch = checkpoint["epoch"]
            self.epoch_begin = epoch

            logger.info(
                f"Loading a checkpoint from trial {self.trial_number} in epoch {epoch}."
            )

            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            loss = checkpoint["loss"]
        else:
            self.trial_checkpoint_dir = models_dir
            self.trial_number = trial.number
            shutil.rmtree(self.trial_checkpoint_dir, ignore_errors=True)
            self.epoch_begin = 0

        self.trial_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # A checkpoint may be corrupted when the process is killed during `torch.save`.
        # Reduce the risk by first calling `torch.save` to a temporary file, then copy.
        self.tmp_checkpoint_path = (
            self.trial_checkpoint_dir / f"{self.model_name}.pt.tmp"
        )

        logger.info(f"Checkpoint path for trial is '{self.checkpoint_path}'.")
        self.logger.info("Trainer initialized.")

    @property
    def checkpoint_path(self):
        return self.trial_checkpoint_dir / f"{self.model_name}_{self.trial_number}.pt"

    def train_step(self):
        """
        Perform a training step.

        Returns:
            float: The average training loss for this step.
        """
        train_loss = 0
        self.model.train()
        for x_batch, y_batch in self.train_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            self.optimizer.zero_grad()
            y_pred = self.model(x_batch)
            loss = self.best_criterion(y_pred, y_batch)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item() * x_batch.size(0)
        return train_loss / len(self.train_loader.dataset)

    def test_step(self):
        """
        Perform a test step.

        Returns:
            float: The average test loss for this step.
        """
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in self.test_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                loss_val = self.best_criterion(self.model(x_batch), y_batch)
                test_loss += loss_val.item() * x_batch.size(0)
        return test_loss / len(self.test_loader.dataset)

    def train(self):
        """
        Train the model for a specified number of epochs.

        Args:
            num_epochs (int): The number of epochs for training.

        Returns:
            tuple: The training and validation losses for each epoch.
        """
        self.logger.info("Training started.")
        train_losses = []
        val_losses = []
        self.model.to(self.device)
        for epoch in range(self.epoch_begin, self.num_epochs):
            train_loss = self.train_step()
            test_loss = self.test_step()

            train_losses.append(train_loss)
            val_losses.append(test_loss)

            self.logger.log(epoch, train_loss, val_losses)
            self.save_checkpoint(epoch, test_loss)

            # Handle pruning based on the intermediate value.
            self.trial.report(test_loss, epoch)
            if self.trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return np.array(train_losses), np.array(val_losses)

    def save_checkpoint(self, epoch, loss, step=10):
        if epoch % step == 0:
            self.logger.info(f"Saving a checkpoint in epoch {epoch}.")

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "loss": loss,
                },
                self.tmp_checkpoint_path,
            )
            shutil.move(self.tmp_checkpoint_path, self.checkpoint_path)

    def best_criterion(self, predictions, targets):
        match predictions:
            case y_pred if isinstance(y_pred, torch.Tensor):
                return self.criterion(y_pred, targets)
            case (y_pred, mu, logvar):
                from mytorch.metrics import mse_plus_kl_divergence

                return mse_plus_kl_divergence(y_pred, targets, mu, logvar)
