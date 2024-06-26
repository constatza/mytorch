import torch
import numpy as np
from pydantic import BaseModel, validator
from typing import Optional, Union

from mytorch.io.config import TrainingConfig


# Define the Trainer class
class Trainer(BaseModel):
    """
    A class for training a PyTorch model.

    Attributes:
        config (TrainingConfig): The configuration for the Trainer.
    """

    config: TrainingConfig


    def __getattr__(self, item):
        """
        Get an attribute from the Trainer's configuration.

        Args:
            item (str): The name of the attribute.

        Returns:
            The attribute from the Trainer's configuration.
        """
        return getattr(self.config, item)

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
            loss = self.criterion(y_pred, y_batch)
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
                loss_val = self.criterion(self.model(x_batch), y_batch)
                test_loss += loss_val.item() * x_batch.size(0)
        return test_loss / len(self.test_loader.dataset)

    def train(self, num_epochs):
        """
        Train the model for a specified number of epochs.

        Args:
            num_epochs (int): The number of epochs for training.

        Returns:
            tuple: The training and validation losses for each epoch.
        """
        train_losses = []
        val_losses = []
        self.model.to(self.device)
        for epoch in range(num_epochs):
            train_loss = self.train_step()
            test_loss = self.test_step()
            train_losses.append(train_loss)
            val_losses.append(test_loss)
            self.logger.log(epoch, train_loss, val_losses)

        return np.array(train_losses), np.array(val_losses)

class AutoEncoderTrainer(Trainer):
    """
    A class for training an autoencoder.

    This class inherits from the Trainer class and overrides the train_step and test_step methods.
    """

    @validator('config')
    def ensure_autoencoder_loader(cls, value):
        """
        Ensure that the training and test loaders are for the input data only.

        Args:
            value (TrainingConfig): The configuration for the Trainer.

        Raises:
            ValueError: If the training or test loaders contain target data.
        """
        # check shape of dataset
        if len(value.train_loader.dataset.shape) > 2:
            raise ValueError("Autoencoder requires input data only")
        return value


    def train_step(self):
        """
        Perform a training step for an autoencoder.

        Returns:
            float: The average training loss for this step.
        """
        self.model.train()  # Set the model to training mode
        train_loss = 0
        for data in self.train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, data)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item() * data.size(0)
        return train_loss / len(self.train_loader.dataset)

    def test_step(self):
        """
        Perform a test step for an autoencoder.

        Args:
            epoch (int): The current epoch number.

        Returns:
            float: The average test loss for this step.
        """
        self.model.eval()  # Set the model to evaluation mode
        test_loss = 0
        with torch.no_grad():
            for data in self.test_loader:
                data = data.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, data).item() * data.size(0)
        return test_loss / len(self.test_loader.dataset)