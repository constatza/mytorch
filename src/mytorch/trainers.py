import io
import sys

import torch
import numpy as np

from pydantic import BaseModel
from typing import Optional, Union, List





class TrainerConfig(BaseModel):
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    criterion: torch.nn.modules.loss._Loss
    train_loader: torch.utils.data.DataLoader
    test_loader: torch.utils.data.DataLoader
    num_epochs: int
    device: torch.device
    logger: Optional[TrainLogger] = None
    unique_id: Optional[Union[str,int]] = None


class Trainer:

    def __init__(self, config: TrainerConfig):
        self.config = config

    def __getattr__(self, item):
        return getattr(self.config, item)

    def train_step(self):
        """ Perform a training step."""
        train_loss = 0
        self.model.train()
        for x_batch, y_batch in self.dataloader_train:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            self.optimizer.zero_grad()
            y_pred = self.model(x_batch)
            loss = self.criterion(y_pred, y_batch)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item() * x_batch.size(0)
        return train_loss / len(self.dataloader_train.dataset)

    def test_step(self):
        """ Perform a test step."""
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in self.dataloader_val:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                loss_val = self.criterion(self.model(x_batch), y_batch)
                test_loss += loss_val.item() * x_batch.size(0)
        return test_loss / len(self.dataloader_val.dataset)

    def train(self, num_epochs):
        """ Train the model for a specified number of epochs. """
        train_losses = []
        val_losses = []
        self.model.to(self.device)
        for epoch in range(num_epochs):
            train_loss = self.train_step()
            test_loss = self.test_step()
            self.logger.log(epoch, train_loss, test_loss)
            train_losses.append(train_loss)
            val_losses.append(test_loss)

        return np.array(train_losses), np.array(val_losses)

class AutoEncoderTrainer(Trainer):

    def __init__(self, config: TrainerConfig):
        super().__init__(config)

    def train_step(self):
        self.model.train()  # Set the model to training mode
        train_loss = 0
        for data in self.train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_function(output, data)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item() * data.size(0)
        return train_loss / len(self.train_loader.dataset)

    def test_step(self, epoch):
        """
        Perform a test step.

        Args:
            epoch (int): The current epoch number.

        Returns:
            float: The test loss.
        """
        self.model.eval()  # Set the model to evaluation mode
        test_loss = 0
        with torch.no_grad():
            for data in self.test_loader:
                data = data.to(self.device)
                output = self.model(data)
                test_loss += self.loss_function(output, data).item() * data.size(0)
        return test_loss / len(self.test_loader.dataset)









