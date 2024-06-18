import torch
import numpy as np

from pydantic import BaseModel
from typing import Optional, Union

class TrainLogger(BaseModel):
    checkpoint_path: str
    checkpoint_relative_tol: float
    min_val_loss: float = float('inf')
    model: Optional[torch.nn.Module] = None


class RelativeToleranceLogger(TrainLogger):
    model: torch.nn.Module

    def log(self, epoch: int, train_loss: float, val_loss) -> None:
        print(f'Epoch {epoch} | Train Loss: {train_loss:.5e} | Val Loss: {val_loss:.5e}')
        if self.relative_tolerance(self.min_val_loss, val_loss) > self.checkpoint_relative_tol:
            print('Checkpointing model...')
            self.min_val_loss = val_loss
            best_model = torch.jit.script(self.model)
            best_model.save(self.checkpoint_path)

    @staticmethod
    def relative_tolerance(min_loss, loss, past_n=10):
        mean = sum(loss[-past_n-1:]) / past_n
        return (min_loss - mean) / mean

class TrainerConfig(BaseModel):
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    loss_function: torch.nn.modules.loss._Loss
    train_loader: torch.utils.data.DataLoader
    test_loader: torch.utils.data.DataLoader
    num_epochs: int
    device: torch.device
    logger: Optional[TrainLogger] = None
    unique_id: Optional[Union[str,int]] = None


class Trainer:

    def __init__(self, config: TrainerConfig):
        for key, value in config.dict().items():
            setattr(self, key, value)


    def train_step(self, epoch):
        raise NotImplementedError

    def test_step(self, epoch):
        raise NotImplementedError

    def train(self, num_epochs):
        self.model.to(self.device)

        for epoch in range(num_epochs):
            train_loss = self.train_step(epoch)
            test_loss = self.test_step(epoch)
            self.logger()

class VAETrainer(Trainer):

    def __init__(self, config: TrainerConfig):
        super().__init__(config)

    def train_step(self, epoch):
        """ Perform a training step.

        Args:
            epoch (int): The current epoch number.

        Returns:
            float: The training loss.
        """
        self.model.train()  # Set the model to training mode
        train_loss = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_function(output, target)
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
            for batch_idx, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.loss_function(output, target)
                test_loss += loss.item() * data.size(0)
        return test_loss / len(self.test_loader.dataset)





class CAETrainer(Trainer):

    def __init__(self, config: TrainerConfig):
        super().__init__(config)



    def train_step(self, epoch):
        self.model.train()
        train_loss = 0
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
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in self.dataloader_val:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                loss_val = self.criterion(self.model(x_batch), y_batch)
                test_loss += loss_val.item() * x_batch.size(0)
        return test_loss / len(self.dataloader_val.dataset)

def train(self):
    train_losses = []
    val_losses = []
    min_val_loss = float('inf')
    self.model.to(self.device)
    for epoch in range(self.num_epochs):
        train_loss = self.train_step()
        train_losses.append(train_loss)
        val_loss = self.test_step()
        val_losses.append(val_loss)
        min_val_loss = self.log_function(epoch, train_losses[-1], val_losses[-1], self.model, self.checkpoint_path, self.checkpoint_relative_tol, min_val_loss)
    return train_losses, val_losses

def relative_tolerance(min_loss, losses, past_n=10):
    mean = sum(losses[-past_n-1:]) / past_n
    return (min_loss - mean) / mean


def print_losses(self, train_loss, val_loss, epoch) -> None:
    epoch_for_print = epoch + 1
    if epoch_for_print % self.epoch_print_interval == 0:
        text = f'Epoch {epoch_for_print}/{self.num_epochs} | Train Loss: {train_loss:.5e} | Val Loss: {val_loss:.5e}'
        print(text)

