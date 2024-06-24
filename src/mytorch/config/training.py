
import torch
from typing import Optional, Union
from pydantic import BaseModel
from uuid import uuid4

from .loggers import ProgressLogger, TrainLogger


# Define the configuration for the Trainer class
class TrainingConfig(BaseModel):
    class Config:
        arbitrary_types_allowed = True
        extra = 'ignore'

    model: torch.nn.Module  # The model to be trained
    optimizer: torch.optim.Optimizer  # The optimizer for training the model
    criterion: torch.nn.modules.loss._Loss  # The loss function
    train_loader: torch.utils.data.DataLoader  # The DataLoader for the training data
    test_loader: torch.utils.data.DataLoader  # The DataLoader for the test data
    num_epochs: int  # The number of epochs for training
    device: torch.device  # The device to train on (e.g., 'cpu' or 'cuda')
    logger: Optional[TrainLogger] = ProgressLogger(console=True)  # The logger for training
    unique_id: Optional[Union[uuid4, str, int]] = None  # A unique identifier for the Trainer
