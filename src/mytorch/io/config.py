
import torch
from typing import Optional, Union, List, Dict
from pydantic import BaseModel, FilePath, DirectoryPath, validate_call

from uuid import uuid4
from pathlib import Path

from loggers import ProgressLogger, TrainLogger
from utils import read_toml, join_root_with_paths, apply_to_dict, replace_placeholders


MaybeIntList = Union[int, List[int]]
MaybeFloatList = Union[float, List[float]]
MaybeStrList = Union[str, List[str]]


class TrainingConfig(BaseModel):
    class Config:
        arbitrary_types_allowed = True
        extra = 'ignore'

    model: torch.nn.Module  # The model to be trained
    train_loader: torch.utils.data.DataLoader  # The DataLoader for the training data
    test_loader: torch.utils.data.DataLoader  # The DataLoader for the test data
    num_epochs: MaybeIntList # The number of epochs for training
    batch_size: MaybeIntList # The batch size for training
    optimizer: Optional[torch.optim.Optimizer] = torch.optim.Adam  # The optimizer for training the model
    criterion: Optional[torch.nn.modules.loss._Loss] = torch.nn.MSELoss  # The loss function
    learning_rate: Optional[MaybeFloatList] = 0.001  # The learning rate for training
    device: Optional[torch.device] = 'cuda' if torch.cuda.is_available() else 'cpu'  # The device to train on (e.g., 'cpu' or 'cuda')
    logger: Optional[TrainLogger] = ProgressLogger(console=True)  # The logger for training
    unique_id: Optional[Union[uuid4, str, int]] = None  # A unique identifier for the Trainer





class ModelConfig(BaseModel):
    name: MaybeStrList
    kernel_size: Optional[MaybeIntList]
    num_layers: Optional[MaybeIntList]
    latent_size: Optional[MaybeIntList]
    hidden_size: Optional[MaybeIntList]


class TestConfig(BaseModel):
    model: MaybeStrList

class PathsInputConfig(BaseModel):
    root: Path
    x_train: Path
    x_test: Path
    y_train: Path
    y_test:  Path
    means: Optional[Path]
    stds: Optional[Path]
    latent: Optional[Path]

class PathsRawConfig(BaseModel):
    root: Optional[Path]
    dataset: Optional[Path]
    dofs: Optional[Path]


class PathsOutputConfig(BaseModel):
    root: Path
    figures: Optional[DirectoryPath]
    models: Optional[DirectoryPath]
    logs: Optional[DirectoryPath]
    results: Optional[DirectoryPath]
    config: Optional[DirectoryPath]

class PathsConfig(BaseModel):
    input: PathsInputConfig
    raw: PathsRawConfig
    output: PathsOutputConfig


class ScenarioConfig(BaseModel):
    class Config:
        extra = 'ignore'
    name: str
    model: ModelConfig
    training: TrainingConfig
    test: TestConfig
    paths: PathsConfig
    description: Optional[str]
    variable: Optional[str]



