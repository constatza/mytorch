import torch
from typing import Optional, Union, List
from pydantic import BaseModel, DirectoryPath, Field

from uuid import uuid4
from pathlib import Path

from mytorch.io.loggers import ProgressLogger, TrainLogger

MaybeIntList = Union[int, List[int]]
MaybeFloatList = Union[float, List[float]]
MaybeStrList = Union[str, List[str]]


class BasicConfig(BaseModel):
    class Config:
        extra = 'ignore'
        alias_generator = lambda s: s.replace('_', '-')
        populate_by_name = True
        allow_population_by_alias = True
        arbitrary_types_allowed = True
        frozena = True

class TrainingConfig(BasicConfig):

    model: torch.nn.Module  # The model to be trained
    train_loader: torch.utils.data.DataLoader  # The DataLoader for the training data
    test_loader: torch.utils.data.DataLoader  # The DataLoader for the test data
    num_epochs: MaybeIntList  # The number of epochs for training
    batch_size: MaybeIntList  # The batch size for training
    optimizer: Optional[torch.optim.Optimizer] = torch.optim.Adam  # The optimizer for training the model
    criterion: Optional[torch.nn.modules.loss._Loss] = torch.nn.MSELoss  # The loss function
    learning_rate: Optional[MaybeFloatList] = Field(default=1e-3, alias='lr')  # The learning rate for training
    device: Optional[torch.device] = 'cuda' if torch.cuda.is_available() \
        else 'cpu'  # The device to train on (e.g., 'cpu' or 'cuda')
    logger: Optional[TrainLogger] = ProgressLogger(console=True)  # The logger for training
    unique_id: Optional[Union[uuid4, str, int]] = None  # A unique identifier for the Trainer


class ModelConfig(BaseModel):
    name: MaybeStrList
    kernel_size: Optional[MaybeIntList]
    num_layers: Optional[MaybeIntList]
    latent_size: Optional[MaybeIntList]
    hidden_size: Optional[MaybeIntList]


class TestConfig(BasicConfig):
    model: MaybeStrList


class PathsInputConfig(BasicConfig):
    root: Path
    x_train: Path
    x_test: Path
    y_train: Path
    y_test: Path
    means: Optional[Path]
    stds: Optional[Path]
    latent: Optional[Path]


class PathsRawConfig(BasicConfig):
    root: Optional[Path] = None
    dataset: Optional[Path] = None
    dofs: Optional[Path] = None


class PathsOutputConfig(BasicConfig):
    root: Path
    figures: Optional[DirectoryPath]
    models: Optional[DirectoryPath]
    logs: Optional[DirectoryPath]
    results: Optional[DirectoryPath]
    parameters: Optional[DirectoryPath]


class PathsConfig(BasicConfig):
    input: PathsInputConfig
    raw: PathsRawConfig
    output: PathsOutputConfig


class ScenarioConfig(BasicConfig):
    class Config:
        extra = 'ignore'

    name: str
    model: ModelConfig
    training: TrainingConfig
    test: TestConfig
    paths: PathsConfig
    description: Optional[str]
    variable: Optional[str]
