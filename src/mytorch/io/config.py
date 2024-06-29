from pathlib import Path
from typing import Optional, Union, List, Callable

import torch
from pydantic import (
    BaseModel,
    DirectoryPath,
    FilePath,
    Field,
    field_validator,
    validate_call,
)

from mytorch.io.loggers import ProgressLogger, TrainLogger

MaybeIntList = Union[int, List[int]]
MaybeFloatList = Union[float, List[float]]
MaybeStrList = Union[str, List[str]]


class BasicConfig(BaseModel):
    class Config:
        populate_by_name = True
        allow_population_by_alias = True
        arbitrary_types_allowed = True
        frozen = True

        @classmethod
        def alias_generator(cls, field_name: str) -> str:
            return field_name.lower().strip().replace("_", "-")


@validate_call
def ensure_dir_exists(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


class ModelConfig(BasicConfig):
    model: type(torch.nn.Module)  # The model to be trained
    kernel_size: Optional[MaybeIntList] = None
    num_layers: Optional[MaybeIntList] = None
    latent_size: Optional[MaybeIntList] = None
    hidden_size: Optional[MaybeIntList] = None


class TestConfig(BasicConfig):
    model: MaybeStrList


class PathsInputConfig(BasicConfig):
    root_dir: DirectoryPath
    x_train: FilePath
    y_train: FilePath
    x_test: FilePath
    y_test: FilePath
    means: Optional[FilePath]
    stds: Optional[FilePath]
    dataset: Optional[FilePath]
    latent: Optional[FilePath]

    _dir_validator = field_validator("root_dir", mode="before")(ensure_dir_exists)


class PathsRawConfig(BasicConfig):
    root_dir: Optional[DirectoryPath] = None
    dataset: Optional[Path] = None
    dofs: Optional[Path] = None

    _dir_validator = field_validator("root_dir", mode="before")(ensure_dir_exists)


class PathsOutputConfig(BasicConfig):
    root_dir: DirectoryPath
    figures_dir: Optional[DirectoryPath]
    models_dir: Optional[DirectoryPath]
    logs_dir: Optional[DirectoryPath]
    results_dir: Optional[DirectoryPath]
    parameters_dir: Optional[DirectoryPath]

    _dir_validator = field_validator(
        "root_dir",
        "figures_dir",
        "models_dir",
        "logs_dir",
        "results_dir",
        "parameters_dir",
        mode="before",
    )(ensure_dir_exists)


class PathsConfig(BasicConfig):
    input: PathsInputConfig
    raw: PathsRawConfig
    output: PathsOutputConfig


class TrainingConfig(BasicConfig):
    train_loader: torch.utils.data.DataLoader  # The DataLoader for the training data
    test_loader: torch.utils.data.DataLoader  # The DataLoader for the test data
    num_epochs: MaybeIntList  # The number of epochs for training
    batch_size: MaybeIntList  # The batch size for training
    optimizer: Optional[Callable[..., torch.optim.Optimizer]] = torch.optim.Adam
    criterion: Optional[Callable[..., torch.nn.modules.loss._Loss]] = (
        torch.nn.MSELoss
    )  # The loss function
    learning_rate: Optional[MaybeFloatList] = Field(
        default=1e-3, alias="lr"
    )  # The learning rate for training
    device: Optional[torch.device] = (
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # The device to train on (e.g., 'cpu' or 'cuda')
    logger: Optional[TrainLogger] = ProgressLogger(
        console=True
    )  # The logger for training


class ScenarioConfig(BasicConfig):
    class Config:
        extra = "ignore"

    name: str
    model: ModelConfig
    training: TrainingConfig
    test: TestConfig
    paths: PathsConfig
    description: Optional[str]
    variable: Optional[str]
