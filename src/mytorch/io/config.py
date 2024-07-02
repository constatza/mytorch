from pathlib import Path
from typing import Annotated, Optional

import torch
from optuna import Trial
from pydantic import (
    BaseModel,
    FilePath,
    Field,
    field_validator,
)

from mytorch.io.loggers import TrainLogger
from mytorch.mytypes import (
    ListLike,
    CreateIfNotExistsDir,
    Maybe,
)


class BasicConfig(BaseModel):
    class Config:
        populate_by_name = True
        allow_population_by_alias = True
        arbitrary_types_allowed = True
        frozen = True

        @classmethod
        def alias_generator(cls, field_name: str) -> str:
            return field_name.lower().strip().replace("_", "-")


class PathsInputConfig(BasicConfig):
    root_dir: CreateIfNotExistsDir
    x_train: FilePath
    y_train: FilePath
    x_test: FilePath
    y_test: FilePath
    means: Optional[FilePath]
    stds: Optional[FilePath]
    dataset: Optional[FilePath]
    latent: Optional[FilePath]


class PathsRawConfig(BasicConfig):
    class Config:
        extra = "allow"

    root_dir: Optional[CreateIfNotExistsDir] = None
    dataset: Optional[Path] = None
    dofs: Optional[Path] = None


class PathsOutputConfig(BasicConfig):
    root_dir: CreateIfNotExistsDir
    figures_dir: Optional[CreateIfNotExistsDir]
    models_dir: Optional[CreateIfNotExistsDir]
    logs_dir: Optional[CreateIfNotExistsDir]
    results_dir: Optional[CreateIfNotExistsDir]
    parameters_dir: Optional[CreateIfNotExistsDir]


class PathsConfig(BasicConfig):
    input: PathsInputConfig
    raw: PathsRawConfig
    output: PathsOutputConfig


class EstimatorsConfig(BasicConfig):
    model: ListLike[str]
    input_shape: Annotated[
        ListLike, Field(min_length=1, max_length=4)
    ]  # The shape of the input data
    convolution_dims: Annotated[int, Field(..., ge=0, le=2)]
    kernel_size: Optional[ListLike[int]] = None
    num_layers: Optional[ListLike[int]] = None
    latent_size: Optional[ListLike[int]] = None
    hidden_size: Optional[ListLike[int]] = None


class TrainingConfig(BasicConfig):
    train_loader: torch.utils.data.DataLoader  # The DataLoader for the training data
    test_loader: torch.utils.data.DataLoader  # The DataLoader for the test data
    num_epochs: ListLike[int] = 100  # The number of epochs for training
    batch_size: ListLike[int] = 32  # The batch size for training
    optimizer: ListLike[str] = "Adam"
    criterion: ListLike[str] = "MSELoss"
    learning_rate: ListLike[float] = 1e-3
    device: Optional[torch.device] = (
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # The device to train on (e.g., 'cpu' or 'cuda')


class StudyConfig(BasicConfig):
    class Config:
        extra = "ignore"

    name: Optional[str] = None
    training: TrainingConfig
    estimators: EstimatorsConfig
    paths: PathsConfig
    description: Optional[str]
    variable: Optional[str]
    logger: Optional[TrainLogger] = "progress"
    delete_old: Optional[bool]

    @field_validator("name", mode="after")
    def validate_name(cls, v: Maybe[str]) -> str:
        """Get the name from the dictionary or return the default value
        of datetime.now() as str
        """
        from datetime import datetime

        return v if v else datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


class TrainerConfig(BasicConfig):
    trial: Trial
    model: torch.nn.Module
    train_loader: torch.utils.data.DataLoader
    test_loader: torch.utils.data.DataLoader
    criterion: torch.nn.modules.loss._Loss
    optimizer: type(torch.optim.Optimizer)
    device: torch.device
    learning_rate: float
    logger: TrainLogger
    models_dir: Path
    num_epochs: int
    delete_old: bool
