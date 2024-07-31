from pathlib import Path
from typing import Annotated, Optional

import torch
from pydantic import (
    BaseModel,
    Field,
    field_validator,
    ConfigDict,
    model_validator,
    DirectoryPath,
)
from pydantic.alias_generators import to_snake

from mytorch.mytypes import (
    TupleLike,
    Maybe,
)


class BasicConfig(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        populate_by_name=True,
        frozen=True,
        extra="allow",
        alias_generator=lambda x: to_snake(x).replace("_", "-"),
    )


class PathsConfig(BasicConfig):

    @model_validator(mode="before")
    def validate_paths(cls, data):
        for key, path in data.items():
            # create dirs if they don't exist
            if path is not None:
                path = Path(path)
                path.parent.mkdir(parents=True, exist_ok=True)
                data[key] = path
        return data

    output: Optional[Path] = None
    input: Optional[Path] = None
    raw: Optional[Path] = None
    processed: Optional[Path] = None
    input_data: Optional[Path] = None
    output_data: Optional[Path] = None
    checkpoint: Optional[DirectoryPath] = None
    model: Optional[Path] = None
    dataset: Optional[Path] = None


class EstimatorsConfig(BasicConfig):
    model: TupleLike[str]
    input_shape: Annotated[
        TupleLike, Field(min_length=1, max_length=4)
    ]  # The shape of the input data
    convolution_dims: Annotated[int, Field(default=0, ge=0, le=2)]
    kernel_size: Optional[TupleLike[int]] = None
    num_layers: Optional[TupleLike[int]] = None
    latent_size: Optional[TupleLike[int]] = None
    hidden_size: Optional[TupleLike[int]] = None


class TrainingConfig(BasicConfig):
    num_epochs: TupleLike[int] = 100  # The number of epochs for training
    batch_size: TupleLike[int] = 32  # The batch size for training
    learning_rate: TupleLike[float] = 1e-3
    device: Optional[torch.device] = (
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # The device to train on (e.g., 'cpu' or 'cuda')


class StudyConfig(BasicConfig):
    class Config:
        extra = "ignore"

    paths: PathsConfig
    name: Optional[str] = None
    training: Optional[TrainingConfig]
    estimators: Optional[EstimatorsConfig]
    description: Optional[str] = ""
    variable: Optional[str] = "x"
    delete_old: Optional[bool] = True
    num_trials: Optional[int] = 1

    @field_validator("name", mode="after")
    def validate_name(cls, v: Maybe[str]) -> str:
        """Get the name from the dictionary or return the default value
        of datetime.now() as str
        """
        from datetime import datetime

        return v if v else datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
