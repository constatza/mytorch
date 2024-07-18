import importlib
from typing import Dict, Any

import numpy as np
import torch
from pydantic import FilePath
from pydantic import validate_call
from tomlkit import parse

from mytorch.io.config import (
    EstimatorsConfig,
    StudyConfig,
    PathsConfig,
    TrainingConfig,
)
from mytorch.io.utils import (
    replace_placeholders_in_toml,
)
from mytorch.mytypes import Maybe


@validate_call
def import_module(name: Maybe[str]) -> Any:
    """Use importlib to import the model class from the model's module.
    Name is separated by dots to indicate the module and the class.
    """
    if name is None:
        return None
    module_name, class_name = name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


@validate_call
def get_model_config(d: Dict) -> EstimatorsConfig:
    d["model"] = d["name"]
    return EstimatorsConfig(**d)


@validate_call
def get_paths_config(paths_dict: Dict) -> PathsConfig:
    return PathsConfig(**paths_dict)


@validate_call
def get_study_config(config: Dict[str, Any]) -> StudyConfig:
    study_config = config.get("study", None)
    model_config = config.get("model", None)
    training_config = config.get("training", None)
    paths_config = config.get("paths", None)

    paths_config = get_paths_config(paths_config)
    model = None
    if model_config is not None:
        model = get_model_config(model_config)

    training = None
    if training_config is not None:
        training = TrainingConfig(**training_config)

    return StudyConfig(
        name=study_config["name"],
        estimators=model,
        training=training,
        paths=paths_config,
    )


@validate_call
def read_toml(config_path: FilePath) -> Dict:
    """Reads a toml configuration file."""
    with open(config_path, "r") as file:
        content = file.read()
    replaced = replace_placeholders_in_toml(content)
    parsed = parse(replaced)
    return parsed


@validate_call
def read_study(config_path: FilePath) -> StudyConfig:
    config: Dict = read_toml(config_path)
    config: Dict = get_study_config(config)
    return config


@validate_call
def read_array_as_numpy(path: FilePath):
    if path.suffix == ".npy":
        return np.load(path)
    elif path.suffix == ".csv":
        return np.loadtxt(path, delimiter=",")
    elif path.suffix == ".pt":
        return torch.load(path).numpy()
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")
