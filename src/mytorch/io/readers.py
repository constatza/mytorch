import importlib
import uuid
from builtins import dict
from datetime import datetime
from typing import Dict, Any

from pydantic import FilePath
from pydantic import validate_call
from tomlkit import parse

from mytorch.dataloaders import create_dataloaders_from_path_config
from mytorch.io.config import (
    TrainingConfig,
    TestConfig,
    ModelConfig,
    PathsInputConfig,
    PathsRawConfig,
    PathsOutputConfig,
    ScenarioConfig,
    PathsConfig,
)
from mytorch.io.utils import join_root_with_paths, replace_placeholders_in_toml

type MaybeStr = str | None
type MaybeInt = int | None
type MaybeIntList = list[int] | int | None


@validate_call
def import_module(name: MaybeStr) -> Any:
    """Use importlib to import the model class from the model's module.
    Name is separated by dots to indicate the module and the class.
    """
    module_name, class_name = name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


@validate_call
def import_if_necessary(data: Dict) -> Dict:
    transformed = data.copy()
    source_module = {
        "optimizer": "torch.optim.",
        "criterion": "torch.nn.",
        "logger": "mytorch.io.loggers.",
        "trainer": "mytorch.trainers.",
        "model": "mytorch.networks.",
        "name": "mytorch.networks.",
    }
    for key, module in source_module.items():
        if key in transformed:
            transformed[key] = import_module(module + transformed[key])
    return transformed


@validate_call
def get_name(d: Dict) -> str:
    """Get the name from the dictionary or return the default value
    of datetime.now() as str
    """
    return d.get("name", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))


@validate_call
def get_training_config(
    d: Dict, paths_input_config: PathsInputConfig, convolution_dims: int
) -> TrainingConfig:
    transformed = import_if_necessary(d)
    train_loader, test_loader = create_dataloaders_from_path_config(
        paths_input_config, convolution_dims
    )
    unique_id = uuid.uuid4()
    return TrainingConfig(
        **transformed,
        train_loader=train_loader,
        test_loader=test_loader,
        unique_id=unique_id,
    )


@validate_call
def get_test_config(d: Dict) -> TestConfig:
    return TestConfig(**d)


@validate_call
def get_model_config(d: Dict) -> ModelConfig:
    d = import_if_necessary(d)
    d["model"] = d["name"]
    return ModelConfig(**d)


@validate_call
def get_paths_config(paths: Dict) -> PathsConfig:
    paths = join_root_with_paths(paths)
    paths_input = get_paths_input_config(paths["input"])
    paths_raw = get_paths_raw_config(paths["raw"])
    paths_output = get_paths_output_config(paths["output"])
    return PathsConfig(input=paths_input, raw=paths_raw, output=paths_output)


@validate_call
def get_paths_input_config(d: Dict) -> PathsInputConfig:
    return PathsInputConfig(**d)


@validate_call
def get_paths_raw_config(d: Dict) -> PathsRawConfig:
    return PathsRawConfig(**d)


@validate_call
def get_paths_output_config(d: Dict) -> PathsOutputConfig:
    return PathsOutputConfig(**d)


@validate_call
def get_scenario_config(config: Dict[str, Any]) -> ScenarioConfig:
    name = get_name(config["scenario"])
    description = config["scenario"].get("description", None)
    variable = config["scenario"].get("variable", None)
    paths = get_paths_config(config["paths"])
    model = get_model_config(config["model"])
    training = get_training_config(
        config["training"], paths.input, config["scenario"]["convolution-dims"]
    )
    test = get_test_config(config["test"])
    return ScenarioConfig(
        name=name,
        description=description,
        variable=variable,
        paths=paths,
        model=model,
        training=training,
        test=test,
    )


@validate_call
def read_toml(config_path: FilePath) -> Dict:
    """Reads a toml configuration file."""
    with open(config_path, "r") as file:
        content = file.read()
    return dict(parse(replace_placeholders_in_toml(content)))


@validate_call
def read_scenario(config_path: FilePath) -> ScenarioConfig:
    config: Dict = read_toml(config_path)
    config: Dict = get_scenario_config(config)
    return config
