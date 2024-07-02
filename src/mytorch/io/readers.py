import importlib
import uuid
from builtins import dict
from typing import Dict, Any, Tuple

from pydantic import FilePath
from pydantic import validate_call
from tomlkit import parse

from mytorch.dataloaders import create_dataloaders_from_path_config
from mytorch.io.config import (
    TrainingConfig,
    EstimatorsConfig,
    PathsInputConfig,
    PathsRawConfig,
    PathsOutputConfig,
    StudyConfig,
    PathsConfig,
)
from mytorch.io.loggers import train_logger_factory
from mytorch.io.utils import join_root_with_paths, replace_placeholders_in_toml
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
def get_training_config(
    d: Dict,
    paths_input_config: PathsInputConfig,
    convolution_dims: int,
) -> TrainingConfig:
    train_loader, test_loader = create_dataloaders_from_path_config(
        paths_input_config.x_train,
        paths_input_config.x_test,
        paths_input_config.y_train,
        paths_input_config.y_test,
        convolution_dims,
    )
    unique_id = uuid.uuid4()
    return TrainingConfig(
        **d,
        train_loader=train_loader,
        test_loader=test_loader,
        unique_id=unique_id,
    )


@validate_call
def get_model_config(d: Dict, input_shape: Tuple) -> EstimatorsConfig:
    d["model"] = d["name"]
    return EstimatorsConfig(input_shape=input_shape, **d)


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
def get_study_config(config: Dict[str, Any]) -> StudyConfig:
    name = config["study"].get("name", None)
    description = config["study"].get("description", None)
    variable = config["study"].get("variable", None)
    paths = get_paths_config(config["paths"])
    training = get_training_config(
        config["training"],
        paths.input,
        config["model"]["convolution-dims"],
    )
    model = get_model_config(
        config["model"], tuple(training.train_loader.dataset[0][0].shape)
    )
    logger_type = config["study"]["logger"]
    logs_dir = paths.output.logs_dir
    logger = train_logger_factory(
        logger_type,
        history_fiel=logs_dir / f"{name}.log",
        error_file=logs_dir / f"{name}.err",
        console=True,
    )
    delete_old = config["study"]["delete-old"]
    if delete_old:
        delete_old = input(f"Are you sure you want to delete old study data? (y/N): ")
    if delete_old.lower() in ["y", "yes"]:
        delete_old = True
    else:
        delete_old = False

    return StudyConfig(
        name=name,
        estimators=model,
        training=training,
        description=description,
        variable=variable,
        paths=paths,
        logger=logger,
        delete_old=delete_old,
    )


@validate_call
def read_toml(config_path: FilePath) -> Dict:
    """Reads a toml configuration file."""
    with open(config_path, "r") as file:
        content = file.read()
    return dict(parse(replace_placeholders_in_toml(content)))


@validate_call
def read_study(config_path: FilePath) -> StudyConfig:
    config: Dict = read_toml(config_path)
    config: Dict = get_study_config(config)
    return config
