import importlib
import uuid
from typing import Dict,  Any, Callable
from pydantic import FilePath
from pydantic import validate_call
from datetime import datetime

from mytorch.io.config import TrainingConfig, TestConfig, ModelConfig, PathsOutputConfig, PathsInputConfig, PathsRawConfig, PathsOutputConfig, ScenarioConfig
from mytorch.io.utils import read_toml, join_root_with_paths, apply_to_dict, replace_placeholders
from mytorch.dataloaders import create_dataloaders_from_path_config

type MaybeStr = str | None
type MaybeInt = int | None
type MaybeIntList = list[int] | int | None


@validate_call
def none_returns_none(func: Callable[[Any], Any]) -> Callable[[Any], Any]:
    def wrapper(arg: Any) -> Any | None:
        if arg is None:
            return None
        else:
            return func(arg)
    return wrapper

@validate_call
def rm_none_from_dict_output(func: Callable[[Any], Dict]) -> Callable:
    @validate_call
    def wrapper(d: Dict) -> Dict:
        d = func(d)
        return {k: v for k, v in d.items() if v is not None}
    return wrapper

@none_returns_none
def import_module(name: MaybeStr) -> Any:
    """ Use importlib to import the model class from the model's module.
    Name is separated by dots to indicate the module and the class.
    """
    module_name, class_name = name.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


@validate_call
def import_if_necessary(data: Dict) -> Dict:
    transformed = data.copy()
    source_module = {
        'optimizer': 'torch.optim.',
        'criterion': 'torch.nn.',
        'logger': 'mytorch.io.loggers.',
        'trainer': 'mytorch.trainers.',
        'model': 'mytorch.networks.',
    }
    for key, module in source_module.items():
        if key in transformed:
            transformed[key] = import_module(module + transformed[key])
    return transformed



@validate_call
def get_name(d: Dict) -> str:
    """ Get the name from the dictionary or return the default value
    of datetime.now() as str
    """
    return d.get('name', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

@validate_call
def get_training_config(d: Dict,
                        paths_input_config: PathsInputConfig,
                        convolution_dims: int) -> TrainingConfig:
    transformed = import_if_necessary(d)
    train_loader, test_loader = create_dataloaders_from_path_config(paths_input_config, convolution_dims)
    unique_id = uuid.uuid4()
    return TrainingConfig(**transformed,
                          train_loader=train_loader,
                          test_loader=test_loader,
                            unique_id=unique_id)

def get_test_config(d: Dict) -> TestConfig:
    return TestConfig(**d)

def get_model_config(d: Dict) -> ModelConfig:
    return ModelConfig(**import_if_necessary(d))

def get_paths_config(paths: Dict) -> PathsOutputConfig:
    paths = apply_to_dict(paths, replace_placeholders)
    paths = join_root_with_paths(paths)
    paths_input = get_paths_input_config(paths['input'])
    paths_raw = get_paths_raw_config(paths['raw'])
    paths_output = get_paths_output_config(paths['output'])
    return PathsOutputConfig(input=paths_input, raw=paths_raw, output=paths_output)

def get_paths_input_config(d: Dict) -> PathsInputConfig:
    return PathsInputConfig(**d)

def get_paths_raw_config(d: Dict) -> PathsRawConfig:
    return PathsRawConfig(**d)

def get_paths_output_config(d: Dict) -> PathsOutputConfig:
    return PathsOutputConfig(**d)

def get_scenario_config(d: Dict) -> ScenarioConfig:
    name = get_name(d['scenario'])
    description = d['scenario'].get('description', None)
    variable = d['scenario'].get('variable', None)
    paths = get_paths_config(d['paths'])
    model = get_model_config(d['model'])
    training = get_training_config(d['training'])
    test = get_test_config(d['test'])
    return ScenarioConfig(
        name=name,
        description=description,
        variable=variable,
        paths=paths,
        model=model,
        training=training,
        test=test
    )

@validate_call
def read_scenario(config_path: FilePath) -> ScenarioConfig:
    data = read_toml(config_path)
    return get_scenario_config(data)

if __name__=='__main__':
    from pathlib import Path
    cwd = Path('/home/archer/projects/mytorch/scenarios/bio-surrogate/config/')

    config_path = cwd  / 'u-cae.toml'
    scenario = read_scenario(config_path)








