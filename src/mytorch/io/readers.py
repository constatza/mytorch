import sys
import importlib
from typing import Dict, Union, Any, Callable
from pydantic import BaseModel, FilePath, DirectoryPath
from pydantic import validate_call

from mytorch.io.config import TrainingConfig, TestConfig, ModelConfig, PathsConfig, PathsInputConfig, PathsRawConfig, PathsOutputConfig, ScenarioConfig
from mytorch.io.utils import read_toml,  join_root_with_paths, apply_to_dict, replace_placeholders, smart_load_tensors, get_proper_convolution_shape
from mytorch.dataloaders import create_dataloaders

import sys
from pathlib import Path

# Get the parent directory


type MaybeStr = str | None
type MaybeInt = int | None
type MaybeIntList = list[int] | int | None


sys.path.insert(0, '../..')

def none_returns_none(func):
    def wrapper(arg):
        if arg is None:
            return None
        else:
            return func(arg)
    return wrapper

def rm_none_from_dict_output(func: Callable[[Any], Dict]) -> Callable:
    def wrapper(d):
        d = func(d)
        return {k: v for k, v in d.items() if v is not None}
    return wrapper

@none_returns_none
def import_module(name: MaybeStr) -> Any:
    """ Use importlib to import the model class from the models module.
    Name is separated by dots to indicate the module and the class.
    """
    module_name, class_name = name.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


class TrainingConfigReader(BaseModel):

    @staticmethod
    def transform(data: Dict):
        """ get the training transformed from the data, trasnform it and return
        the TrainingConfig object.
        """
        transformed = import_if_necessary(data['training'])
        return TrainingConfig(**transformed)

class PathsInputConfigReader(BaseModel):


    @staticmethod
    def read(paths_dict: Dict):
        """ get the input paths from the paths_dict and return the PathsInputConfig object.
        """
        x_train = paths_dict['input']['x-train']
        x_test = paths_dict['input']['x-test']
        y_train = paths_dict['input']['y-train']
        y_test = paths_dict['input']['y-test']
        means = paths_dict['input'].get('means', None)
        stds = paths_dict['input'].get('stds', None)
        latent = paths_dict['input'].get('latent', None)
        return PathsInputConfig(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test,
                                means=means, stds=stds, latent=latent)




class ScenarioReader(BaseModel):
    @staticmethod
    def get_dataloaders(config: Dict):
        """ get the train_loader and test_loader from the data and return them.
        """
        convolution_dims = config['scenario']['convolution_dims']

        paths_input_config = PathsInputConfigReader.read(config['paths'])
        x_train = paths_input_config.x_train
        x_test = paths_input_config.x_test
        y_train = paths_input_config.y_train
        y_test = paths_input_config.y_test

        x_train = smart_load_tensors(x_train, convolution_dims)
        x_test = smart_load_tensors(x_test, convolution_dims)

        if x_train != y_train and x_test != y_test:
            y_train = smart_load_tensors(y_train, convolution_dims)
            y_test = smart_load_tensors(y_test, convolution_dims)
        else:
            y_train = None
            y_test = None

        dataloader_train, dataloader_val = create_dataloaders(x_train, x_test, y_train, y_test)
        return dataloader_train, dataloader_val


    @staticmethod
    def create(name,
               model_config: ModelConfig,
               training_config: TrainingConfig,
               test_config: TestConfig,
               paths_config: PathsConfig,
               description: str = None,
               variable: str = None):

        # Old code in case is needed
        # self.is_autoencoder = self.dataloader_val.dataset.is_autoencoder
        # self.x_shape = self.dataloader_train.dataset.x_shape
        # self.y_shape = self.dataloader_train.dataset.y_shape
        # self.input_shape = get_proper_convolution_shape(self.x_shape, self.convolution_dims)
        # self.output_shape = get_proper_convolution_shape(self.y_shape, self.convolution_dims)

        return ScenarioConfig(name=name,
                                model=model_config,
                                training=training_config,
                                test=test_config,
                                paths=paths_config,
                                description=description,
                                variable=variable)



def import_if_necessary(self, data: Dict):
    transformed = data.copy()
    source_module = {
        'optimizer': 'torch.optim.',
        'criterion': 'torch.nn.',
        'logger': 'mytorch.io.loggers.',
        'trainer': 'mytorch.trainers.',
    }
    for key, module in source_module.items():
        if key in transformed:
            transformed[key] = import_module(module + transformed[key])
    return transformed

class ScenarioConfigReader(BaseModel):
    class Config:
        extra = 'allow'

    @validate_call
    def from_toml(self, config_file: FilePath):
        data = read_toml(config_file)
        return self.from_dict(data)

    @validate_call
    def from_dict(self, data: Dict):
        paths = data['paths']
        paths = apply_to_dict(paths, replace_placeholders)
        paths = join_root_with_paths(paths)
        data['paths'] = paths
        self.data = data
        return self




    def get_test_config(self):
        transformed = self.data['test'].copy()
        return TestConfig(**transformed)

    def get_model_config(self):
        transformed = self.data['model'].copy()
        transformed['name'] = get_model(transformed.get('name', None))
        return ModelConfig(**transformed)

    def get_paths_config(self):
        return PathsConfig(**self.data['paths'])

    def get_paths_input_config(self):
        return PathsInputConfig(**self.data['paths']['input'])

    def get_paths_raw_config(self):
        return PathsRawConfig(**self.data['paths']['raw'])

    def get_paths_output_config(self):
        return PathsOutputConfig(**self.data['paths']['output'])

    def get_scenario_config(self):
        training = self.get_training_config()
        test = self.get_test_config()
        model = self.get_model_config()
        input_config = self.get_paths_input_config()
        raw_config = self.get_paths_raw_config()
        output_config = self.get_paths_output_config()
        paths_config = PathsConfig(input=input_config, raw=raw_config, output=output_config)

        training_config = TrainingConfig(**self.data['training'])
        test_config = TestConfig(**self.data['test'])

        return ScenarioConfig(**self.data['scenario'],
                              training=training,
                              test=test,
                              model=model,
                              paths=paths)


if __name__=='__main__':
    from pathlib import Path
    cwd = Path('/home/archer/projects/mytorch/scenarios/bio-surrogate/config/')

    config_path = cwd  / 'u-cae.toml'
    reader = ScenarioConfigReader()
    reader.from_toml(config_path)
    reader.get_scenario_config()
    print(reader.data)








