import importlib
from mytorch.utils.system import import_class


def initialize_datamodule(trial, datamodel_config):
    # Include hyperparameter suggestions
    datamodule_class = import_class(
        datamodel_config.pop("name"), prepend="mytorch.datamodules"
    )
    datamodule_instance = datamodule_class(**datamodel_config)
    return datamodule_instance
