import torch.nn as nn
from mytorch.utils.system import import_class


def initialize_model(model_config: dict, datamodule) -> nn.Module:
    """
    Dynamically imports and sets up the model based on the provided configuration.
    The configuration should include the name of the model as well as any parameters
    that need to be passed to the model's constructor.

    Args:
        trial (optuna.Trial): The Optuna trial object.
        model_config (dict): The configuration object for the model.

    Returns:
        nn.Module: The instantiated model object.
    """
    model_class = import_class(model_config.pop("name"), prepend="mytorch.networks")

    return model_class(
        **model_config,
        input_shape=datamodule.shapes[0],
        output_shapes=datamodule.shapes[1]
    )
