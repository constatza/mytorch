import torch.nn as nn
from mytorch.utils.system import import_dynamically
from mytorch.utils.system import filter_kwargs


def initialize_model(config: dict, shapes: dict) -> nn.Module:
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
    model_config = config["model"]
    model_class = import_dynamically(
        model_config.get("name"), prepend="mytorch.networks"
    )
    model_config = {
        **model_config,
        **shapes,
        "input_shape": shapes["features"],
        "output_shape": shapes["targets"],
        "optimizer_config": config["optimizer"],
        "scheduler_config": config["scheduler"],
    }
    model = filter_kwargs(model_class, model_config)

    return model
