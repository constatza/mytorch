import torch.nn as nn
from mytorch.utils.system_utils import import_dynamically
from mytorch.utils.system_utils import filter_kwargs


def initialize_model(config: dict, shapes: dict) -> nn.Module:
    """
    Dynamically imports and sets up the model based on the provided configuration.
    The configuration should include the name of the model as well as any parameters
    that need to be passed to the model's constructor.

    Args:
        model_config (dict): The configuration object for the model.
        shapes (dict): A dictionary containing the shapes of the features and targets.

    Returns:
        nn.Module: The instantiated model object.
    """
    model_config = config["model"]
    model_class = import_dynamically(
        model_config.get("name"), prepend="mytorch.networks"
    )
    model_config.update(
        {
            "input_shape": shapes["features"],
            "output_shape": shapes.get("targets", None),
            "optimizer_config": config["optimizer"],
            "scheduler_config": config["scheduler"],
        }
    )
    model = model_class(**filter_kwargs(model_config))
    return model
