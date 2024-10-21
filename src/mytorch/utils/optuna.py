import importlib
from collections.abc import Callable

import optuna
from optuna.pruners import BasePruner
from optuna.samplers import BaseSampler
from optuna.study import Study
from typing import Any, Optional
from pydantic import ValidationError
from mytorch.io.readers import read_study


def build_model(hparams: dict):
    """
    Dynamically imports and builds a model based on the 'model' field inside the hparams dictionary.
    The 'model' field should contain the module path of the model class (e.g., "my_project.models.MyModel").
    The remaining keys in hparams will be passed as initialization parameters to the model.

    Args:
        hparams (dict): A dictionary containing the 'model' key (the module path of the model class)
                        and other hyperparameters to initialize the model.

    Returns:
        model (Any): An instance of the model class initialized with the given hyperparameters.
    """
    # Extract the model class path from hparams
    model_name = hparams.pop("name")

    # Split the model_name into module and class
    module_name, class_name = model_name.rsplit(".", 1)

    # Dynamically import the module
    model_module = importlib.import_module(module_name)

    # Get the model class from the module
    model_class = getattr(model_module, class_name)

    # Initialize the model class with the remaining hyperparameters
    model = model_class(**hparams)

    return model


# Function to dynamically import and set up the optimizer
def setup_optimizer(model: Any, config: Any) -> Any:
    """
    Dynamically imports and sets up the optimizer for the model using the specified configuration.
    The optimizer is selected based on the 'name' provided in the configuration, and any additional
    optimizer-specific parameters (like learning rate, weight decay) are passed dynamically.

    Args:
        model (Any): The model whose parameters will be optimized.
        config (Any): The configuration object (Pydantic model) for the optimizer.

    Returns:
        Any: The instantiated optimizer object.
    """
    optimizer_class = getattr(importlib.import_module("torch.optim"), config.name)
    optimizer_params = {
        "params": model.parameters(),
        "lr": config.learning_rate,
        "weight_decay": config.weight_decay,
    }
    return optimizer_class(**optimizer_params, **config.dict(exclude_unset=True))


# Function to dynamically import and set up the scheduler
def setup_scheduler(optimizer: Any, config: Any) -> Optional[Any]:
    """
    Dynamically imports and sets up the learning rate scheduler for the optimizer based on the provided configuration.
    If no scheduler is specified (name is None), the function returns None.

    Args:
        optimizer (Any): The optimizer object to be used with the scheduler.
        config (Any): The configuration object (Pydantic model) for the scheduler.

    Returns:
        Optional[Any]: The instantiated scheduler object or None if no scheduler is specified.
    """
    if config.name is None:
        return None
    scheduler_class = getattr(
        importlib.import_module("torch.optim.lr_scheduler"), config.name
    )
    scheduler_params = {
        "optimizer": optimizer,
    }
    return scheduler_class(**scheduler_params, **config.dict(exclude_unset=True))


# Function to dynamically import and set up the pruner
def setup_pruner(config: dict) -> Optional[BasePruner]:
    """
    Dynamically imports and sets up the pruner for Optuna based on the provided configuration.
    If no pruner is specified (name is None), the function returns None.

    Args:
        config (Any): The configuration object (Pydantic model) for the pruner.

    Returns:
        Optional[BasePruner]: The instantiated pruner object or None if no pruner is specified.
    """
    name = config.pop("name")
    if name is None:
        return None
    pruner_class = getattr(importlib.import_module("optuna.pruners"), name)
    return pruner_class(**config)


# Function to dynamically import and set up the sampler
def setup_sampler(config: dict) -> Optional[BaseSampler]:
    """
    Dynamically imports and sets up the sampler for Optuna based on the provided configuration.
    If no sampler is specified (name is None), the function returns None.

    Args:
        config (Any): The configuration object (Pydantic model) for the sampler.

    Returns:
        Optional[BaseSampler]: The instantiated sampler object or None if no sampler is specified.
    """
    name = config.pop("name")
    if name is None:
        return None
    sampler_class = getattr(importlib.import_module("optuna.samplers"), name)
    return sampler_class(**config)


def suggest_hparams(trial, config: dict) -> dict:
    """
    Suggests hyperparameters based on the configuration and the Optuna trial object.
    The function loops over all tables (model, optimizer, etc.) in the configuration
    and processes each field. If the field is a literal, it passes it as-is, but if
    the field is a range or categorical definition (a Pydantic model field), it uses
    the appropriate Optuna suggestion method based on the 'type' key.

    Args:
        trial (optuna.Trial): The Optuna trial object that suggests hyperparameters.
        config (StudyConfig): The configuration object containing the model and training settings.

    Returns:
        dict: A dictionary of suggested hyperparameters.
    """

    # Helper function to process each field in the Pydantic model
    def suggest_value(field_name, field_value):
        # if literal value, return as-is
        if isinstance(field_value, (int, float, str, bool)):
            return field_value

        try:
            field_type = field_value["type"]
        except KeyError as ke:
            raise ValueError(f"Field type not found in {field_name}: {ke}")
        except TypeError as te:
            raise TypeError(f"Field {field_name} is not a dictionary: {te}")

        if field_type == "int":
            return trial.suggest_int(
                field_name, field_value["low"], field_value["high"]
            )

        if field_type == "float":
            return trial.suggest_float(
                field_name, field_value["low"], field_value["high"]
            )

        if field_type == "categorical":
            return trial.suggest_categorical(field_name, field_value["choices"])

    # Loop over all tables in the configuration
    suggested_params = {}
    for table_name, table_value in config.items():
        suggested_params[table_name] = {
            field_name: suggest_value(field_name, field_value)
            for field_name, field_value in table_value.items()
        }

    return suggested_params


def objective(trial, config: dict, training: Callable) -> float:
    """
    Defines the objective function for the Optuna study, which is called for each trial.
    This function initializes the model, sets up the optimizer and scheduler dynamically,
    and runs the training loop for hyperparameter optimization.

    Args:
        trial (optuna.Trial): The Optuna trial object, which suggests hyperparameters.
        config (StudyConfig): The full configuration object, including model, optimizer, scheduler, etc.

    Returns:
        Any: The evaluation metric (e.g., loss or accuracy) used to optimize the objective.
    """
    # Suggest hyperparameters using the Optuna configuration
    hparams = suggest_hparams(trial, config)

    # Build the model using the suggested hyperparameters
    loss, trainer = training(hparams)
    return loss


# Main function to run the Optuna study
def run_optuna_study(config_path: str, training: Callable):
    """
    Main function to run the Optuna study. Loads the configuration from TOML, sets up
    the pruner and sampler dynamically, and runs the study with the specified number of trials.

    Args:
        config_path (str): The path to the TOML configuration file.
    """
    try:
        config = read_study(config_path)

        # Set up the pruner and sampler
        pruner = setup_pruner(config["pruner"])
        sampler = setup_sampler(config["sampler"])

        # Build the Optuna study
        study: Study = optuna.create_study(
            direction=config["strategy"]["direction"], pruner=pruner, sampler=sampler
        )

        # Run the optimization
        study.optimize(
            lambda trial: objective(trial, config, training),  # Objective function
            n_trials=config["strategy"]["n_trials"],  # Number of trials
        )

    except ValidationError as e:
        print("Validation error in configuration:", e)
