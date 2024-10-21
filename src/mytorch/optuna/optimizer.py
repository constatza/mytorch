import torch.optim as optim


def initialize_optimizer(trial, config, model):
    optimizer_config = config.get("optimizer", {})
    optimizer_name = optimizer_config.get("name", "Adam")
    optimizer_class = getattr(optim, optimizer_name, optim.Adam)
    optimizer_params = optimizer_config.get("params", {})

    # Include hyperparameter suggestions
    for param_name, param_range in optimizer_params.items():
        if isinstance(param_range, dict) and "min" in param_range:
            optimizer_params[param_name] = trial.suggest_float(
                param_name,
                param_range["min"],
                param_range["max"],
                step=param_range.get("step", None),
                log=param_range.get("log", False),
            )
    optimizer = optimizer_class(model.parameters(), **optimizer_params)
    return optimizer
