from torch import optim


def initialize_scheduler(trial, config, optimizer):
    scheduler = None
    if "scheduler" in config:
        scheduler_config = config["scheduler"]
        scheduler_name = scheduler_config.get("name")
        scheduler_class = getattr(optim.lr_scheduler, scheduler_name, None)
        if scheduler_class is None:
            raise ValueError(
                f"Scheduler '{scheduler_name}' not found in torch.optim.lr_scheduler."
            )
        scheduler_params = scheduler_config.get("params", {})

        # Include hyperparameter suggestions
        for param_name, param_range in scheduler_params.items():
            if isinstance(param_range, dict) and "min" in param_range:
                suggestion_name = f"scheduler_{param_name}"
                scheduler_params[param_name] = trial.suggest_float(
                    suggestion_name,
                    param_range["min"],
                    param_range["max"],
                    step=param_range.get("step", None),
                    log=param_range.get("log", False),
                )

        scheduler_instance = scheduler_class(optimizer, **scheduler_params)
        scheduler = {
            "scheduler": scheduler_instance,
            "interval": scheduler_config.get("interval", "epoch"),
            "frequency": scheduler_config.get("frequency", 1),
        }
    return scheduler
