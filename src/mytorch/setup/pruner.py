from mytorch.utils.system import import_dynamically


def initialize_pruner(pruner_config):
    if not pruner_config:
        pruner_config = {"name": "NopPruner"}

    pruner_name = pruner_config.pop("name", "NopPruner")
    pruner_class = import_dynamically(pruner_name, prepend="optuna.pruners")
    return pruner_class(**pruner_config)
