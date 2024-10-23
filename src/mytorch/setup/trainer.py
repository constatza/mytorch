from lightning import Trainer
from mytorch.utils.system import filter_kwargs
from pathlib import Path


def initialize_trainer(config):
    output = Path(config.get("paths", ".").get("output", "output"))
    output.mkdir(exist_ok=True, parents=True)
    total_params = {**config.get("trainer", {}), "default_root_dir": output}
    trainer = filter_kwargs(Trainer, total_params)
    return trainer
