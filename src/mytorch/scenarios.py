import os
from pathlib import Path
from collections.abc import Iterable
from itertools import product

from pydantic import BaseModel, validate_call
from typing import List, Dict, Any

from mytorch.io.config import ScenarioConfig, PathsOutputConfig, PathsOutputConfig




def create_combinations(parameters: Dict) -> Dict:
    new_keys = parameters.keys()
    values = list(parameters.values())
    values = [tuple(x) if isinstance(x, Iterable) else (x,) for x in values]
    combinations = tuple(tuple(x) for x in product(*values))
    return {i: dict(zip(new_keys, x)) for i, x in enumerate(combinations)}



def prepare_dirs(paths_config: PathsOutputConfig, delete_old: bool) -> None:
    if delete_old:
        delete_old_files(paths_config)
    else:
        decide_which_to_run(paths_config.parameters)

@validate_call
def delete_old_files(output_config: PathsOutputConfig) -> None:
    """Use pathlib to delete all files in the output directory."""
    for directory in output_config.model_dump().values():
        for file in directory.iterdir():
            file.unlink()

@validate_call
def decide_which_to_run(parameters_dir: Path) -> List[int]:
    experiments_ran = []
    if os.path.exists(parameters_dir):
        for file_name in os.listdir(parameters_dir):
            # get id from filename end
            experiment_id = int(file_name.split('.')[0].split('_')[-1])
            experiments_ran.append(int(experiment_id))
    return experiments_ran


class Scenario(BaseModel):
    """Creates all possible combinations of parameters for the experiment using cartesian product."""
    config: ScenarioConfig

    def run(self) -> None:
        logger = self.logger
        losses = []
        for model_id, model_parameters in self.model_combinations.items():
            for train_id, training_parameters in self.training_combinations.items():

                experiment_id = train_id + model_id * len(self.training_combinations)
                if experiment_id not in self.experiments_ran:
                    # separate stdout line
                    print('-' * 80)
                    print(f'Running experiment {experiment_id}/{self.num_experiments}')
                    model = model_parameters['model'](**self.model_shape_parameters, **model_parameters)
                    experiment = Experiment(model, dataloader_train=self.dataloader_train, dataloader_val=self.dataloader_val,
                                            optimizer=self.optimizer, criterion=self.criterion, uid=experiment_id,
                                            **training_parameters,
                                            checkpoint_dir=self.logger.paths_dict['output']['models'])
                    # search parameters if experiment has already been run

                    try:
                        train_loss, val_loss = experiment.train()
                        losses.append(val_loss[-1])
                        logger.log(f'Experiment {experiment_id:d} completed successfully.')
                        logger.write(f'{experiment.name}.pt', model, dirname='models')
                        parameters = {**model_parameters, **training_parameters, 'id': experiment_id}
                        logger.write(f'{experiment.name}.toml', parameters, dirname='parameters')
                        logger.write(f'losses', val_loss[-1])
                    except Exception as e:
                        raise e
                        logger.error(f'Experiment {experiment_id} failed with error: {e}')



