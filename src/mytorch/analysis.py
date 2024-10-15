from pathlib import Path
from typing import Dict, Tuple, List

import optuna
import torch
from optuna.storages import RetryFailedTrialCallback

from mytorch.io.config import (
    StudyConfig,
    EstimatorsConfig,
    TrainingConfig,
)
from mytorch.io.readers import import_module


torch.manual_seed(0)


class StudyRunner:
    """Uses optuna to run a study with a given configuration, creates all possible combinations of model
    and training parameters, and runs the experiments that have not been run before."""

    def __init__(self, config: StudyConfig):
        self.config = config

    def run(self) -> None:
        optuna.logging.enable_propagation()
        study_name = self.config.name
        storage_name = f"sqlite:///{study_name}.db"
        sampler = optuna.samplers.TPESampler()

        if self.config.delete_old:
            optuna.delete_study(study_name=study_name, storage=storage_name)

        storage = optuna.storages.RDBStorage(
            url=storage_name,
            heartbeat_interval=10,
            failed_trial_callback=RetryFailedTrialCallback(),
        )

        study = optuna.create_study(
            storage=storage,
            study_name=study_name,
            direction="minimize",
            load_if_exists=True,
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=2, n_warmup_steps=10, interval_steps=5
            ),
            sampler=sampler,
        )
        study.optimize(
            func=lambda trial: objective(trial, self.config),
            n_trials=self.config.num_trials,
        )

        pruned_trials = study.get_trials(states=(optuna.trial.TrialState.PRUNED,))
        complete_trials = study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        # The line of the resumed trial's intermediate values begins with the restarted epoch.
        optuna.visualization.plot_intermediate_values(study).show()
        # visualize parameter space with parallel coordinates
        optuna.visualization.plot_parallel_coordinate(study).show()

        # dump the best trial as a config
        best_trial = study.best_trial
        best_trial_config = study.trials_dataframe().iloc[best_trial.number]
        output_dir = self.config.paths.output.parameters_dir
        best_trial_config.to_csv(output_dir / "{study_name}_best_trial.csv")


def get_suggestions(trial, config) -> Dict:
    # run through all attributes of the estimator config
    # if the attribute is a list, then we need to get the value from the trial
    # if the attribute is a single value, then we just use that value
    # return the model initialized with the values

    # for each attribute in the estimator config
    suggestions = {}
    frozen = ("input_shape", "output_shape", "device")

    importable = {
        "model": "mytorch.networks",
        "optimizer": "torch.optim",
        "criterion": "torch.nn",
    }
    skip = ("logger",)

    for attr in config.__dict__.keys():
        # if the attribute is a list
        value = getattr(config, attr)
        if attr in frozen or not isinstance(value, (list, tuple)):
            # use the value
            suggestions[attr] = value
        elif attr not in skip:
            match value:
                case (low, high, option):
                    match (low, high, option):
                        case (int(), int(), int()):
                            suggestions[attr] = trial.suggest_int(
                                attr, low, high, step=option
                            )
                        case (float(), float(), bool()):
                            suggestions[attr] = trial.suggest_float(
                                attr, low, high, log=option
                            )
                        case _:
                            suggestions[attr] = trial.suggest_categorical(
                                attr, [low, high, option]
                            )

                case (low, high):
                    match (low, high):
                        case (int(), int()):
                            suggestions[attr] = trial.suggest_int(attr, low, high)
                        case (float(), float()):
                            suggestions[attr] = trial.suggest_float(attr, low, high)
                case (single_value,):
                    suggestions[attr] = single_value
                case options if isinstance(options, (List, Tuple)):
                    match options:
                        case strings if all(isinstance(s, str) for s in strings):
                            suggestions[attr] = trial.suggest_categorical(attr, strings)
                        case _:
                            pass
                case single_value if isinstance(single_value, (int, float, bool, str)):
                    suggestions[attr] = single_value
                case _:
                    raise ValueError(f"Invalid type for {attr}: {type(value)}")

            if attr in importable.keys():

                suggestions[attr] = import_module(
                    f"{importable[attr]}.{suggestions[attr]}"
                )

    return suggestions


def define_trainer(
    trial: optuna.Trial,
    training_config: TrainingConfig,
    model: torch.nn.Module,
    logger,
    models_dir: Path,
    delete_old: bool,
) -> Trainer:
    # Define trainer
    # get suggestions from the trial
    suggestions = get_suggestions(trial, training_config)
    filtered = filtered_dict(suggestions, Trainer.__init__.__code__.co_varnames)

    # return the trainer
    return Trainer(
        **filtered,
        trial=trial,
        model=model,
        logger=logger,
        models_dir=models_dir,
        delete_old=delete_old,
        kld_weight=trial.suggest_float("kld_weight", 1e-5, 1e-3, log=True),
    )


def define_estimator(trial, estimator_config: EstimatorsConfig) -> torch.nn.Module:
    # Define estimator
    # get suggestions from the trial
    suggestions = get_suggestions(trial, estimator_config)
    model = suggestions["model"]
    # accept only the keys that are in the model
    acceptable = filtered_dict(suggestions, model.__init__.__code__.co_varnames)
    # return the estimator
    return model(**acceptable)


def filtered_dict(d: Dict, keys: Tuple) -> Dict:
    return {k: v for k, v in d.items() if k in keys}


def objective(trial, study_config: StudyConfig):
    # Generate the model.
    estimator = define_estimator(trial, study_config.estimators)

    trainer = define_trainer(
        trial,
        study_config.training,
        estimator,
        study_config.logger,
        study_config.paths.output.models_dir,
        delete_old=study_config.delete_old,
    )

    # Training of the model.

    # return the loss
    train_loss, val_loss = trainer.train()
    loss = val_loss[-1]

    # Save optimization status. We should save the objective value because the process may be
    # killed between saving the last model and recording the objective value to the storage.

    return loss


if __name__ == "__main__":
    from mytorch.io.readers import read_study

    config = read_study(
        r"C:\Users\cluster\constantinos\mytorch\studies\bio-surrogate\config\u-cae.toml"
    )
    study = StudyRunner(config=config)

    study.run()
