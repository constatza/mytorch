import importlib
import os
from typing import Optional, Union, Dict
from collections.abc import Iterable
from itertools import product

import numpy as np
import torch
from utils import smart_load_tensors, get_proper_convolution_shape

from datasets import create_dataloaders
from utils import filtered


@filtered
class Experiment:
    def __init__(self, model=None, uid=None, dataloader_train=None, dataloader_val=None,
                 optimizer=None, criterion=None, lr=None, batch_size=None, num_epochs=None, checkpoint_dir=None,
                 epoch_print_interval=10):
        self.name = model.__class__.__name__ + '_' + str(uid)
        self.model = model
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val
        self.id = uid
        self.optimizer = optimizer(self.model.parameters(), lr=lr)
        self.criterion = criterion
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.checkpoint_relative_tol = 0.5
        self.checkpoint_path = os.path.join(checkpoint_dir, self.name + '.pt')
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.epoch_print_interval = epoch_print_interval
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using {self.device} for training.')

    def run(self, *args, **kwargs):
        return self.train(*args, **kwargs)


    def train(self) -> tuple:
        """Training loop for the model with both training and validation loss."""
        # use the GPU if available
        optimizer = self.optimizer
        criterion = self.criterion
        device = self.device

        # Initialize lists to store training and validation losses
        train_losses = []
        val_losses = []
        min_val_loss = float('inf')
        self.model.to(device)
        for epoch in range(self.num_epochs):
            # Training
            self.model.train()
            train_loss = 0
            for x_batch, y_batch in self.dataloader_train:

                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()
                y_pred = self.model(x_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * x_batch.size(0)
            train_losses.append(train_loss / len(self.dataloader_train))

            # Validation
            val_loss = 0
            self.model.eval()
            with torch.no_grad():
                for x_batch, y_batch in self.dataloader_val:
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
                    loss_val = criterion(self.model(x_batch), y_batch)
                    val_loss += loss_val.item() * x_batch.size(0)
                val_losses.append(val_loss / len(self.dataloader_val))

            self.print_losses(train_losses[-1], val_losses[-1], epoch)


            # Check if the validation loss has decreased by more than the tolerance
            if relative_tolerance(min_val_loss, val_losses) > self.checkpoint_relative_tol:
                print('Checkpointing model...')
                min_val_loss = val_losses[-1]
                # Save the model state
                best_model = torch.jit.script(self.model)
                # Save the model state to a file
                best_model.save(self.checkpoint_path)

        return train_losses, val_losses


    def print_losses(self, train_loss, val_loss, epoch) -> None:
        epoch_for_print = epoch + 1
        if epoch_for_print % self.epoch_print_interval == 0:
            text = f'Epoch {epoch_for_print}/{self.num_epochs} | Train Loss: {train_loss:.5e} | Val Loss: {val_loss:.5e}'
            print(text)




def relative_tolerance(min_loss, losses):
    if len(losses) > 5:
        mean = np.mean(losses[-5:])
    else:
        mean = np.mean(losses)
    return (min_loss - mean) / mean




class Parameters:

    @staticmethod
    def create_combinations(parameters: dict) -> tuple([dict]):
        new_keys = Parameters.substitute_keys(parameters.keys())
        values = list(parameters.values())
        values = [tuple(x) if isinstance(x, Iterable) else (x,) for x in values]
        combinations = tuple(tuple(x) for x in product(*values))
        return {i: dict(zip(new_keys, x)) for i, x in enumerate(combinations)}

    @staticmethod
    def substitute_keys(old_keys) -> tuple:
        key_mapping = {'learning-rate': 'lr'}
        new_keys = []
        for key in old_keys:
            new_key = key_mapping.get(key, key).replace('-', '_')
            new_keys.append(new_key)
        return new_keys




class AnalysisLoader:

    def __init__(self, paths_dict: Dict, delete_old: bool = False, convolution_dims: int = 0) -> None:
        self.paths_dict = paths_dict
        self.convolution_dims = convolution_dims
        self.load_networks()
        self.load_data()
        joined = {**self.model_variable_parameters, **self.models_dict}
        self.model_combinations = Parameters.create_combinations(joined)
        self.training_combinations = Parameters.create_combinations(self.training_parameters)
        self.num_experiments = len(self.model_combinations) * len(self.training_combinations)
        self.experiments_ran = []
        if delete_old:
            self.delete_old_files()
        else:
            self.decide_which_to_run()

    def delete_old_files(self):
        dirpath = self.paths_dict['paths']['output']['root']
        if os.path.basename(dirpath).lower() not in ('output', 'out', 'results'):
            for root, dirs, files in os.walk(dirpath):
                for file in files:
                    os.remove(os.path.join(root, file))

    def decide_which_to_run(self):
        parameters_dir = self.paths_dict['paths']['output']['parameters']
        if os.path.exists(parameters_dir):
            for file_name in os.listdir(parameters_dir):
                # get id from filename end
                experiment_id = file_name.split('.')[0].split('_')[-1]
                self.experiments_ran.append(int(experiment_id))

    def load_data(self) -> None:
        self.training_parameters = self.paths_dict['training']
        x_train = self.paths_dict['paths']['input']['x-train']
        x_test = self.paths_dict['paths']['input']['x-test']
        y_train = self.paths_dict['paths']['input']['y-train']
        y_test = self.paths_dict['paths']['input']['y-test']

        self.x_train = smart_load_tensors(x_train, self.convolution_dims)
        self.x_test = smart_load_tensors(x_test, self.convolution_dims)

        if x_train != y_train and x_test != y_test:
            self.y_train = smart_load_tensors(y_train, self.convolution_dims)
            self.y_test = smart_load_tensors(y_test, self.convolution_dims)
        else:
            self.y_train = None
            self.y_test = None

        self.dataloader_train, self.dataloader_val = create_dataloaders(self.x_train, self.x_test, self.y_train,
                                                                        self.y_test)
        self.is_autoencoder = self.dataloader_val.dataset.is_autoencoder
        self.x_shape = self.dataloader_val.dataset.x_shape
        self.y_shape = self.dataloader_val.dataset.y_shape
        self.input_shape = get_proper_convolution_shape(self.x_shape, self.convolution_dims)
        self.output_shape = get_proper_convolution_shape(self.y_shape, self.convolution_dims)



        self.model_variable_parameters = self.paths_dict['model']
        self.model_shape_parameters = {'input_size': self.input_shape[-1],
                                        'output_size': self.output_shape[-1],
                                        'input_shape': self.input_shape,
                                        'output_shape': self.output_shape}

        print(f'Loaded data with input shape {self.input_shape} and output shape {self.output_shape}.')

    def load_networks(self):
        paths_dict = self.paths_dict
        module_paths = [string.split('.') for string in paths_dict['model']['networks']]
        modules = ['networks.' + ''.join(module_paths[:-1]) for module_paths in module_paths]
        classes = [module_paths[-1] for module_paths in module_paths]

        modules = [importlib.import_module(module) for module in modules]
        models = [getattr(module, model) for module, model in zip(modules, classes)]
        self.paths_dict['model']['networks'] = models
        self.models = [filtered(model) for model in models]
        self.models_dict = {'model': [model for model in self.models]}


class Analysis:
    """Creates all possible combinations of parameters for the experiment using cartesian product."""

    def __init__(self, analysis_loader: AnalysisLoader, logger, optimizer, criterion, is_convolutional=False) -> None:

        self.loader = analysis_loader
        self.is_convolutional = is_convolutional
        self.optimizer = optimizer
        self.criterion = criterion
        self.model_combinations = analysis_loader.model_combinations
        self.training_combinations = analysis_loader.training_combinations
        self.num_experiments = analysis_loader.num_experiments
        self.experiments_ran = analysis_loader.experiments_ran
        self.model_shape_parameters = analysis_loader.model_shape_parameters
        self.dataloader_train = analysis_loader.dataloader_train
        self.dataloader_val = analysis_loader.dataloader_val
        self.logger = logger
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



