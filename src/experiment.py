import importlib
import inspect
import os
from collections.abc import Iterable
from itertools import product

import numpy as np
import tomli_w as tw
import torch
from torch.utils.data import DataLoader

from parsers import TOMLParser


def filtered(func):
    def wrapper(*args, **kwargs):
        params = inspect.signature(func).parameters
        filtered = {k: v for k, v in kwargs.items() if k in params}
        return func(*args, **filtered)

    return wrapper


@filtered
class Experiment:
    def __init__(self, model=None, uid=None, x_train=None, x_test=None, y_train=None, y_test=None,
                 optimizer=None, criterion=None, lr=None, batch_size=None, num_epochs=None, checkpoint_dir=None,
                 epoch_print_interval=10):
        self.name = model.__class__.__name__ + '_' + str(uid)
        self.model = model
        self.id = uid
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.optimizer = optimizer(self.model.parameters(), lr=lr)
        self.criterion = criterion
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.checkpoint_relative_tol = 0.5
        self.dataloader_train, self.dataloader_val = create_dataloaders(x_train, y_train, x_test, y_test, batch_size)
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
            for batch in self.dataloader_train:

                if len(batch) == 2:
                    x_batch, y_batch = batch
                else:
                    x_batch = batch[0]
                    y_batch = x_batch
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()
                y_pred = self.model(x_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_losses.append(train_loss / len(self.dataloader_train))

            # Validation
            val_loss = 0
            self.model.eval()
            with torch.no_grad():
                for batch in self.dataloader_val:
                    if len(batch) == 2:
                        x_batch, y_batch = batch
                    else:
                        x_batch = batch[0]
                        y_batch = x_batch
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
                    loss_val = criterion(self.model(x_batch), y_batch)
                    val_loss += loss_val.item()
                val_losses.append(val_loss / len(self.dataloader_val))
            # print with scientific notation and 6 decimal places
            if (epoch + 1) % self.epoch_print_interval == 0:
                print(
                    f'Epoch {epoch + 1}/{self.num_epochs} | Train Loss: {train_losses[-1]:.6e} | Val Loss: {val_losses[-1]:.6e}')

            # Check if the validation loss has decreased by more than the tolerance
            if relative_tolerance(min_val_loss, val_losses) > self.checkpoint_relative_tol:
                print('Checkpointing model...')
                min_val_loss = val_losses[-1]
                # Save the model state
                best_model = torch.jit.script(self.model)
                # Save the model state to a file
                best_model.save(self.checkpoint_path)

        return train_losses, val_losses


def relative_tolerance(min_loss, losses):
    if len(losses) > 5:
        mean = np.mean(losses[-5:])
    else:
        mean = np.mean(losses)
    return (min_loss - mean) / mean


def create_dataloaders(x_train=None, y_train=None, x_test=None, y_test=None, batch_size=32):
    """Create dataloaders from the data."""
    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    if y_train is not None:
        y_train = torch.tensor(y_train, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)
        train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
        val_dataset = torch.utils.data.TensorDataset(x_test, y_test)
    else:
        train_dataset = torch.utils.data.TensorDataset(x_train)
        val_dataset = torch.utils.data.TensorDataset(x_test)

    dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dataloader_val = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return dataloader_train, dataloader_val


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


class Analysis:
    """Creates all possible combinations of parameters for the experiment using cartesian product."""

    def __init__(self, parser: TOMLParser, optimizer, criterion, new=False) -> None:

        self.parser = parser
        self.load_networks()
        self.load_data()
        self.optimizer = optimizer
        self.criterion = criterion

        joined = {**self.model_parameters, **self.models_dict}
        self.model_combinations = Parameters.create_combinations(joined)
        self.training_combinations = Parameters.create_combinations(self.training_parameters)
        self.num_experiments = len(self.model_combinations) * len(self.training_combinations)
        self.experiments_ran = []
        if new:
            self.delete_old_files()
        else:
            self.decide_which_to_run()

    def delete_old_files(self):
        dirpath = self.parser['paths']['output']['root']
        if os.path.basename(dirpath).lower() not in ('output', 'out', 'results'):
            for root, dirs, files in os.walk(dirpath):
                for file in files:
                    os.remove(os.path.join(root, file))

    def decide_which_to_run(self):
        parameters_dir = self.parser['paths']['output']['parameters']
        if os.path.exists(parameters_dir):
            for file_name in os.listdir(parameters_dir):
                # get id from filename end
                experiment_id = file_name.split('.')[0].split('_')[-1]
                self.experiments_ran.append(int(experiment_id))

    def load_networks(self):
        parser = self.parser
        module_paths = [string.split('.') for string in parser['model']['networks']]
        modules = ['networks.' + ''.join(module_paths[:-1]) for module_paths in module_paths]
        classes = [module_paths[-1] for module_paths in module_paths]

        modules = [importlib.import_module(module) for module in modules]
        models = [getattr(module, model) for module, model in zip(modules, classes)]
        self.parser['model']['networks'] = models
        self.models = [filtered(model) for model in models]
        self.models_dict = {'model': [model for model in self.models]}

    def load_data(self):
        self.model_parameters = self.parser['model']
        self.training_parameters = self.parser['training']
        x_train = self.parser['paths']['input']['x-train']
        x_test = self.parser['paths']['input']['x-test']
        if x_train.endswith('.npy'):
            self.x_train = np.load(x_train)
        else:
            self.x_train = torch.load(x_train)
        if x_test.endswith('.npy'):
            self.x_test = np.load(x_test)
        else:
            self.x_test = torch.load(x_test)
        self.input_shape = self.x_train.shape[2:]
        try:
            self.y_train = np.load(self.parser['paths']['input']['y-train'])
            self.y_test = np.load(self.parser['paths']['input']['y-test'])

        except KeyError:
            self.y_train = None
            self.y_test = None

    def run(self):
        writer = Writer(self.parser)
        losses = []
        for model_id, model_parameters in self.model_combinations.items():
            for train_id, training_parameters in self.training_combinations.items():

                experiment_id = train_id + model_id * len(self.training_combinations)
                if experiment_id not in self.experiments_ran:
                    # separate stdout line
                    print('-' * 80)
                    print(f'Running experiment {experiment_id}/{self.num_experiments}')
                    model = model_parameters['model'](self.input_shape, **model_parameters)
                    experiment = Experiment(model, x_train=self.x_train, x_test=self.x_test,
                                            optimizer=self.optimizer, criterion=self.criterion, uid=experiment_id,
                                            **training_parameters,
                                            checkpoint_dir=self.parser['paths']['output']['models'])
                    # search parameters if experiment has already been run

                    try:
                        train_loss, val_loss = experiment.train()
                        losses.append(val_loss[-1])
                        writer.log(f'Experiment {experiment_id:d} completed successfully.')
                        writer.write(f'{experiment.name}.pt', model, dirname='models')
                        parameters = {**model_parameters, **training_parameters, 'id': experiment_id}
                        writer.write(f'{experiment.name}.toml', parameters, dirname='parameters')
                        writer.write(f'losses', val_loss[-1])
                    except Exception as e:
                        raise e
                        writer.error(f'Experiment {experiment_id} failed with error: {e}')


class Writer:
    def __init__(self, parser: TOMLParser):
        self.parser = parser

    def write(self, name, data, dirname='root'):
        suffix = name.split('.')[-1]

        # select the correct extension and method
        if suffix == 'npy':
            saver = np.save
            mode = 'wb'
        elif suffix in ('pt', 'pth'):
            saver = torch.save
            mode = 'wb'
        elif suffix == 'csv':
            saver = np.savetxt
            mode = 'a'
        elif suffix == 'toml':
            def saver(dictionary, file):
                for key, value in dictionary.items():
                    # check if value is a class
                    if inspect.isclass(value) or inspect.isfunction(value):
                        dictionary[key] = None
                dictionary = {k: v for k, v in dictionary.items() if v is not None}
                tw.dump(dictionary, file)

            mode = 'wb'
        else:
            def saver(data, file):
                if not isinstance(data, str):
                    data = str(data)
                file.write(data)
                file.write('\n')

            mode = 'a'

        directory = self.parser['paths']['output'][dirname]
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, name)
        with open(path, mode) as file:
            saver(data, file)

    def log(self, message):
        return self.write('log', message)

    def error(self, message):
        return self.write('error', message)
