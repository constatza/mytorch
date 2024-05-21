import os

import numpy as np
import torch

from networks.caes import CAE1dLinear
from preprocessing import scale_data, split_data, training_autoencoder


class Experiment:
    def __init__(self, model, optimizer, criterion, batch_size, num_epochs):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.batch_size = batch_size
        self.num_epochs = num_epochs

    def run(self, x_train, x_test):
        train_loss, val_loss = training_autoencoder(self.model, x_train, x_test, optimizer=self.optimizer,
                                                    criterion=self.criterion, batch_size=self.batch_size,
                                                    num_epochs=self.num_epochs)
        return train_loss, val_loss


# load data
torch.seed()  # set seed

config_path = os.path.join('..', 'scrips', 'exp1.toml')
parser = TOMLParser(config_path)

datadir = os.path.join('..', 'data', 'solutions500')
solutions_path = os.path.join(datadir, 'formatted_solutions.npy')
x_train_path = os.path.join(datadir, 'x_train.npy')
x_test_path = os.path.join(datadir, 'x_test.npy')
dataset = np.load(solutions_path)

dataset = scale_data(dataset)
x_train, x_test = split_data(dataset)

# converto to torch tensors
x_train = torch.tensor(x_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
torch.save(x_train, x_train_path)
torch.save(x_test, x_test_path)

# create dataloaders
input_shape = x_train.shape[2:]
encoded_size = 300
num_layers = 8
model = CAE1dLinear(input_shape, num_layers=num_layers)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
train_loss, val_loss = training_autoencoder(model, x_train, x_test, optimizer=optimizer, batch_size=80, num_epochs=500)
np.save('../data/models/train_loss.npy', train_loss)
np.save('../data/models/val_loss.npy', val_loss)
model_name = model.__class__.__name__
torch.save(model.state_dict(), f'../data/models/{model_name}.pth')
