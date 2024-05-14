import os

import numpy as np
import torch

from networks.caes import UNet
from preprocessing import scale_data, split_data, training_autoencoder

# load data
print(os.getcwd())
data_dir = os.path.join('..', 'data', 'solutions')
solutions_path = os.path.join(data_dir, 'formatted_solutions.npy')
dataset = np.load(solutions_path)

dataset = scale_data(dataset)
x_train, x_test = split_data(dataset)

# converto to torch tensors
x_train = torch.tensor(x_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
torch.save(x_train, '../data/solutions/x_train.pt')
torch.save(x_test, '../data/solutions/x_test.pt')

# create dataloaders
input_shape = x_train.shape[2:]
encoded_size = 1000
num_layers = 4
model = UNet(1)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
train_loss, val_loss = training_autoencoder(model, x_train, x_test, optimizer=optimizer, batch_size=50, num_epochs=500)
np.save('../data/models/train_loss.npy', train_loss)
np.save('../data/models/val_loss.npy', val_loss)
torch.save(model.state_dict(), f'../data/models/UNet.pth')
