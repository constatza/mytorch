
import torch
import numpy as np
from preprocessing import scale_data, split_data, training_autoencoder
from src.blocks import CAE2d
import torch.nn as nn

dataset = np.load('../data/solutions/formatted_solutions.npy')

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
model = CAE2d(input_shape, num_layers, encoded_size, activation=nn.Sigmoid())

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
train_loss, val_loss = training_autoencoder(model, x_train, x_test, optimizer=optimizer, batch_size=20, num_epochs=2000)
np.save('../data/models/train_loss.npy', train_loss)
np.save('../data/models/val_loss.npy', val_loss)
torch.save(model.state_dict(), f'data/models/CAE_{encoded_size}.pth')







