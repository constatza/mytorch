import torch.nn as nn


class ConstantHiddenSizeFFNN(nn.Module):
    def __init__(self, input_size=None, output_size=None, hidden_layers=None, hidden_size=None, activation=nn.ReLU):
        super(ConstantHiddenSizeFFNN, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size
        self.activation = activation

        self.layers = nn.ModuleList()

        self.layers.append(nn.BatchNorm1d(input_size))
        self.layers.append(nn.Linear(input_size, hidden_size))
        self.layers.append(activation())

        for _ in range(hidden_layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(activation())

        self.layers.append(nn.Linear(hidden_size, output_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
