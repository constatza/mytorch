import torch.nn as nn

from utils import conv_out_repeated
from ..blocks import ResidualBlock1d, ResidualBlockLinear


class ConvolutionalEncoder1d(nn.Module):
    """Convolutional Encoder for AutoEncoder with batch normalization and residual connections."""

    def __init__(self, input_shape, encoded_size, num_layers, kernel_size=3, stride=2, padding=1, hidden_size=1000,
                 activation=nn.ReLU()):
        super(ConvolutionalEncoder1d, self).__init__()
        self.activation = activation
        last_output_size, _ = conv_out_repeated(input_shape[1], kernel_size, stride, padding,
                                                num_layers=num_layers)
        self.convolutions = nn.ModuleList()
        dofs = input_shape[0]
        channels = [dofs * 2 ** i for i in range(num_layers + 1)]
        for i in range(num_layers):
            input_channels = channels[i]
            output_channels = channels[i + 1]
            self.convolutions.append(nn.Conv1d(input_channels, output_channels, kernel_size, stride, padding))
            self.convolutions.append(nn.BatchNorm1d(output_channels))
            self.convolutions.append(activation)
        self.convolutions = nn.Sequential(*self.convolutions)

        input_size_linear = output_channels * last_output_size
        self.fc_layers = nn.Sequential(
            nn.Linear(input_size_linear, hidden_size),
            activation,
            nn.Linear(hidden_size, encoded_size),
            # activation,
        )

        self.convolution_part = ResidualBlock1d(self.convolutions, input_shape[0], output_channels,
                                                input_shape[1], last_output_size, activation)
        self.linear_part = ResidualBlockLinear(self.fc_layers, input_size_linear, encoded_size, activation)

    def forward(self, x):
        # remove the channel dimension
        x = x.squeeze()
        x = self.convolution_part(x)
        out = x.view(x.size(0), -1)
        out = self.linear_part(out)
        return out
