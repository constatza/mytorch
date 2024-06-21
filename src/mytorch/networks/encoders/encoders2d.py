import torch.nn as nn

from networks.blocks import ResidualBlock2d
from utils import conv_out_repeated


class ConvolutionalEncoder(nn.Module):
    """Convolutional Encoder for AutoEncoder with batch normalization
    and residual connections."""

    def __init__(self, input_shape, encoded_size, num_layers, kernel_size=3, stride=2, padding=1, hidden_size=1000,
                 activation=nn.ReLU()):
        super(ConvolutionalEncoder, self).__init__()
        self.activation = activation

        initial_channels = 4
        kernel_size = (2, 6)
        last_output_shape = conv_out_repeated(input_shape, kernel_size, stride, padding,
                                              num_layers=num_layers)
        channels = [(initial_channels) ** i for i in range(num_layers + 1)]
        self.convolutions = nn.ModuleList()

        for i in range(num_layers):
            input_channels = channels[i]
            output_channels = channels[i + 1]
            self.convolutions.append(nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding))
            self.convolutions.append(nn.BatchNorm2d(output_channels))
            self.convolutions.append(activation)

        self.convolutions = nn.Sequential(*self.convolutions)

        input_size_linear = channels[-1] * last_output_shape[0] * last_output_shape[1]
        self.fc_layers = nn.Sequential(
            nn.Linear(input_size_linear, hidden_size),
            activation,
            nn.Linear(hidden_size, encoded_size),
            nn.BatchNorm1d(encoded_size),
            # activation
        )

        # self.convolution_part = BatchResidualBlock2d(self.convolutions, 1, 64,
        #                                              input_shape, last_output_shape, activation)
        # self.linear_part = BatchResidualBlockLinear(self.fc_layers, input_size_linear, encoded_size, activation)
        self.convolution_part = ResidualBlock2d(self.convolutions, 1, channels[-1], input_shape,
                                                last_output_shape)

        self.linear_part = self.fc_layers

    def forward(self, x):
        x = self.convolution_part(x)
        out = x.view(x.size(0), -1)
        out = self.linear_part(out)
        return out
