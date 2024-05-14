import torch.nn as nn

from utils import conv_out_transpose_repeated
from ..blocks import ResidualBlock, ResidualBlockLinear, ResidualBlockTranspose1d


class ConvolutionalDecoder1d(nn.Module):

    def __init__(self, encoded_size, output_shape, num_layers, kernel_size=3, stride=2, padding=1,
                 activation=nn.ReLU()):
        super(ConvolutionalDecoder1d, self).__init__()
        self.encoded_size = encoded_size
        self.output_shape = output_shape
        self.activation = activation
        convolutional_input_size = 2
        dofs, timesteps = output_shape

        num_channels = [dofs * 2 ** i for i in range(num_layers, 0, -1)]
        self.num_channels = num_channels
        linear_output_size = num_channels[0] * convolutional_input_size
        self.fc_layers = nn.Sequential(
            nn.Linear(encoded_size, 2 * encoded_size),
            activation,
            nn.Linear(2 * encoded_size, linear_output_size),
            activation)

        self.deconvolutions = nn.ModuleList()
        for i in range(num_layers - 1):
            input_channels = num_channels[i]
            output_channels = num_channels[i + 1]
            self.deconvolutions.append(
                nn.ConvTranspose1d(input_channels, output_channels, kernel_size, stride, padding))
            self.deconvolutions.append(nn.BatchNorm1d(output_channels))
            self.deconvolutions.append(activation)

        _, last_output_size = conv_out_transpose_repeated(convolutional_input_size, kernel_size, stride,
                                                          padding, num_layers=num_layers - 1)
        output_conformed_transpose_layer = ResidualBlock.shortcut_conv_transpose_1d(num_channels[-1], dofs,
                                                                                    last_output_size, timesteps)
        self.deconvolutions.append(output_conformed_transpose_layer)
        self.deconvolutions = nn.Sequential(*self.deconvolutions)

        self.linear_part = ResidualBlockLinear(self.fc_layers, encoded_size, linear_output_size, activation)
        self.deconvolution_part = ResidualBlockTranspose1d(self.deconvolutions, num_channels[0], dofs,
                                                           convolutional_input_size, timesteps, activation)

    def forward(self, x):
        x = x.squeeze()
        x = self.linear_part(x)
        x = x.view(x.size(0), self.num_channels[0], -1)
        x = self.deconvolution_part(x)
        return x.unsqueeze(1)
