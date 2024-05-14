from torch import nn

from utils import conv_out_transpose_repeated
from ..blocks import ResidualBlockTranspose2d


class ConvolutionalDecoder(nn.Module):
    def __init__(self, encoded_size, output_shape, num_layers=3, kernel_size=3, stride=1, padding=1, hidden_size=1000,
                 activation=nn.ReLU()):
        super(ConvolutionalDecoder, self).__init__()
        self.encoded_size = encoded_size
        self.output_shape = output_shape
        self.activation = activation

        initial_channels = 4
        self.channels = [initial_channels ** i for i in range(num_layers, 0, -1)]

        deconvolution_input_size = 4
        self.deconvolutions_input_shape = (deconvolution_input_size, deconvolution_input_size)
        linear_output_size = self.channels[0] * deconvolution_input_size ** 2

        self.fc_layers = nn.Sequential(
            nn.Linear(encoded_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            activation,
            nn.Linear(hidden_size, linear_output_size),
            nn.BatchNorm1d(linear_output_size),
            activation)

        stride = 2
        kernel_size = (2, 6)
        previous_output_shape = conv_out_transpose_repeated(self.deconvolutions_input_shape, kernel_size,
                                                            stride, padding, num_layers=num_layers - 1)
        print(previous_output_shape)
        # output_conformed_transpose_layer = ResidualBlock.shortcut_conv_transpose_2d(self.channels[-1], 1, previous_output_shape, output_shape)
        last_layer = ConvPrefix2d(self.channels[-1], 1, previous_output_shape, output_shape, kernel_size)

        self.deconvolutions = nn.ModuleList()
        for i in range(num_layers - 1):
            input_channels = self.channels[i]
            output_channels = self.channels[i + 1]
            self.deconvolutions.append(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding))
            self.deconvolutions.append(nn.BatchNorm2d(output_channels))
            self.deconvolutions.append(activation)

        self.deconvolutions.append(last_layer)
        self.deconvolutions = nn.Sequential(*self.deconvolutions)

        self.linear_part = self.fc_layers
        self.deconvolution_part = ResidualBlockTranspose2d(self.deconvolutions, self.channels[0], 1,
                                                           self.deconvolutions_input_shape,
                                                           output_shape)

    def forward(self, x):
        x = self.linear_part(x)
        x = x.view(x.size(0), self.channels[0], *self.deconvolutions_input_shape)
        x = self.deconvolution_part(x)
        return x


class DecoderFixed2d(nn.Module):

    def __init__(self, output_shape, encoded_size=128):
        super(DecoderFixed2d, self).__init__()
        linear_sizes = [2 ** i * encoded_size for i in range(4)]
        # final size must be output_shape
        self.sequential = nn.Sequential(
            nn.Linear(linear_sizes[0], linear_sizes[1]),
            nn.ReLU(),
            nn.Linear(linear_sizes[1], linear_sizes[2]),
            nn.ReLU(),
            nn.Linear(linear_sizes[2], linear_sizes[3]),
            nn.ReLU(),
            nn.Unflatten(1, (linear_sizes[-1] // 16, 4, 4)),
            nn.Upsample(size=(16, 32), mode='bicubic'),
            nn.ConvTranspose2d(linear_sizes[-1] // 16, 256, 3, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 16, 3, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 16, 9, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 16, 6, 1, 1),
            nn.AdaptiveAvgPool2d(output_shape),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, 1, 1),
        )

    def forward(self, x):
        x = self.sequential(x)
        return x
