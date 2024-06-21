import numpy as np
import torch
import torch.nn as nn

from utils import conv_out_transpose_repeated, conv_out_transpose_vect
from ..blocks import ResidualBlock, ResidualBlockLinear, ResidualBlockTranspose1d


class ConvolutionalDecoder1d(nn.Module):

    def __init__(self, output_shape, encoded_size, num_layers, kernel_size=3, stride=2, padding=1,
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
            self.deconvolutions.append(activation)
            self.deconvolutions.append(nn.BatchNorm1d(output_channels))

        last_output_size = conv_out_transpose_repeated(convolutional_input_size, kernel_size, stride,
                                                       padding, num_reps=num_layers - 1)
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


class ConvDecoder1dFixed(nn.Module):

    def __init__(self, output_shape, encoded_size, num_layers=4):
        super(ConvDecoder1dFixed, self).__init__()
        output_channels, timesteps = output_shape
        self.initial_channels = output_channels * 16
        encoded_timesteps = 16

        self.linear = nn.Sequential(
            nn.Linear(encoded_size, 2 * encoded_size),
            nn.ReLU(),
            nn.Linear(2 * encoded_size, self.initial_channels * encoded_timesteps),
            nn.ReLU(),
        )

        channels = np.logspace(np.log2(output_channels), np.log2(self.initial_channels), num_layers + 1, base=2,
                               dtype=int)
        # channels = [output_channels] * (num_layers + 1)
        channels = np.flip(channels)
        channels_next = channels[1:]
        channels_previous = channels[:-1]

        kernels = [4] * num_layers
        strides = [2] * num_layers
        paddings = [1] * num_layers
        dimensions = conv_out_transpose_vect(encoded_timesteps, kernels, strides, paddings, which=0)
        self.decoder = nn.ModuleList()
        for i in range(num_layers):
            self.decoder.append(
                nn.ConvTranspose1d(channels_previous[i], channels_next[i], kernels[i], strides[i], paddings[i]))
            self.decoder.append(nn.ReLU())
            if i % 2 == 0:
                self.decoder.append(nn.BatchNorm1d(channels_next[i]))
        self.decoder = nn.Sequential(*self.decoder)

        self.output_comformed_transpose_layer = ResidualBlock.shortcut_conv_transpose_1d(channels_next[-1],
                                                                                         output_channels,
                                                                                         dimensions[-1],
                                                                                         timesteps)
        self.refining_layers = nn.Sequential(
            nn.ConvTranspose1d(output_channels, output_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(output_channels),
            nn.ConvTranspose1d(output_channels, output_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(output_channels),
        )

    def forward(self, x):
        x = x.squeeze()
        x = self.linear(x)
        x = x.view(x.size(0), self.initial_channels, -1)
        x = self.decoder(x)
        x = self.output_comformed_transpose_layer(x)
        x = self.refining_layers(x)
        return x.unsqueeze(1)

    def __repr__(self):
        print()


class ConvDecoder1dFixed2(ConvDecoder1dFixed):

    def __init__(self, output_shape, encoded_size, num_layers=4, channels=None, fc_dims=None, signal_dims=None):
        super(ConvDecoder1dFixed, self).__init__()
        output_channels, timesteps = output_shape

        channels = np.flip(channels)
        channels_previous = channels[:-1]
        channels_next = channels[1:]

        fc_dims = np.flip(fc_dims)
        encoded_timesteps = timesteps // 2 ** num_layers

        self.input_shape = torch.Size([encoded_size, 1])
        self.output_shape = output_shape
        self.channels = channels
        self.initial_channels = channels[0]

        self.linear = nn.Sequential(
            nn.Linear(fc_dims[0], fc_dims[1]),
            nn.ReLU(),
            nn.Linear(fc_dims[1], fc_dims[2]),
            nn.ReLU(),
        )

        self.decoder = nn.ModuleList()
        for i in range(num_layers):
            self.decoder.append(nn.Upsample(scale_factor=2, mode='linear', ))

            self.decoder.append(
                nn.ConvTranspose1d(channels_previous[i], channels_next[i], 3, 1, 1))
            self.decoder.append(nn.ReLU())
        self.decoder = nn.Sequential(*self.decoder)

        self.latent_dims = conv_out_transpose_vect(encoded_timesteps, 3, [2, 2, 2, 2], 1, which=0)
        self.signal_dims = np.flip(signal_dims)

        self.output_comformed_transpose_layer = ResidualBlock.shortcut_conv_transpose_1d(channels_next[-1],
                                                                                         output_channels,
                                                                                         self.signal_dims[-1],
                                                                                         timesteps)
        self.refining_layers = nn.Sequential(
            nn.ConvTranspose1d(channels_next[-1], output_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(output_channels),
            nn.ReLU(),
            nn.Conv1d(output_channels, output_channels, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(output_channels, output_channels, kernel_size=3, stride=1, padding=1),
        )

    def __repr__(self):
        x = [f"Input shape: {self.input_shape}"]
        for i, (channels, timesteps) in enumerate(zip(self.channels, self.latent_dims)):
            x.append(f"Layer {i}: {channels} channels, {timesteps} timesteps")
        x.append(f"Output shape: {self.output_shape}")
        return "\n".join(x)

    def forward(self, x):
        x = x.squeeze()
        x = self.linear(x)
        x = x.view(x.size(0), self.initial_channels, -1)
        x = self.decoder(x)
        # x = self.output_comformed_transpose_layer(x, output_size=(-1, 83, 730))

        x = self.refining_layers(x)
        x = x[:, :, :self.output_shape[-1]]
        return x.unsqueeze(1)
