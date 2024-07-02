from typing import Tuple, List

import numpy as np
from torch import nn

from mytorch.networks.utils import conv_out_repeated
from .base import CAE

ListLike = Tuple | List


class CAE1d(CAE):
    def __init__(self, encoder, decoder):
        super(CAE1d, self).__init__(encoder, decoder)
        # print summary of the model with torchsummary

    def forward(self, x):
        x = x.squeeze(1)
        return super(CAE1d, self).forward(x).unsqueeze(1)

    def encode(self, x):
        x = x.squeeze(1)
        return super(CAE1d, self).encode(x).unsqueeze(1)

    def decode(self, x):
        x = x.squeeze(1)
        return super(CAE1d, self).decode(x).unsqueeze(1)


class LinearChannelDescentLatent2d(CAE1d):
    def __init__(
        self,
        input_shape: ListLike,
        latent_size: int = 20,
        num_layers: int = 4,
        kernel_size: int = 5,
    ):
        self.input_shape = input_shape
        self.latent_size = latent_size
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.stride = 3
        self.padding = 1
        self.num_reduced_time_steps = conv_out_repeated(
            input_shape[-1], kernel_size, self.stride, self.padding, num_reps=num_layers
        )
        encoder = self.create_encoder()
        decoder = self.create_decoder()
        super(LinearChannelDescentLatent2d, self).__init__(encoder, decoder)
        self.encoder = encoder
        self.decoder = decoder

    def create_encoder(self):
        input_shape = self.input_shape
        num_layers = self.num_layers
        dofs = input_shape[0]

        kernel_size = self.kernel_size
        stride = self.stride
        padding = self.padding

        # get channels as python integers, not numpy.in32
        in_channels = (
            np.linspace(dofs, self.latent_size, num_layers + 1).astype(int).tolist()
        )
        self.in_channels = in_channels

        encoder = nn.ModuleList()
        # encoder.append(nn.BatchNorm1d(dofs, affine=False))
        for i in range(num_layers):
            encoder.append(
                nn.Conv1d(in_channels[i], in_channels[i], kernel_size, stride, padding)
            )
            encoder.append(nn.GELU())
            encoder.append(
                nn.Conv1d(
                    in_channels[i],
                    in_channels[i + 1],
                    kernel_size=kernel_size,
                    stride=1,
                    padding="same",
                )
            )
            encoder.append(nn.GELU())

        encoder.append(
            nn.Conv1d(
                in_channels[-1],
                in_channels[-1],
                kernel_size=kernel_size,
                stride=1,
                padding="same",
            )
        )

        encoder = nn.Sequential(*encoder)
        return encoder

    def create_decoder(self):
        input_shape = self.input_shape
        num_layers = self.num_layers
        kernel_size = self.kernel_size
        stride = self.stride
        padding = self.padding
        in_channels = self.in_channels

        dofs = input_shape[0]

        decoder = nn.ModuleList()

        for i in range(1, num_layers + 1):
            decoder.append(
                nn.ConvTranspose1d(
                    in_channels[-i],
                    in_channels[-i],
                    kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )
            decoder.append(nn.GELU())
            decoder.append(
                nn.Conv1d(
                    in_channels[-i],
                    in_channels[-i - 1],
                    kernel_size=kernel_size,
                    stride=1,
                    padding="same",
                )
            )
            decoder.append(nn.GELU())

        decoder.append(
            nn.Conv1d(dofs, dofs, kernel_size, stride=1, padding=kernel_size)
        )
        decoder.append(nn.GELU())
        decoder.append(
            nn.Conv1d(dofs, dofs, kernel_size=kernel_size, stride=1, padding="same")
        )
        decoder = nn.Sequential(*decoder)

        return decoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = x[:, :, : self.input_shape[-1]]
        return x


class LinearChannelDescentLatent1d(LinearChannelDescentLatent2d):
    def __init__(self, input_shape, latent_size=20, num_layers=4, kernel_size=5):
        super(LinearChannelDescentLatent1d, self).__init__(
            input_shape, latent_size, num_layers, kernel_size
        )

        self.linear_encoder = nn.Linear(
            self.latent_size * self.num_reduced_time_steps, self.latent_size
        )
        self.linear_decoder = nn.Linear(
            self.latent_size, self.latent_size * self.num_reduced_time_steps
        )
        # self.encoder = nn.Sequential(self.encoder, self.linear_encoder)
        # self.decoder = nn.Sequential(self.linear_decoder, self.decoder)

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.linear_encoder(x)
        return x

    def decode(self, x):
        x = self.linear_decoder(x)
        x = x.view(x.size(0), self.latent_size, self.num_reduced_time_steps)
        x = self.decoder(x)
        x = x[:, :, : self.input_shape[-1]]
        return x
