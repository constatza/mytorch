from typing import List

import numpy as np
import torch
from lightning import LightningModule
from pydantic import validate_call
from torch import nn

from mytorch.mytypes import TupleLike
from mytorch.networks.caes.base import CAE


def create_encoder(
    channels: List[int],
    num_layers: int,
    kernel_size: int,
    timesteps: List[int],
):

    encoder = nn.ModuleList()

    for i in range(num_layers):
        encoder.append(
            nn.Conv1d(
                channels[i],
                channels[i],
                kernel_size,
                stride=1,
                padding="same",
            )
        )
        if timesteps[-1] < timesteps[0]:
            encoder.append(nn.GELU())
            encoder.append(nn.AdaptiveMaxPool1d(timesteps[i + 1]))

        encoder.append(nn.GELU())
        encoder.append(
            nn.Conv1d(
                channels[i],
                channels[i + 1],
                kernel_size=kernel_size,
                stride=1,
                padding="same",
            )
        )

        encoder.append(nn.GELU())

    encoder.append(
        nn.Conv1d(
            channels[-1],
            channels[-1],
            kernel_size=kernel_size,
            stride=1,
            padding="same",
        )
    )

    encoder = nn.Sequential(*encoder)
    return encoder


def create_decoder(
    channels: List[int],
    num_layers: int,
    kernel_size: int,
    timesteps: List[int],
    input_shape: TupleLike,
):

    dofs = input_shape[-2]
    channels.reverse()
    timesteps.reverse()

    decoder = nn.ModuleList()

    for i in range(num_layers):
        decoder.append(
            nn.Conv1d(channels[i], channels[i], kernel_size, stride=1, padding="same")
        )

        if timesteps[-1] > timesteps[0]:
            decoder.append(nn.GELU())
            decoder.append(nn.AdaptiveMaxPool1d(timesteps[i + 1]))

        decoder.append(nn.GELU())
        decoder.append(
            nn.Conv1d(
                channels[i],
                channels[i + 1],
                kernel_size=kernel_size,
                stride=1,
                padding="same",
            )
        )
        decoder.append(nn.GELU())

    decoder.append(nn.AvgPool1d(kernel_size=7, stride=1, padding=3))
    decoder.append(nn.GELU())
    decoder.append(nn.Conv1d(dofs, dofs, 7, stride=1, padding="same"))

    decoder = nn.Sequential(*decoder)

    return decoder


class BasicCAE(CAE):

    def __init__(
        self,
        input_shape: tuple,
        reduced_channels: int = 10,
        reduced_timesteps: int = 5,
        latent_size: int = 10,
        num_layers: int = 4,
        kernel_size: int = 5,
        lr: float = 1e-3,
        activation: nn.functional = nn.functional.gelu,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(ignore="activation")
        self.activation = activation
        self.input_shape = input_shape

        self.example_input_array = torch.randn(input_shape[-3:])

        channels = (
            np.linspace(input_shape[-2], reduced_channels, num_layers + 1)
            .astype(int)
            .tolist()
        )

        timesteps = (
            np.linspace(input_shape[-1], reduced_timesteps, num_layers + 1)
            .astype(int)
            .tolist()
        )

        self.encoder = create_encoder(
            channels,
            num_layers,
            kernel_size,
            timesteps=timesteps,
        )
        self.decoder = create_decoder(
            channels,
            num_layers,
            kernel_size,
            timesteps=timesteps,
            input_shape=input_shape,
        )

        self.linear_encoder = nn.Linear(
            self.hparams.reduced_channels * self.hparams.reduced_timesteps,
            self.hparams.latent_size,
        )

        self.linear_decoder = nn.Linear(
            self.hparams.latent_size,
            self.hparams.reduced_channels * self.hparams.reduced_timesteps,
        )

    def encode(self, x):
        x = self.encoder(x)
        x = self.activation(x)
        x = torch.flatten(x, 1)
        x = self.activation(x)
        x = self.linear_encoder(x)
        return x

    def decode(self, x):
        x = self.linear_decoder(x)
        x = self.activation(x)
        x = x.view(
            x.size(0), self.hparams.reduced_channels, self.hparams.reduced_timesteps
        )
        x = self.decoder(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.6,
            patience=10,
            min_lr=1e-5,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1,
            },
        }
