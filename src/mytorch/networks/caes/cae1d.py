from typing import List

import numpy as np
import torch
from pydantic import validate_call
from torch import nn

from mytorch.mytypes import TupleLike
from mytorch.networks.caes.base import CAE


class CAE1d(CAE):
    def __init__(
        self,
        encoder,
        decoder,
        activation: nn.functional = nn.functional.gelu,
        lr: float = 1e-3,
    ):
        self.save_hyperparameters(ignore=["encoder", "decoder"])
        super(CAE1d, self).__init__(encoder, decoder)
        self.activation = activation

    def forward(self, x):
        x = x.squeeze(1)
        x = self.encode(x)
        x = self.decode(x)
        return x

    def encode(self, x):
        x = x.squeeze(1)
        return self.encoder(x)

    def decode(self, x):
        x = x.squeeze(1)
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


class LinearChannelDescentLatent2d(CAE1d):
    @validate_call(config={"arbitrary_types_allowed": True})
    def __init__(
        self,
        input_shape: TupleLike,
        reduced_channels: int = 10,
        reduced_timesteps: int = 5,
        num_layers: int = 4,
        kernel_size: int = 5,
        lr: float = 1e-3,
        activation: nn.functional = nn.functional.gelu,
    ):
        self.save_hyperparameters(ignore="activation")
        self.activation = activation
        # lightining test tensor to print the network shapes
        self.example_input_array = torch.zeros(input_shape)

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

        encoder = self.create_encoder(
            channels,
            num_layers,
            kernel_size,
            timesteps=timesteps,
        )
        decoder = self.create_decoder(
            channels,
            num_layers,
            kernel_size,
            timesteps=timesteps,
            input_shape=input_shape,
        )
        super(LinearChannelDescentLatent2d, self).__init__(encoder, decoder)

    @staticmethod
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

    @staticmethod
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
                nn.Conv1d(
                    channels[i], channels[i], kernel_size, stride=1, padding="same"
                )
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

    def decode(self, x):
        return super(LinearChannelDescentLatent2d, self).decode(x)[
            ..., : self.hparams.input_shape[-1]
        ]


class BasicCAE(LinearChannelDescentLatent2d):
    def __init__(self, *args, **kwargs):
        self.save_hyperparameters()
        _ = kwargs.pop("latent_size", None)
        super(BasicCAE, self).__init__(*args, **kwargs)
        self.example_input_array = torch.randn(self.hparams.input_shape)

        self.linear_encoder = nn.Linear(
            self.hparams.reduced_channels * self.hparams.reduced_timesteps,
            self.hparams.latent_size,
        )
        self.linear_decoder = nn.Linear(
            self.hparams.latent_size,
            self.hparams.reduced_channels * self.hparams.reduced_timesteps,
        )

    def encode(self, x):
        x = super(BasicCAE, self).encode(x)
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
        x = super(BasicCAE, self).decode(x)
        return x


class ForcedLatentSpace(BasicCAE):

    def __init__(self, *args, **kwargs):
        super(ForcedLatentSpace, self).__init__(*args, **kwargs)
        print(self.hparams)

    def training_step(self, batch, batch_idx):
        u, latent = batch
        u_hat = self.forward(u)
        latent_hat = self.encode(u)
        loss = self.loss(u_hat, u, latent_hat, latent)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        u, latent = batch
        u_hat = self.forward(u)
        latent_hat = self.encode(u)
        loss = self.loss(u_hat, u, latent_hat, latent)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        u = batch[0]
        u_hat = self.forward(u)
        loss = self.test_loss(u_hat, u)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "strict": True,
            },
        }

    @staticmethod
    def loss(u_hat, u, latent_hat, latent, weight: float = 1e-4):
        reconstruction_loss = nn.functional.mse_loss(u_hat, u)
        latent_loss = nn.functional.mse_loss(latent_hat, latent)
        return reconstruction_loss + weight * latent_loss
