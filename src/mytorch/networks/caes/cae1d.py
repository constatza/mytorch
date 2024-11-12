from typing import List

import numpy as np
import torch
from sympy import Identity
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

        # Instantiate feature extractor and latent encoder

        self.encoder = Encoder(
            input_shape=input_shape,
            latent_dim=latent_size,
            channels=channels,
            kernel_size=kernel_size,
            timesteps=timesteps,
            activation=activation,
        )

        # Instantiate latent decoder and feature decoder
        self.decoder = Decoder(
            latent_dim=latent_size,
            channels=channels,
            kernel_size=kernel_size,
            timesteps=timesteps,
            output_shape=input_shape,
        )

    def encode(self, x):
        x = self.encoder(x)
        return x

    def decode(self, x):
        x = self.decoder(x)
        return x


import torch
from torch import nn
from typing import List, Optional


class ResidualConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        downsample_timesteps: Optional[int] = None,
        activation: nn.Module = nn.GELU(),
    ):
        """
        A residual convolutional block with one downsampling layer.

        Parameters:
        - in_channels (int): Number of input channels.
        - out_channels (int): Number of output channels.
        - kernel_size (int): Kernel size for convolutions.
        - downsample_timesteps (Optional[int]): Target timesteps for adaptive pooling, None if not downsampling.
        - activation (nn.Module): Activation function to use.
        """
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, stride=1, padding="same"
        )
        self.activation = activation
        self.downsample = (
            nn.AdaptiveMaxPool1d(downsample_timesteps)
            if downsample_timesteps
            else nn.Identity()
        )

        # Residual projection if in_channels != out_channels or if downsampling alters timesteps
        self.residual_projection = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        # Residual path
        residual = self.residual_projection(x)

        # Main path: convolution + activation + optional downsampling
        x = self.activation(self.conv1(x))
        x = self.downsample(x)

        # Ensure the residual connection matches the spatial dimensions
        residual = nn.functional.adaptive_max_pool1d(
            residual, x.shape[-1]
        )  # Match time dimension if needed

        return x + residual  # Residual connection


class FeatureExtractor(nn.Module):
    def __init__(
        self,
        input_shape: tuple,
        channels: List[int],
        kernel_size: int = 3,
        timesteps: List[int] = None,
        activation: nn.Module = nn.GELU(),
    ):
        """
        Progressive encoder that downsamples using residual convolution blocks.

        Parameters:
        - input_shape (tuple): Shape of the input (batch_size, channels, timesteps).
        - channels (List[int]): List of channels for each layer.
        - kernel_size (int): Kernel size for convolutions.
        - timesteps (List[int]): List of timesteps for adaptive pooling at each layer.
        - activation (nn.Module): Activation function for each block.
        """
        super().__init__()
        self.input_shape = input_shape

        layers = []
        num_layers = len(channels) - 1

        for i in range(num_layers):
            downsample_timesteps = (
                timesteps[i + 1] if timesteps and i < len(timesteps) - 1 else None
            )
            layers.append(
                ResidualConvBlock(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    kernel_size=kernel_size,
                    downsample_timesteps=downsample_timesteps,
                    activation=activation,
                )
            )

        self.feature_extractor = nn.Sequential(*layers)

    def forward(self, x):
        return self.feature_extractor(x)


class FeatureToLatent(nn.Module):
    def __init__(self, input_shape: tuple, latent_dim: int):
        """
        Converts the feature map into a latent vector.

        Parameters:
        - input_shape (tuple): Shape of the feature map (channels, timesteps).
        - latent_dim (int): Dimension of the latent vector.
        """
        super().__init__()
        channels, timesteps = input_shape
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(channels * timesteps, latent_dim)

    def forward(self, x):
        x = self.flatten(x)
        return self.fc(x)


class Encoder(nn.Module):
    def __init__(
        self,
        input_shape: tuple,
        latent_dim: int,
        channels: List[int],
        kernel_size: int = 3,
        timesteps: List[int] = None,
        activation: nn.Module = nn.GELU(),
    ):
        """
        Complete encoder that compresses the input into a latent vector.

        Parameters:
        - input_shape (tuple): Shape of the input (batch_size, channels, timesteps).
        - latent_dim (int): Dimension of the latent vector.
        - channels (List[int]): List of channels for each layer.
        - kernel_size (int): Kernel size for convolutions.
        - timesteps (List[int]): List of timesteps for adaptive pooling at each layer.
        - activation (nn.Module): Activation function for each block.
        """
        super().__init__()
        self.feature_extractor = FeatureExtractor(
            input_shape, channels, kernel_size, timesteps, activation
        )

        # Compute the output shape of the feature extractor for initializing FeatureToLatent
        final_channels, final_timesteps = channels[-1], timesteps[-1]
        self.feature_to_latent = FeatureToLatent(
            (final_channels, final_timesteps), latent_dim
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.feature_to_latent(x)
        return x


class ResidualConvTransposeBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        upsample_timesteps: Optional[int] = None,
        activation: nn.Module = nn.GELU(),
    ):
        """
        A residual transposed convolutional block with upsampling.

        Parameters:
        - in_channels (int): Number of input channels.
        - out_channels (int): Number of output channels.
        - kernel_size (int): Kernel size for transposed convolutions.
        - stride (int): Stride for transposed convolutions to control upsampling.
        - padding (int): Padding for transposed convolutions.
        - upsample_timesteps (Optional[int]): Target timesteps for adaptive upsampling, None if not upsampling.
        - activation (nn.Module): Activation function to use.
        """
        super().__init__()
        self.conv1 = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=stride - 1,
        )
        self.activation = activation
        self.upsample = (
            nn.AdaptiveAvgPool1d(upsample_timesteps)
            if upsample_timesteps
            else nn.Identity()
        )

        # Residual projection to match dimensions
        self.residual_projection = (
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        residual = self.residual_projection(x)
        x = self.activation(self.conv1(x))
        x = self.upsample(x)
        residual = nn.functional.adaptive_avg_pool1d(residual, x.shape[-1])
        return x + residual


class LatentToFeature(nn.Module):
    def __init__(self, latent_dim: int, target_shape: tuple):
        """
        Converts latent vector into a feature map for the decoder.

        Parameters:
        - latent_dim (int): Dimension of the latent vector.
        - target_shape (tuple): Target shape as (channels, timesteps) for the feature map.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.target_shape = target_shape
        self.fc = nn.Linear(latent_dim, target_shape[0] * target_shape[1])

    def forward(self, x):
        x = self.fc(x)
        return x.view(
            x.size(0), *self.target_shape
        )  # Reshape to (batch_size, channels, timesteps)


class FeatureDecoder(nn.Module):
    def __init__(
        self,
        channels: List[int],
        kernel_size: int = 3,
        timesteps: List[int] = None,
        activation: nn.Module = nn.GELU(),
    ):
        """
        Progressive decoder that upsamples using residual transpose blocks.

        Parameters:
        - channels (List[int]): List of channels for each layer, in reverse order from the encoder.
        - kernel_size (int): Kernel size for transposed convolutions.
        - timesteps (List[int]): List of timesteps for adaptive upsampling at each layer.
        - activation (nn.Module): Activation function to use.
        """
        super().__init__()
        layers = []
        num_layers = len(channels) - 1

        for i in range(num_layers):
            upsample_timesteps = (
                timesteps[i + 1] if timesteps and i < len(timesteps) - 1 else None
            )
            layers.append(
                ResidualConvTransposeBlock(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    kernel_size=kernel_size,
                    stride=2,
                    padding=1,
                    upsample_timesteps=upsample_timesteps,
                    activation=activation,
                )
            )

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)


class Decoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        channels: List[int],
        kernel_size: int = 3,
        timesteps: List[int] = None,
        activation: nn.Module = nn.GELU(),
        output_shape: tuple = None,
    ):
        """
        Complete decoder that reconstructs the input from a latent vector.

        Parameters:
        - latent_dim (int): Dimension of the latent vector input.
        - channels (List[int]): List of channels for each layer, in reverse order from the encoder.
        - kernel_size (int): Kernel size for transposed convolutions.
        - timesteps (List[int]): List of timesteps for adaptive upsampling.
        - activation (nn.Module): Activation function for each block.
        - output_shape (tuple): Target output shape (batch_size, channels, timesteps) to guarantee correct reconstruction.
        """
        super().__init__()
        channels = channels[::-1]
        timesteps = timesteps[::-1]
        self.latent_to_feature = LatentToFeature(
            latent_dim, (channels[0], timesteps[0])
        )
        self.feature_decoder = FeatureDecoder(
            channels, kernel_size, timesteps, activation
        )
        self.output_shape = output_shape

    def forward(self, x):
        x = self.latent_to_feature(x)
        x = self.feature_decoder(x)

        # Ensure the output shape matches the input shape
        target_channels, target_timesteps = self.output_shape[-2:]
        x = nn.functional.interpolate(
            x, size=(target_timesteps,), mode="linear", align_corners=False
        )

        return x
