import torch
import torch.nn as nn

from utils import conv_out_repeated, conv_out_vect


class ConvolutionalEncoder1d(nn.Module):
    """Convolutional Encoder for AutoEncoder with batch normalization and residual connections."""

    def __init__(self, input_shape, encoded_size, num_layers, kernel_size=3, stride=1, padding=1,
                 activation=nn.ReLU()):
        super(ConvolutionalEncoder1d, self).__init__()
        hidden_size = 2 * encoded_size
        self.convolutions = nn.ModuleList()
        dofs = input_shape[0]
        channels = [dofs * 2 ** i for i in range(num_layers + 1)]
        for i in range(num_layers):
            input_channels = channels[i]
            output_channels = channels[i + 1]
            self.convolutions.append(nn.Conv1d(input_channels, output_channels, kernel_size, stride, padding))
            self.convolutions.append(activation)
            self.convolutions.append(nn.BatchNorm1d(output_channels))

        self.convolutions = nn.Sequential(*self.convolutions)

        last_output_size = conv_out_repeated(input_shape[1], kernel_size, stride, padding,
                                             num_reps=num_layers)
        input_size_linear = output_channels * last_output_size
        self.fc_layers = nn.Sequential(
            nn.Linear(input_size_linear, hidden_size),
            activation,
            nn.Linear(hidden_size, encoded_size),
        )

        self.convolution_part = self.convolutions
        self.linear_part = self.fc_layers

    def forward(self, x):
        # remove the channel dimension
        x = x.squeeze()
        x = self.convolution_part(x)
        out = x.view(x.size(0), -1)
        out = self.linear_part(out)
        return out


class ConvEncoder1dFixed(nn.Module):

    def __init__(self, input_shape, encoded_size, num_layers=4):
        super(ConvEncoder1dFixed, self).__init__()
        initial_channels = input_shape[0]
        timesteps = input_shape[1]
        channels_out = [2 ** (i + 3) * initial_channels for i in range(num_layers)]
        channels_in = [initial_channels] + channels_out[:-1]
        kernels = [3, 3, 3, 3]
        strides = [1, 1, 1, 1]
        paddings = [1, 1, 1, 1]
        self.encoder = nn.ModuleList()

        for i in range(num_layers):
            self.encoder.append(nn.Conv1d(channels_in[i], channels_out[i], kernels[i], strides[i], paddings[i]))
            self.encoder.append(nn.ReLU())
            if i % 2 == 0:
                self.encoder.append(nn.BatchNorm1d(channels_out[i]))

        self.encoder = nn.Sequential(*self.encoder)

        dimensions = conv_out_vect(timesteps, kernels, strides, paddings, which=0)

        self.fc_dims = [dimensions[-1] * channels_out[-1], 2 * encoded_size, encoded_size]
        self.fc = nn.Sequential(
            nn.Linear(self.fc_dims[0], self.fc_dims[1]),
            nn.ReLU(),
            nn.Linear(self.fc_dims[1], self.fc_dims[2])
        )
        self.signal_dims = dimensions
        self.channels = channels_in + [channels_out[-1]]

    def __repr__(self):
        print(f"Encoder summary:")
        # format num_channels, num_timesteps
        print(f"Input shape: {self.input_shape}")
        for i, (channels, timesteps) in enumerate(zip(self.channels, self.signal_dims)):
            print(f"Layer {i}: {channels} channels, {timesteps} timesteps")
        print(f"Output shape: {self.output_shape}")
        print(f"Number of layers: {self.num_layers}")

    def forward(self, x):
        x = x.squeeze(1)
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ConvEncoder1dFixed2(ConvEncoder1dFixed):

    def __init__(self, input_shape, encoded_size, num_layers=4):
        super(ConvEncoder1dFixed, self).__init__()
        self.input_shape = input_shape
        self.output_shape = torch.Size([encoded_size, 1])
        self.num_layers = num_layers

        initial_channels = input_shape[0]
        timesteps = input_shape[1]
        channels_out = [2 ** (i + 2) * initial_channels for i in range(num_layers)]
        channels_in = [initial_channels] + channels_out[:-1]

        self.encoder = nn.ModuleList()

        for i in range(num_layers):
            self.encoder.append(nn.Conv1d(channels_in[i], channels_out[i], 3, 1, 1))
            self.encoder.append(nn.ReLU())
            self.encoder.append(nn.MaxPool1d(3, 2, 1))

        self.encoder = nn.Sequential(*self.encoder)

        signal_dims = conv_out_vect(timesteps, 3, [2] * num_layers, 1, which=0)

        self.fc_dims = [signal_dims[-1] * channels_out[-1], 2 * encoded_size, encoded_size]
        self.fc = nn.Sequential(
            nn.Linear(self.fc_dims[0], self.fc_dims[1]),
            nn.ReLU(),
            nn.Linear(self.fc_dims[1], self.fc_dims[2]),
        )
        self.signal_dims = signal_dims
        self.channels = channels_in + [channels_out[-1]]

    def __repr__(self):
        x = [f"Input shape: {self.input_shape}"]
        for i, (channels, timesteps) in enumerate(zip(self.channels, self.signal_dims)):
            x.append(f"Layer {i}: {channels} channels, {timesteps} timesteps")
        x.append(f"Output shape: {self.output_shape}")
        return "\n".join(x)
