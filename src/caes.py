from torch import nn

import blocks
from utils import convolutional_output_repeated, convolutional_output_transpose_repeated


class CAE(nn.Module):

    def __init__(self, encoder, decoder):
        super(CAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)


class ConvolutionalEncoder(nn.Module):
    """Convolutional Encoder for AutoEncoder with batch normalization
    and residual connections."""

    def __init__(self, input_shape, encoded_size, num_layers, kernel_size=3, stride=2, padding=1, hidden_size=1000,
                 activation=nn.ReLU()):
        super(ConvolutionalEncoder, self).__init__()
        self.activation = activation

        initial_channels = 4
        kernel_size = (2, 6)
        last_output_shape = convolutional_output_repeated(input_shape, kernel_size, stride, padding,
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
        self.convolution_part = blocks.ResidualBlock2d(self.convolutions, 1, channels[-1], input_shape,
                                                       last_output_shape)

        self.linear_part = self.fc_layers

    def forward(self, x):
        x = self.convolution_part(x)
        out = x.view(x.size(0), -1)
        out = self.linear_part(out)
        return out


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
        previous_output_shape = convolutional_output_transpose_repeated(self.deconvolutions_input_shape, kernel_size,
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
        self.deconvolution_part = blocks.ResidualBlockTranspose2d(self.deconvolutions, self.channels[0], 1,
                                                                  self.deconvolutions_input_shape,
                                                                  output_shape)

    def forward(self, x):
        x = self.linear_part(x)
        x = x.view(x.size(0), self.channels[0], *self.deconvolutions_input_shape)
        x = self.deconvolution_part(x)
        return x


class CAE2d(CAE):
    def __init__(self, input_shape, num_layers=3, encoded_size=500, kernel_size=3, stride=2, padding=1,
                 activation=nn.ReLU()):
        encoder = ConvolutionalEncoder(input_shape, encoded_size, num_layers, kernel_size, stride, padding,
                                       encoded_size, activation)
        decoder = ConvolutionalDecoder(encoded_size, input_shape, num_layers, kernel_size, stride + 1, padding,
                                       encoded_size, activation)
        super(CAE2d, self).__init__(encoder, decoder)


class ConvolutionalEncoder1d(nn.Module):
    """Convolutional Encoder for AutoEncoder with batch normalization and residual connections."""

    def __init__(self, input_shape, encoded_size, num_layers, kernel_size=3, stride=2, padding=1, hidden_size=1000,
                 activation=nn.ReLU()):
        super(ConvolutionalEncoder1d, self).__init__()
        self.activation = activation
        last_output_size, _ = convolutional_output_repeated(input_shape[1], kernel_size, stride, padding,
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

        self.convolution_part = blocks.ResidualBlock1d(self.convolutions, input_shape[0], output_channels,
                                                       input_shape[1], last_output_size, activation)
        self.linear_part = blocks.ResidualBlockLinear(self.fc_layers, input_size_linear, encoded_size, activation)

    def forward(self, x):
        # remove the channel dimension
        x = x.squeeze()
        x = self.convolution_part(x)
        out = x.view(x.size(0), -1)
        out = self.linear_part(out)
        return out


class ConvolutionalDecoder1d(nn.Module):

    def __init__(self, encoded_size, output_shape, num_layers, kernel_size=3, stride=1, padding=1, hidden_size=1000,
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
            nn.Linear(encoded_size, hidden_size),
            activation,
            nn.Linear(hidden_size, linear_output_size),
            activation)

        self.deconvolutions = nn.ModuleList()
        for i in range(num_layers - 1):
            input_channels = num_channels[i]
            output_channels = num_channels[i + 1]
            self.deconvolutions.append(
                nn.ConvTranspose1d(input_channels, output_channels, kernel_size, stride, padding))
            self.deconvolutions.append(nn.BatchNorm1d(output_channels))
            self.deconvolutions.append(activation)

        _, last_output_size = convolutional_output_transpose_repeated(convolutional_input_size, kernel_size, stride,
                                                                      padding, num_layers=num_layers - 1)
        output_conformed_transpose_layer = blocks.ResidualBlock.shortcut_conv_transpose_1d(num_channels[-1], dofs,
                                                                                           last_output_size, timesteps)
        self.deconvolutions.append(output_conformed_transpose_layer)
        self.deconvolutions = nn.Sequential(*self.deconvolutions)

        self.linear_part = blocks.ResidualBlockLinear(self.fc_layers, encoded_size, linear_output_size, activation)
        self.deconvolution_part = blocks.ResidualBlockTranspose1d(self.deconvolutions, num_channels[0], dofs,
                                                                  convolutional_input_size, timesteps, activation)

    def forward(self, x):
        x = x.squeeze()
        x = self.linear_part(x)
        x = x.view(x.size(0), self.num_channels[0], -1)
        x = self.deconvolution_part(x)
        return x.unsqueeze(1)


class CAE1d(CAE):
    def __init__(self, input_shape, num_layers=5, encoded_size=500, kernel_size=3, stride=2, padding=1,
                 activation=nn.ReLU()):
        encoder = ConvolutionalEncoder1d(input_shape, encoded_size, num_layers, kernel_size, stride, padding,
                                         encoded_size, activation)
        decoder = ConvolutionalDecoder1d(encoded_size, input_shape, num_layers, kernel_size, stride + 1, padding,
                                         encoded_size, activation)
        super(CAE1d, self).__init__(encoder, decoder)


class ConvPrefix2d(nn.Conv2d):
    """Convolutional layer with a prefixed output shape given input shape,
    kernel size, stride and padding."""

    def __init__(self, out_shape, *args, **kwargs):
        super(ConvPrefix2d, self).__init__()
