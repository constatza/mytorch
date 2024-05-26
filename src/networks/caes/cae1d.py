import numpy as np
from torch import nn

from .base import CAE
from ..decoders import ConvDecoder1dFixed, ConvDecoder1dFixed2
from ..encoders import ConvEncoder1dFixed, ConvEncoder1dFixed2


class CAE1d(CAE):
    def __init__(self, encoder, decoder):
        super(CAE1d, self).__init__(encoder, decoder)
        # print summary of the model with torchsummary

    def forward(self, x):
        x = x.squeeze(1)
        return super(CAE1d, self).forward(x).unsqueeze(1)


class CAE1dFixed(CAE):
    def __init__(self, input_shape, encoded_size=500, num_layers=4):
        encoder = ConvEncoder1dFixed(input_shape, encoded_size, num_layers=4)

        decoder = ConvDecoder1dFixed(input_shape, encoded_size, num_layers=4, channels=encoder.channels,
                                     fc_dims=encoder.fc_dims)
        super(CAE1dFixed, self).__init__(encoder, decoder)
        # print summary of the model with torchsummary


class CAE1dFixed2(CAE):
    def __init__(self, input_shape, encoded_size=500, num_layers=4):
        encoder = ConvEncoder1dFixed2(input_shape, encoded_size, num_layers=num_layers)

        decoder = ConvDecoder1dFixed2(input_shape, encoded_size, num_layers=num_layers, channels=encoder.channels,
                                      fc_dims=encoder.fc_dims, signal_dims=encoder.signal_dims)
        print(encoder)
        print(decoder)
        super(CAE1dFixed2, self).__init__(encoder, decoder)


class CAE1dLinear(CAE1d):
    def __init__(self, input_shape, num_layers=4):
        print(input_shape)
        self.input_shape = input_shape[-2:]
        encoder, decoder = self.create_encoder_decoder(input_shape, num_layers)
        super(CAE1dLinear, self).__init__(encoder, decoder)

    def create_encoder_decoder(self, input_shape, num_layers):
        kernel_size = 5
        stride = 2
        padding = 1
        dofs = input_shape[0]
        in_channels = np.linspace(dofs, dofs * num_layers, num_layers + 1).astype(int)
        encoder = nn.ModuleList()
        for i in range(num_layers):
            encoder.append(nn.Conv1d(in_channels[i], in_channels[i + 1], kernel_size, stride, padding))
            encoder.append(nn.ReLU())

        encoder = nn.Sequential(*encoder)

        decoder = nn.ModuleList()
        for i in range(1, num_layers + 1):
            decoder.append(
                nn.ConvTranspose1d(in_channels[-i], in_channels[-i - 1], kernel_size, stride=stride + 1,
                                   padding=padding))
            decoder.append(nn.ReLU())

        decoder.append(nn.Conv1d(dofs, dofs, kernel_size, stride=1, padding='same'))
        decoder.append(nn.ReLU())
        decoder.append(nn.Conv1d(dofs, dofs, kernel_size=3, stride=1, padding='same'))
        decoder = nn.Sequential(*decoder)

        return encoder, decoder

    def forward(self, x):
        x = super(CAE1dLinear, self).forward(x)
        x = x[:, :, :, :self.input_shape[-1]]
        return x
