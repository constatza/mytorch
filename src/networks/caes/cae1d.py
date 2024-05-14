from torch.nn import ReLU

from base import CAE
from ..decoders import ConvolutionalDecoder1d
from ..encoders import ConvolutionalEncoder1d


class CAE1d(CAE):
    def __init__(self, input_shape, num_layers=5, encoded_size=500, kernel_size=3, stride=2, padding=1,
                 activation=ReLU()):
        encoder = ConvolutionalEncoder1d(input_shape, encoded_size, num_layers, kernel_size, stride, padding,
                                         encoded_size, activation)
        decoder = ConvolutionalDecoder1d(encoded_size, input_shape, num_layers, kernel_size, stride + 1, padding,
                                         encoded_size, activation)
        super(CAE1d, self).__init__(encoder, decoder)
