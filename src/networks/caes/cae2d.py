from base import CAE
from ..decoders import ConvolutionalDecoder
from ..encoders import ConvolutionalEncoder


class CAE2d(CAE):
    def __init__(self, input_shape, num_layers=3, encoded_size=500, kernel_size=3, stride=2, padding=1,
                 activation=nn.ReLU()):
        encoder = ConvolutionalEncoder(input_shape, encoded_size, num_layers, kernel_size, stride, padding,
                                       encoded_size, activation)
        decoder = ConvolutionalDecoder(encoded_size, input_shape, num_layers, kernel_size, stride + 1, padding,
                                       encoded_size, activation)
        super(CAE2d, self).__init__(encoder, decoder)


class CAE2dFixed(CAE):
    def __init__(self, input_shape, encoded_size=500):
        encoder = EncoderFixed2d(input_shape, encoded_size=encoded_size)
        decoder = DecoderFixed2d(input_shape, encoded_size=encoded_size)
        super(CAE2dFixed, self).__init__(encoder, decoder)
