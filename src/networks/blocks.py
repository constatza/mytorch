import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Residual Block for 1d Convolutional Layers. The output shape is in general
    different from the input shape, so a linear convolution layer is used to match the shapes."""

    def __init__(self, shortcut, block, activation=nn.Identity()):
        super(ResidualBlock, self).__init__()
        self.block = block
        self.shortcut = shortcut
        self.activation = activation

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.block(x)
        out = out + residual
        out = self.activation(out)
        return out

    @staticmethod
    def shortcut_conv_2d(in_channels, out_channels, input_shape, output_shape):
        if in_channels != out_channels:
            # apply formula to all elements of the tuple
            stride = []
            kernel_size = []
            for i in range(2):
                stride.append(input_shape[i] // output_shape[i])
                kernel_size.append(input_shape[i] - (output_shape[i] - 1) * stride[i])
            padding = 0
            return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            return nn.Identity()

    @staticmethod
    def shortcut_conv_transpose_2d(in_channels, out_channels, input_shape, output_shape):
        if in_channels != out_channels:
            stride = []
            kernel_size = []
            for i in range(2):
                stride.append(output_shape[i] // input_shape[i])
                kernel_size.append(output_shape[i] - (input_shape[i] - 1) * stride[i])
            padding = 0
            return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding)
        else:
            return nn.Identity()

    @staticmethod
    def shortcut_conv_1d(in_channels, out_channels, in_shape, out_shape):
        if in_channels != out_channels:
            stride = in_shape // out_shape
            kernel_size = in_shape - (out_shape - 1) * stride
            padding = 0
            return nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            return nn.Identity()

    @staticmethod
    def shortcut_conv_transpose_1d(in_channels, out_channels, input_shape, output_shape):
        if input_shape != output_shape:
            stride = output_shape // input_shape
            kernel_size = output_shape - (input_shape - 1) * stride
            padding = 0
            return nn.ConvTranspose1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding)
        else:
            return nn.Identity()

    @staticmethod
    def shortcut_linear(input_size, output_size):
        if input_size != output_size:
            return nn.Linear(input_size, output_size)
        else:
            return nn.Identity


class ResidualBlock1d(ResidualBlock):
    """Residual Block for 1d Convolutional Layers. The output shape is in general
    different from the input shape, so a linear convolution layer is used to match the shapes."""

    def __init__(self, block, in_channels, out_channels, in_shape, out_shape, activation=nn.Identity()):
        shortcut = ResidualBlock.shortcut_conv_1d(in_channels, out_channels, in_shape, out_shape)
        super(ResidualBlock1d, self).__init__(shortcut, block, activation)


class ResidualBlock2d(ResidualBlock):
    """Residual Block for 2d Convolutional Layers. The output shape is in general
    different from the input shape, so a linear convolution layer is used to match the shapes."""

    def __init__(self, block, in_channels, out_channels, input_shape, output_shape, activation=nn.Identity()):
        shortcut = ResidualBlock.shortcut_conv_2d(in_channels, out_channels, input_shape, output_shape)
        super(ResidualBlock2d, self).__init__(shortcut, block, activation)


class ResidualBlockTranspose2d(ResidualBlock):
    """Residual Block for Transposed 2d Convolutional Layers. The output shape is in general
    different from the input shape, so a linear convolution layer is used to match the shapes."""

    def __init__(self, block, in_channels, out_channels, input_shape, output_shape, activation=nn.Identity()):
        shortcut = ResidualBlock.shortcut_conv_transpose_2d(in_channels, out_channels, input_shape, output_shape)
        super(ResidualBlockTranspose2d, self).__init__(shortcut, block, activation)


class ResidualBlockLinear(ResidualBlock):
    """Residual Block for Linear Layers. The output shape is in general
    different from the input shape, so a linear convolution layer is used to match the shapes."""

    def __init__(self, block, input_size, output_size, activation=nn.Identity()):
        shortcut = ResidualBlock.shortcut_linear(input_size, output_size)
        super(ResidualBlockLinear, self).__init__(shortcut, block, activation)


class ResidualBlockTranspose1d(ResidualBlock):
    """Residual Block for Transposed 1d Convolutional Layers. The output shape is in general
    different from the input shape, so a linear convolution layer is used to match the shapes."""

    def __init__(self, block, in_channels, out_channels, input_shape, output_shape, activation=nn.Identity()):
        shortcut = ResidualBlock.shortcut_conv_transpose_1d(in_channels, out_channels, input_shape, output_shape)
        super(ResidualBlockTranspose1d, self).__init__(shortcut, block, activation)


class Shape(nn.Module):
    def __init__(self, text='Shape'):
        super(Shape, self).__init__()
        self.text = text

    def forward(self, x):
        with torch.no_grad():
            print(f'{self.text}: {list(x.shape)}')
            return x
