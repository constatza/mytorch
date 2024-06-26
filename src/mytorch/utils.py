from collections.abc import Iterable
from math import ceil
import numpy as np

import torch
import inspect


def assure_iterable(arg):
    return arg if isinstance(arg, Iterable) else (arg, arg)


def conv_out(input_size, kernel_size, stride, padding):
    return (input_size - kernel_size + 2 * padding) // stride + 1


def conv_out_transpose(input_size, kernel_size, stride, padding):
    return (input_size - 1) * stride - 2 * padding + kernel_size


def conv_kernel(input_size, output_size, stride, padding):
    kernel = input_size - (output_size - 1) * stride + 2 * padding
    if kernel < 1:
        raise ValueError('Invalid input parameters.')
    return kernel

def calculate_padding(w_in, w_out, kernel_size, stride=1):
    padding = ((w_out - 1) * stride + kernel_size - w_in) / 2
    return int(padding)

def conv_kernel_transpose(input_size, output_size, stride, padding):
    return conv_kernel(output_size, input_size, stride, padding)


def conv_padding_half_out(input_size, kernel_size, stride, dilation=1, deconv=False):
    """
    :param input_size:       Input length (or width)
    :param kernel_size:     Kernel size (or width)
    :param stride:       Stride
    :param dilation:       Dilation Factor
    :return:        Returns padding such that output width is exactly half of input width
    """
    return ceil((stride * (input_size / 2) - input_size + dilation * (kernel_size - 1) - 1) / 2)


def conv_padding_double_out_tranpose(input_size, kernel_size, stride, dilation=1):
    """
    :param input_size:       Input length (or width)
    :param kernel_size:     Kernel size (or width)
    :param stride:       Stride
    :param dilation:       Dilation Factor
    :return:        Returns padding such that output width is exactly half of input width
    """
    pad = ceil(((input_size - 1) * stride + dilation * (kernel_size - 1) + 1 - input_size * 2) / 2)
    output_size = (input_size - 1) * stride - 2 * pad + dilation * (kernel_size - 1) + 1
    assert output_size == input_size * 2
    return pad


def atleast_2d(func):
    def wrapper(*args):
        args_2d = [assure_iterable(arg) for arg in args]
        lengths = [len(arg_2d) for arg_2d in args_2d]
        if len(set(lengths)) != 1:
            raise ValueError('All arguments must have the same length.')

        inputs = tuple(zip(*args_2d))
        results = tuple(func(*input) for input in inputs)

        if isinstance(results[0], Iterable):
            return tuple(zip(*results))
        else:
            return results

    return wrapper


conv_output_2d = atleast_2d(conv_out)
conv_output_transpose_2d = atleast_2d(conv_out_transpose)


def vectorized(func):
    def wrapper(*args, which=0, **kwargs):
        """Applies the arguments of the function that are iterable in succession.
        All other arguments are kept constant. The cartesian product of the iterable and
        constant arguments is taken as input to the function calls"""
        iterables = {i: iter(arg) for i, arg in enumerate(args) if isinstance(arg, Iterable)}
        max_len = max(len(iterable) for iterable in args if isinstance(iterable, Iterable))
        num_args = len(args)
        current_args = args
        outputs = []
        for i in range(max_len):
            current_args = [next(iterables[i]) if i in iterables else current_args[i] for i in range(num_args)]
            current_args[which] = func(*current_args, **kwargs)
            outputs.append(current_args[which])
        return tuple(outputs)

    return wrapper


conv_out_vect = vectorized(conv_out)
conv_out_transpose_vect = vectorized(conv_out_transpose)


def recursive_apply(func):
    def wrapper(*args, num_reps=1, which=0, **kwargs):
        """recursively apply a function to the output of the previous function
        specified by which"""
        new_value = func(*args, **kwargs)
        if num_reps == 1:
            return new_value
        else:
            new_args = list(args)
            new_args[which] = new_value
            return wrapper(*new_args, num_reps=num_reps - 1, **kwargs)

    return wrapper


@recursive_apply
def conv_out_repeated(input_size, kernel_size, stride, padding, num_reps=1, which=0):
    return conv_out(input_size, kernel_size, stride, padding)


@recursive_apply
def conv_out_repeated_2d(input_size, kernel_size, stride, padding, num_reps=1, which=0):
    return conv_output_2d(input_size, kernel_size, stride, padding)


@recursive_apply
def conv_out_transpose_repeated(input_size, kernel_size, stride, padding, num_reps=1, which=0):
    return conv_out_transpose(input_size, kernel_size, stride, padding)


@recursive_apply
def conv_out_transpose_repeated_2d(input_size, kernel_size, stride, padding, num_reps=1, which=0):
    return conv_output_transpose_2d(input_size, kernel_size, stride, padding)




if __name__ == "__main__":
    pass
    x = conv_out(100, 3, 2, 1)
    print(x)
