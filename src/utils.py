from collections.abc import Iterable


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


def conv_kernel_transpose(input_size, output_size, stride, padding):
    return conv_kernel(output_size, input_size, stride, padding)


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
