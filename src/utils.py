def calculate_output_size(input_size, kernel_size, stride, padding):
    return (input_size - kernel_size + 2 * padding) // stride + 1


def calculate_output_size_transpose(input_size, kernel_size, stride, padding):
    return (input_size - 1) * stride - 2 * padding + kernel_size


def calculate_padding_transpose(output_size, input_size, kernel_size, stride):
    return ((input_size - 1) * stride - output_size + kernel_size) // 2


def calculate_output_size_rectangular(func, input_size, kernel_size, stride, padding):
    output = []
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(input_size, int):
        input_size = (input_size, input_size)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(stride, int):
        stride = (stride, stride)

    for i in range(2):
        output.append(func(input_size[i], kernel_size[i], stride[i], padding[i]))
    return output


def convolutional_output(input_size, kernel_size, stride, padding):
    return calculate_output_size_rectangular(calculate_output_size, input_size, kernel_size, stride, padding)

def convolutional_output_transpose(input_size, kernel_size, stride, padding):
    return calculate_output_size_rectangular(calculate_output_size_transpose, input_size, kernel_size, stride, padding)

def convolutional_padding_transpose(output_size, input_size, kernel_size, stride):
    return calculate_output_size_rectangular(calculate_padding_transpose, output_size, input_size, kernel_size, stride)



def repeated(func):
    def wrapper(*args, num_layers):
        input_shape = args[0]
        for i in range(num_layers):
            input_shape = func(input_shape, *args[1:])
        return input_shape
    return wrapper


@repeated
def convolutional_output_repeated(input_size, kernel_size, stride, padding):
    return convolutional_output(input_size, kernel_size, stride, padding)

@repeated
def convolutional_output_transpose_repeated(input_size, kernel_size, stride, padding):
    return convolutional_output_transpose(input_size, kernel_size, stride, padding)


def check_2d(shape):
    if isinstance(shape, int):
        return (shape, shape)
    elif isinstance(shape, tuple) and len(shape) == 2:
        return shape
    elif isinstance(shape, list) and len(shape) == 2:
        return shape
    else:
        raise ValueError('Shape must be an integer or a tuple or list of length 2.')

def test_conv_params(in_size, out_size, kernel_size, stride, padding):
    out = np.floor((in_size - kernel_size + 2*padding)/stride) + 1
    assert(out == out_size)

def get_stride_padding(in_shape, out_shape, kernel_size):
    in_shape = check_2d(in_shape)
    out_shape = check_2d(out_shape)
    kernel_size = check_2d(kernel_size)
    strides = []
    paddings = []
    for i in range(2):
        strides.append(in_shape[i]//out_shape[i])


        paddings.append(pad)
        strides.append(stride)

    return strides, paddings

def get_padding(out_shape, in_shape, kernel_size, stride):
    return ((out_shape - 1) * stride + kernel_size - in_shape) // 2 + 1

def get_output(in_shape, kernel_size, stride, padding):
    return (in_shape - kernel_size + 2*padding) // stride + 1

if __name__=="__main__":

    output = convolutional_output_repeated((100, 100), (3, 10),2, 1, num_layers=3)
    print(output)

