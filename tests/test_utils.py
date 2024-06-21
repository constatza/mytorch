import pytest
from mytorch.utils import atleast_2d, recursive_apply
from mytorch.utils import conv_out, conv_out_repeated, conv_out_repeated_2d
from mytorch.utils import conv_out_transpose, conv_out_transpose
from mytorch.utils import conv_kernel, conv_kernel_transpose
from mytorch.utils import conv_out_vect, conv_out_transpose_vect

@pytest.mark.parametrize('args',[
    (10, 3, 2, 1, 5),
    (8, 3, 2, 1, 4),
    (7, 2, 2, 0, 3),
    (5, 3, 1, 0, 3),
    (6, 4, 1, 1, 5),
])
def test_conv_out(args):
    input_size, kernel_size, stride, padding, expected_output_size = args
    assert conv_out(input_size, kernel_size, stride, padding) == expected_output_size

@pytest.mark.parametrize('args',[
    (10, 3, 2, 1, 19),
    (8, 3, 2, 1, 15),
    (7, 2, 2, 0, 14),
    (6, 4, 1, 1, 7),
    (5, 3, 1, 0, 7),
])
def test_conv_out_transpose(args):
    input_size, kernel_size, stride, padding, expected_output_size = args
    assert conv_out_transpose(input_size, kernel_size, stride, padding) == expected_output_size

@pytest.mark.parametrize('args',[
    (64, 64, 1, 1, 3),
    (10, 5, 2, 1, 4),
    (8, 5, 2, 1, 2),
    (6, 5, 1, 1, 4),
])
def test_conv_kernel(args):
    """Test the conv_kernel function"""
    input_size, output_size, stride, padding, expected_output = args
    assert conv_kernel(input_size, output_size, stride, padding) == expected_output

# get kernel for transpose convolution
@pytest.mark.parametrize('args',[
    (64, 64, 1, 1, 3),
    (16, 32, 2, 1, 4),
    (8, 16, 2, 1, 4),
])
def test_conv_kernel_transpose(args):
    """Test the transpose convolutional kernel function"""
    input_size, output_size, stride, padding, expected_output = args
    assert conv_kernel_transpose(input_size, output_size, stride, padding) == expected_output

@pytest.mark.parametrize('args, expected_output',[
    [(1, 2), ((1,1), (2, 2))],
    [(1, 2, 3), ((1, 1), (2, 2), (3, 3))],
    [(1, (2, 3)), ((1,1), (2, 3))]
])
def test_atleast_2d(args, expected_output):
    wrapped = atleast_2d(lambda *x: x)
    assert wrapped(*args) == expected_output


def test_recursive_apply(request):
    func = lambda x, y: x + y
    wrapped = recursive_apply(func)
    assert wrapped(1, 1, num_reps=3, which=0) == 4

@pytest.mark.parametrize('args, expected_output',[
    [(100, 3, 1, 1, 1), 100],
    [(100, 3, 1, 1, 2), 100],
    [(100, 3, 2, 1, 2), 25],
    [(100, 3, 2, 1, 3), 13],
])
def test_conv_output_repeated(args, expected_output):
    input_size, kernel_size, stride, padding, num_reps = args
    assert conv_out_repeated(input_size, kernel_size, stride, padding, num_reps=num_reps) == expected_output


def test_conv_output_repeated_2d():
    input_size, kernel_size, stride, padding, num_reps = 100, 3, 1, 1, 1
    assert conv_out_repeated_2d(input_size, kernel_size, stride, padding, num_reps=num_reps) == (100, 100)


def test_conv_out_vect():
    padding = [1, 1, 0]
    kernel =  [3, 3, 3]
    assert conv_out_vect(10, kernel, 1, padding, which=0) == (10, 10, 8)

