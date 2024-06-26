import re
import tomli
import torch
import inspect
import numpy as np
from pathlib import Path
from pydantic import validate_call, FilePath, DirectoryPath

from typing import Dict, Union, ForwardRef, Any, Callable, TypeAlias
from functools import partial


type PathLike = FilePath | DirectoryPath | Path
type PathDict = Dict[str, PathLike | PathDict]


@validate_call
def find_placeholders_in_string(string: str):
    """Finds the placeholder in a string."""
    return re.findall(r'\{([^}]*)\}', string)


@validate_call
def read_toml(config_path: FilePath):
    """Reads a toml configuration file."""
    with open(config_path, 'rb') as file:
        dictionary = tomli.load(file)
    return dictionary

def apply_to_dict(original_dict: dict, func: Callable) -> dict:
    """Replaces placeholders marked from {} in a dictionary in a recursive way."""
    for key, value in original_dict.items():
        if isinstance(value, dict):
            original_dict[key] = apply_to_dict(value, func)
        elif isinstance(value, str):
            value = func(value, original_dict)
            original_dict[key] = value
    return original_dict

def replace_placeholders(string: str, original_dict: str):
    placeholders = find_placeholders_in_string(string)
    for placeholder in placeholders:
        string = string.replace(f"{{{placeholder}}}",
                                find_in_dict(original_dict, placeholder))
    return string


def join_root_with_paths(paths_dict:  dict) -> dict:
    """Joins the root path with the paths in the dictionary."""
    paths_dict = convert_path_dict_to_pathlib(paths_dict)
    # join all root paths with the same-level relative paths by using
    # apply_to_dict
    if 'root' in paths_dict:
        root = paths_dict['root']
        joined_paths_dict = {k: root / v if k != 'root' else v for k, v in paths_dict.items()}

    else:
        joined_paths_dict = paths_dict.copy()
        for key, subdict in paths_dict.items():
            root = subdict['root']
            joined_paths_dict[key] = {k: root / v if k != 'root' else v for k, v in subdict.items()}

    return joined_paths_dict




def convert_path_dict_to_pathlib(d: dict) -> PathDict:
    """Converts paths in a dictionary to pathlib.Path objects."""
    return recursively_apply_to_dict(d, lambda x: Path(x))



@validate_call
def find_placeholders_in_string(fstring: str):
    """Finds the placeholder in a string."""
    return re.findall(r'\{([^}]*)\}', fstring)

replace_placeholders_in_dict = partial(apply_to_dict, func=replace_placeholders)

def set_nested_dict_value(d, keys, value):
    if len(keys) > 1:
        key = keys.pop(0)
        set_nested_dict_value(d[key], keys, value)
    else:
        d[keys[0]] = value
        return d


@validate_call
def recursively_apply_to_dict(d: PathDict,
                              func: Callable[[PathLike], PathLike]) -> PathDict:
    for k, v in d.items():
        if isinstance(v, dict):
            recursively_apply_to_dict(v, func)
        else:
            d[k] = func(v)
    return d

@validate_call
def get_nested_dict_value(d: dict, keys: Union[list, tuple], default="") -> Any:
    """Gets a value from a nested dictionary."""
    if len(keys) > 1:
        key = keys.pop(0)
        return get_nested_dict_value(d.get(key, {}), keys)
    else:
        return d.get(keys[0], default)


@validate_call
def find_in_dict(d: dict, subkey: str, default="") -> Any:
    """
    Searches whole dict and returns the first value that matches the subkey.
    """
    for key, value in d.items():
        if key == subkey:
            return value
        elif isinstance(value, dict):
            result = find_in_dict(value, subkey)
            if result:
                return result
    return default
def to_tensor(func):
    """Convert numpy array to tensor."""

    def wrapper(*args, **kwargs):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        new_args = tuple(torch.from_numpy(a).to(device) if not isinstance(a, torch.Tensor) else a for a in args)
        return func(*new_args, **kwargs)

    return wrapper

def smart_load_tensors(path, convolutional_dims, **kwargs):
    """Load with either numpy or torch depending on file extension."""
    if path.endswith('.pt'):
        tensor = torch.load(path, **kwargs)
    elif path.endswith('.npy'):
        tensor = np.load(path, **kwargs)
    elif path.endswith('.csv'):
        tensor = np.loadtxt(path, delimiter=',', **kwargs)
    else:
        raise ValueError('Invalid file extension. Must be either .npy or .pt.')

    if isinstance(tensor, np.ndarray):
        # Sometimes even files with .pt extension are loaded as numpy arrays
        tensor = torch.from_numpy(tensor).float()

    return shape_correction(tensor, convolutional_dims)

def shape_correction(tensor, convolution_dims):
    """
    Correction in order to be in pytorch shape
    (N, C, L) or (N, C, H, W)

    Args:
        tensor:
        convolution_dims:

    Returns:

    """
    shape_dims = len(tensor.shape)
    if shape_dims > convolution_dims + 2:
        tensor = tensor.squeeze()
    shape_dims = len(tensor.shape)
    if shape_dims > 2:
        assert shape_dims == convolution_dims + 2
    return tensor

def get_proper_convolution_shape(shape, convolution_dims):
    return shape[-convolution_dims-1:]

def filtered(func):
    def wrapper(*args, **kwargs):
        params = inspect.signature(func).parameters
        filtered = {k: v for k, v in kwargs.items() if k in params}
        return func(*args, **filtered)

    return wrapper
