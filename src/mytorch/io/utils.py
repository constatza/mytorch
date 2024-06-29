import re
from pathlib import Path
from typing import Dict, Union, Any, Callable

import numpy as np
import torch
from pydantic import validate_call, FilePath, DirectoryPath
from tomlkit import parse

type PathLike = FilePath | DirectoryPath | Path
type PathDict = Dict[str, PathLike | PathDict]


@validate_call
def replace_placeholders_in_toml(string: str) -> str:
    """Uses re to replace placeholders in a toml string.
    Placeholder format is {section.key} where {'section.key': 'value'}"""
    mapping = parse(string)
    flattened = flatten_dict(mapping)
    final = interpolate_placeholders(string, flattened)
    return final


@validate_call
def interpolate_placeholders(text: str, replacement_dict: Dict[str, Any]):
    # Define a pattern that matches '{word}'
    pattern = r"\{(.*)\}"

    # Function to perform replacement
    def replace_with_dict(match):
        key = match.group(1)
        if key in replacement_dict:
            return replacement_dict[key]
        else:
            raise ValueError(f"Key {key} not found in configuration.")

    # Perform substitution using re.sub() with a function
    return re.sub(pattern, replace_with_dict, text)


@validate_call
def apply_to_dict(original_dict: Dict, func: Callable) -> Dict:
    """Replaces placeholders marked from {} in a dictionary in a recursive way."""
    for key, value in original_dict.items():
        if isinstance(value, Dict):
            original_dict[key] = apply_to_dict(value, func)
        elif isinstance(value, str):
            value = func(value, original_dict)
            original_dict[key] = value
    return original_dict


@validate_call
def flatten_dict(d: Dict, parent_key: str = "", sep: str = ".") -> Dict:
    """
    Flatten a nested dictionary while retaining keys.

    Args:
    - d: The input dictionary to flatten.
    - parent_key (optional): The parent key for recursion. Default is ''.
    - sep (optional): The separator between keys. Default is '.'.

    Returns:
    - dict: The flattened dictionary.
    """
    flattened = {}
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, Dict):
            flattened.update(flatten_dict(v, parent_key=new_key, sep=sep).items())
        else:
            flattened[new_key] = v
    return flattened


@validate_call
def join_root_with_paths(paths_dict: Dict) -> Dict:
    """Joins the root path with the paths in the dictionary."""
    paths_dict = convert_path_dict_to_pathlib(paths_dict)
    # join all root paths with the same-level relative paths by using
    # apply_to_dict
    key_root = "root-dir"
    if key_root in paths_dict:
        root = paths_dict[key_root]
        joined_paths_dict = {
            k: root / v if k != key_root else v for k, v in paths_dict.items()
        }

    else:
        joined_paths_dict = paths_dict.copy()
        for key, subdict in paths_dict.items():
            root = subdict[key_root]
            joined_paths_dict[key] = {
                k: root / v if k != key_root else v for k, v in subdict.items()
            }

    return joined_paths_dict


@validate_call
def convert_path_dict_to_pathlib(d: Dict) -> PathDict:
    """Converts paths in a dictionary to pathlib.Path objects."""
    return recursively_apply_to_dict(d, lambda x: Path(x))


@validate_call
def find_placeholders_in_string(fstring: str):
    """Finds the placeholder in a string."""
    return re.findall(r"\{([^}]*)\}", fstring)


@validate_call
def recursively_apply_to_dict(
    d: PathDict, func: Callable[[PathLike], PathLike]
) -> PathDict:
    for k, v in d.items():
        if isinstance(v, Dict):
            recursively_apply_to_dict(v, func)
        else:
            d[k] = func(v)
    return d


@validate_call
def get_nested_dict_value(d: Dict, keys: Union[list, tuple], default="") -> Any:
    """Gets a value from a nested dictionary."""
    if len(keys) > 1:
        key = keys.pop(0)
        return get_nested_dict_value(d.get(key, {}), keys)
    else:
        return d.get(keys[0], default)


@validate_call
def smart_load_tensors(path: FilePath, convolutional_dims: int, **kwargs):
    """Load with either numpy or torch depending on file extension."""
    if path.suffix == ".pt":
        tensor = torch.load(path, **kwargs)
    elif path.suffix == ".npy":
        tensor = np.load(path, **kwargs)
    elif path.suffix == ".csv":
        tensor = np.loadtxt(path, delimiter=",", **kwargs)
    else:
        raise ValueError("Invalid file extension. Must be either .npy or .pt.")

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
    return shape[-convolution_dims - 1 :]
