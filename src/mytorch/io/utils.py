import re
from functools import reduce
from pathlib import Path
from typing import Dict, Union, Any, Callable, Optional

import numpy as np
import torch
from pydantic import validate_call, FilePath
from tomlkit import parse

from mytorch.mytypes import PathLike, PathDict


@validate_call
def replace_placeholders_in_toml(string: str) -> str:
    """Uses re to replace placeholders in a toml string.
    Placeholder format is {section.key} where {'section.key': 'value'}"""
    mapping = parse(string)
    # set keys to snake case
    flattened = flatten_dict(mapping)
    final = interpolate_placeholders(string, flattened)
    # in keys replace hyphens with underscores
    return final.replace("//", "/")


@validate_call
def interpolate_placeholders(text: str, replacement_dict: Dict[str, Any]) -> str:
    """Interpolates placeholders in a string with values from a dictionary.
    The placeholders are marked with curly braces {}.
    Multiple matches are possible, so the function uses re to find all matches.
    Args:
    - text: The string to interpolate.
    - replacement_dict: The dictionary with the replacements.
    Returns:
    - str: The interpolated string.
    """
    pattern = re.compile(r"\{([^}]*)\}")
    matches = pattern.findall(text)
    # apply recursively until no more placeholders are found
    maximum_iterations = 4
    i = 0
    while matches and i < maximum_iterations:
        for match in matches:
            for key, value in replacement_dict.items():
                if match == key:
                    text = text.replace("{" + match + "}", str(value))

        matches = pattern.findall(text)
        i += 1
    if i == maximum_iterations:
        raise ValueError(f"Matches not found {matches}.")


    return text


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
    # root is contained in either the first or second level of the dictionary
    # find where the root is and join same-level paths with it

    key_root = reduce(lambda x, y: x if ("root" in x) else y, paths_dict)

    if "root" in key_root:
        root = paths_dict[key_root]
        joined_paths_dict = {
            k: root / v if k != key_root else v for k, v in paths_dict.items()
        }

    else:
        joined_paths_dict = paths_dict.copy()
        for key, subdict in paths_dict.items():
            key_root = reduce(lambda x, y: x if ("root" in x) else y, subdict)
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
def smart_load_tensors(path: FilePath, convolutional_dims: Optional[int], **kwargs):
    """Load with either numpy or torch depending on file extension."""
    if convolutional_dims is None:
        print("Assuming convolutional_dims=0 i.e. (N, M) shape")
        convolutional_dims = 0
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
