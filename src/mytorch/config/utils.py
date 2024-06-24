import re
from pathlib import Path
from pydantic import validate_call, FilePath, DirectoryPath
import tomli

from typing import Dict, Union, ForwardRef, Any, Callable, TypeAlias
from pydantic import BaseModel


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

def replace_placeholders_in_dict(original_dict: dict) -> dict:
    """Replaces placeholders marked from {} in a dictionary in a recursive way."""
    for key, value in original_dict.items():
        if isinstance(value, dict):
            original_dict[key] = replace_placeholders_in_dict(value)
        elif isinstance(value, str):
            placeholders = find_placeholders_in_string(value)
            for placeholder in placeholders:
                value = value.replace(f"{{{placeholder}}}", original_dict.get(placeholder, ""))
            original_dict[key] = value
    return original_dict


@validate_call
def get_nested_dict_value(d: dict, keys: Union[list, tuple]) -> Any:
    """Gets a value from a nested dictionary."""
    if len(keys) > 1:
        key = keys.pop(0)
        return get_nested_dict_value(d.get(key, {}), keys)
    else:
        return d.get(keys[0])

def join_root_with_paths(paths_dict:  dict) -> dict:
    """Joins the root path with the paths in the dictionary."""
    paths_dict = convert_path_dict_to_pathlib(paths_dict)
    root = paths_dict['root']
    return {k: root / v if k != 'root' else v for k, v in paths_dict.items()}


def convert_path_dict_to_pathlib(d: dict) -> dict:
    """Converts paths in a dictionary to pathlib.Path objects."""
    return recursively_apply_to_dict(d, lambda x: Path(x) if isinstance(x, str) else x)



@validate_call
def find_placeholders_in_string(fstring: str):
    """Finds the placeholder in a string."""
    return re.findall(r'\{([^}]*)\}', fstring)


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