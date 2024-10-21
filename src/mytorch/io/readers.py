from collections import namedtuple
from typing import Dict, Any, List

import numpy as np
import torch
from pydantic import validate_call, FilePath
from pathlib import Path

import re
import tomlkit


def check_paths(paths_dict: dict) -> None:
    for key, path_name in paths_dict.items():
        path = Path(path_name)
        if path.is_file() and not path.exists():
            raise FileNotFoundError(f"{key} path not found: {path_name}")
        elif path.is_dir():
            path.mkdir(parents=True, exist_ok=True)


@validate_call
def read_toml(config_path: FilePath) -> Dict:
    """Reads a toml configuration file."""
    with open(config_path, "r") as file:
        content = file.read()
    parsed = parse_self_referencing_toml(content)
    return parsed


def parse_self_referencing_toml(toml_string):
    """
    Parses a TOML string and resolves self-references in the format {table.key}.

    Args:
        toml_string (str): The TOML input string with potential self-references.

    Returns:
        dict: The resolved TOML data as a dictionary.
    """
    data = tomlkit.parse(toml_string)
    resolving = set()  # To track and prevent circular references
    pattern = re.compile(r"\{(.+?)\}")  # Regex pattern to find {table.key}

    def resolve(value):
        """Recursively resolve a single value (either string or other data types)."""
        if isinstance(value, str):
            matches = pattern.findall(value)
            for match in matches:
                table, key = match.split(".")
                # Recursive resolution of reference
                if table in data and key in data[table]:
                    if (table, key) in resolving:
                        raise ValueError(
                            f"Circular reference detected in {table}.{key}"
                        )
                    # Track the resolving process to prevent circular references
                    resolving.add((table, key))
                    referenced_value = resolve(data[table][key])
                    resolving.remove((table, key))

                    # Replace placeholder with the actual resolved value
                    value = value.replace(f"{{{match}}}", str(referenced_value))
        return value

    def resolve_all():
        """Recursively resolve all references in the TOML data."""
        for table, contents in data.items():
            for key, value in contents.items():
                data[table][key] = resolve(value)

    # Resolve all self-references
    resolve_all()

    return data


@validate_call
def load_config(config_path: FilePath) -> dict:
    config: dict = read_toml(config_path)
    check_paths(config["paths"])
    return config


@validate_call
def read_array_as_numpy(path: FilePath):
    if path.suffix == ".npy":
        return np.load(path)
    elif path.suffix == ".csv":
        return np.loadtxt(path, delimiter=",")
    elif path.suffix == ".pt":
        return torch.load(path).numpy()
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")
