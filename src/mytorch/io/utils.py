import re
from typing import Dict, Any

from pydantic import validate_call
from tomlkit import parse


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





