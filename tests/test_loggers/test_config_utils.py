import io

import pytest
from mytorch.config.utils import (find_placeholders_in_string, replace_placeholders_in_dict,
                                   get_nested_dict_value, join_root_with_paths, convert_path_dict_to_pathlib,
                                   set_nested_dict_value)
from pathlib import Path
from pydantic import FilePath

@pytest.fixture
def setup_data():
    return {
        "fstring": "This is a {test} string with an {example}",
        "placeholder_dict": {"key": "value", "placeholder": "{key}"},
        "dict": {"key": {"subkey": "value"}},
        "paths_dict": {"root": "root/path", "file": "file.txt"},
        "path_dict": {"file": "/path/to/file"}
    }

def test_find_placeholders_in_string(setup_data):
    assert find_placeholders_in_string(setup_data["fstring"]) == ["test", "example"]


def test_convert_path_dict_to_pathlib(setup_data):
    assert convert_path_dict_to_pathlib(setup_data["path_dict"]) == {"file": Path("/path/to/file")}


def test_replace_placeholders_in_dict(setup_data):
    assert replace_placeholders_in_dict(setup_data["placeholder_dict"]) == {"key": "value", "placeholder": "value"}


def test_get_nested_dict_value(setup_data):
    assert get_nested_dict_value(setup_data["dict"], ["key", "subkey"]) == "value"


def test_join_root_with_paths(setup_data):
    assert join_root_with_paths(setup_data["paths_dict"])['file'] == Path("root/path/file.txt")


def test_set_nested_dict_value(setup_data):
    set_nested_dict_value(setup_data["dict"], ["key", "subkey"], "new_value")
    assert setup_data["dict"] == {"key": {"subkey": "new_value"}}