from pathlib import Path

import pytest
from torch.nn import Module


@pytest.fixture
def logger_data():
    return {
        "history_file": "history.log",
        "error_file": "error.log",
        "model": Module(),
        "checkpoint_path": "checkpoint.pth",
        "print_every": 5,
    }


@pytest.fixture
def utils_dict():
    return {
        "integer": 123,
        "list": [1, 2, 3],
        "dict": {"key1": "value1", "key2": "value2"},
        "path": Path("/path/to/somewhere"),
        "nested_dict": {"key1": {"subkey1": "subvalue1"}, "key2": "value2"},
    }


@pytest.fixture
def toml_string():
    toml_string = """
    [section.subsection]
    key = 'value'
    key2 = '{section.subsection.key}'
    """
    return "\n".join([line.lstrip() for line in toml_string.splitlines()])
