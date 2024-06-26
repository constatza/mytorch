
import pytest
from torch.nn import Module
from pathlib import Path

@pytest.fixture
def logger_data():
    return {
        'history_file': 'history.log',
        'error_file': 'error.log',
        'model': Module(),
        'checkpoint_path': 'checkpoint.pth',
        'print_every': 5
    }

@pytest.fixture
def utils_dict():
    return {
        "string": "This is a string with a {placeholder}",
        "key_to_placeholder": {"placeholder": "placeholder_value"},
        "integer": 123,
        "list": [1, 2, 3],
        "dict": {"key1": "value1", "key2": "value2"},
        "path": Path("/path/to/somewhere"),
        "nested_dict": {"key1": {"subkey1": "subvalue1"}, "key2": "value2"},
    }
