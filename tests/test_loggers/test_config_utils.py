import pytest
from pathlib import Path
from src.mytorch.io.utils import *

# Define a fixture for the tests
@pytest.fixture
def fixture_dict():
    return {
        "string": "This is a string with a {placeholder}",
        "key_to_placeholder": {"placeholder": "placeholder_value"},
        "integer": 123,
        "list": [1, 2, 3],
        "dict": {"key1": "value1", "key2": "value2"},
        "path": Path("/path/to/somewhere"),
        "nested_dict": {"key1": {"subkey1": "subvalue1"}, "key2": "value2"},
    }

def test_find_placeholders_in_string(fixture_dict):
    result = find_placeholders_in_string(fixture_dict["string"])
    assert result == ["placeholder"]

def test_read_toml(fixture_dict, tmp_path):
    # Create a temporary toml file for testing
    toml_file = tmp_path / "test.toml"
    toml_file.write_text("[section]\nkey = 'value'")
    result = read_toml(toml_file)
    assert result == {"section": {"key": "value"}}

def test_apply_to_dict(fixture_dict):
    func = lambda x, y: x.upper() if isinstance(x, str) else x
    result = apply_to_dict(fixture_dict["dict"], func=func)
    assert result == {"key1": "VALUE1", "key2": "VALUE2"}

def test_replace_placeholders(fixture_dict):
    result = replace_placeholders(fixture_dict["string"], fixture_dict)
    assert result == "This is a string with a placeholder_value"

def test_join_root_with_paths(fixture_dict):
    paths_dict = {"root": fixture_dict["path"], "other": "other/path"}
    result = join_root_with_paths(paths_dict)
    assert result == {"root": fixture_dict["path"], "other": fixture_dict["path"] / "other/path"}

def test_convert_path_dict_to_pathlib(fixture_dict):
    paths_dict = {"path1": "/path/to/somewhere", "path2": "/another/path"}
    result = convert_path_dict_to_pathlib(paths_dict)
    assert isinstance(result["path1"], Path)
    assert isinstance(result["path2"], Path)

def test_get_nested_dict_value(fixture_dict):
    result = get_nested_dict_value(fixture_dict["nested_dict"], ["key1", "subkey1"])
    assert result == "subvalue1"