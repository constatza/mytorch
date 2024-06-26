from mytorch.io.utils import *


def test_find_placeholders_in_string(utils_dict):
    result = find_placeholders_in_string(utils_dict["string"])
    assert result == ["placeholder"]


def test_read_toml(utils_dict, tmp_path):
    # Create a temporary toml file for testing
    toml_file = tmp_path / "test.toml"
    toml_file.write_text("[section]\nkey = 'value'")
    result = read_toml(toml_file)
    assert result == {"section": {"key": "value"}}


def test_apply_to_dict(utils_dict):
    func = lambda x, y: x.upper() if isinstance(x, str) else x
    result = apply_to_dict(utils_dict["dict"], func=func)
    assert result == {"key1": "VALUE1", "key2": "VALUE2"}


def test_replace_placeholders(utils_dict):
    result = replace_placeholders(utils_dict["string"], utils_dict)
    assert result == "This is a string with a placeholder_value"


def test_join_root_with_paths(utils_dict):
    paths_dict = {"root": utils_dict["path"], "other": "other/path"}
    result = join_root_with_paths(paths_dict)
    assert result == {"root": utils_dict["path"], "other": utils_dict["path"] / "other/path"}


def test_convert_path_dict_to_pathlib(utils_dict):
    paths_dict = {"path1": "/path/to/somewhere", "path2": "/another/path"}
    result = convert_path_dict_to_pathlib(paths_dict)
    assert isinstance(result["path1"], Path)
    assert isinstance(result["path2"], Path)


def test_get_nested_dict_value(utils_dict):
    result = get_nested_dict_value(utils_dict["nested_dict"], ["key1", "subkey1"])
    assert result == "subvalue1"
