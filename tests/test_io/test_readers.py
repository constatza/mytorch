from mytorch.io.readers import read_toml


def test_read_toml(utils_dict, tmp_path):
    # Create a temporary toml file for testing
    toml_file = tmp_path / "test.toml"
    toml_file.write_text("[section]\nkey = 'value'")
    result = read_toml(toml_file)
    assert result == {"section": {"key": "value"}}
