import pytest

from mytorch.io.utils import (
    replace_placeholders_in_toml,
    interpolate_placeholders,
    flatten_dict,
)


def test_replace_placeholders_in_toml(toml_string):
    result = replace_placeholders_in_toml(toml_string)
    assert result.splitlines()[-1] == "key2 = 'value'"


@pytest.mark.parametrize(
    "text, replacement_dict, expected_output",
    [
        ("Hello {name}", {"name": "World"}, "Hello World"),
        (
            "{greeting}, {name}!",
            {"greeting": "Hello", "name": "World"},
            "Hello, World!",
        ),
        ("{a} + {b} = {c}", {"a": 1, "b": 2, "c": 3}, "1 + 2 = 3"),
        ("{a} + {b} = {c}", {"a": 1, "b": 2}, ValueError),
    ],
)
def test_interpolate_placeholders(text, replacement_dict, expected_output):
    if expected_output == ValueError:
        with pytest.raises(ValueError):
            interpolate_placeholders(text, replacement_dict)
    else:
        assert interpolate_placeholders(text, replacement_dict) == expected_output


@pytest.mark.parametrize(
    "d, expected_output",
    [
        ({"a": 1, "b": {"c": 2}}, {"a": 1, "b.c": 2}),
        ({"a": {"b": {"c": 3}}}, {"a.b.c": 3}),
        ({"a": {"b": 2}, "c": 3}, {"a.b": 2, "c": 3}),
    ],
)
def test_flatten_dict_flattens(d, expected_output):
    assert flatten_dict(d) == expected_output
