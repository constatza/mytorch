import re
from abc import ABC, abstractmethod

import tomli


class Parser(ABC):
    """Base class for configuration file parsers."""

    def __init__(self, config):
        if isinstance(config, str):
            with open(config, 'rb') as file:
                self.parse(file)
        elif isinstance(config, dict):
            self.config = config
        else:
            raise TypeError('config must be either a file path (str) or a dictionary (dict)')

    @abstractmethod
    def parse(self, file):
        """Parses a configuration file."""
        pass

    def get(self, name):
        # convert _ to - in the name
        name = name.replace('_', '-')
        try:
            return self.config[name]
        except KeyError:
            raise AttributeError(f"Attribute {name} not found in configuration file.")


class TOMLParser(Parser):
    """Parser for TOML configuration files."""

    def __init__(self, config):
        super().__init__(config)
        self.replace_placeholders_in_dict(self.config)

    def parse(self, file):
        self.config = tomli.load(file)

    def join_dirs_with_file(self):
        """Joins directories with files in the configuration file.
        Dirs end with a slash, files don't.
        """
        for key, value in self.config.items():
            if isinstance(value, dict):
                self.config[key] = TOMLParser.from_dict(value)
            elif isinstance(value, str):
                self.config[key] = self.join_dir_with_file(value)

    def replace_placeholders_in_dict(self, d, keys=None):
        if keys is None:
            keys = []
        for key, value in d.items():
            new_keys = keys + [key]
            if isinstance(value, str):
                placeholders = find_placeholders_in_fstring(value)
                for placeholder in placeholders:
                    placeholder_keys = placeholder.split('.')
                    placeholder_value = get_nested_dict_value(self.config, placeholder_keys)
                    if placeholder_value is not None:
                        value = value.replace('{' + placeholder + '}', placeholder_value)
                        set_nested_dict_value(self.config, new_keys, value)
            elif isinstance(value, dict):
                self.replace_placeholders_in_dict(value, new_keys)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, str):
                        placeholders = find_placeholders_in_fstring(item)
                        for placeholder in placeholders:
                            placeholder_keys = placeholder.split('.')
                            placeholder_value = get_nested_dict_value(d, placeholder_keys)
                            if placeholder_value is not None:
                                item = item.replace('{' + placeholder + '}', placeholder_value)
                                set_nested_dict_value(d, new_keys + [i], item)
                    elif isinstance(item, dict):
                        self.replace_placeholders_in_dict(item, new_keys + [i])


def find_placeholders_in_fstring(fstring):
    """Finds the placeholder in a string."""
    return re.findall(r'\{([^}]*)\}', fstring)


def set_nested_dict_value(d, keys, value):
    if len(keys) > 1:
        key = keys.pop(0)
        set_nested_dict_value(d[key], keys, value)
    else:
        d[keys[0]] = value


def get_nested_dict_value(d, keys):
    if len(keys) > 1:
        key = keys.pop(0)
        return get_nested_dict_value(d[key], keys)
    else:
        return d.get(keys[0])


if __name__ == "__main__":
    config = {
        'model': {
            'name': 'TestModel',
            'path': '{model.name}/path'
        },
        'data': {
            'dir': 'data',
            'file': '{data.dir}/file'
        }
    }
    parser = TOMLParser(config)
    print(parser.config)
