import pytest
from parsers import TOMLParser

@pytest.fixture
def parser():
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
    return TOMLParser(config)

def test_format_variables_in_fstrings(parser):
    assert parser.config['model']['path'] == 'TestModel/path'
    assert parser.config['data']['file'] == 'data/file'

def test_attributes_workins(parser):
    assert isinstance(parser.data, TOMLParser)
    assert parser.model.name == 'TestModel'