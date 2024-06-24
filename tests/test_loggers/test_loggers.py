import pytest
from torch.nn import Module
from mytorch.config.loggers import TrainLogger, ProgressLogger, CheckpointLogger, ProgressCheckpointLogger, train_logger_factory


@pytest.fixture
def data():
    return {
        'history_file': 'history.log',
        'error_file': 'error.log',
        'model': Module(),
        'checkpoint_path': 'checkpoint.pth',
        'print_every': 5
    }


def test_TrainLogger(data):
    logger = TrainLogger(**data)
    assert isinstance(logger, TrainLogger)
    assert str(logger.history_file) == 'history.log'
    assert str(logger.error_file) == 'error.log'

def test_ProgressLogger(data):
    logger = ProgressLogger(**data)
    assert isinstance(logger, ProgressLogger)
    assert logger.print_every == 5

def test_CheckpointLogger(data):
    logger = CheckpointLogger(**data)
    assert isinstance(logger, CheckpointLogger)
    assert str(logger.checkpoint_path) == 'checkpoint.pth'

def test_ProgressCheckpointLogger(data):
    logger = ProgressCheckpointLogger(**data)
    assert isinstance(logger, ProgressCheckpointLogger)
    assert logger.print_every == 5
    assert str(logger.checkpoint_path) == 'checkpoint.pth'

def test_train_logger_factory(data):
    logger = train_logger_factory('progress', **data)
    assert isinstance(logger, ProgressLogger)

    logger = train_logger_factory('checkpoint', **data)
    assert isinstance(logger, CheckpointLogger)

    logger = train_logger_factory('progress_checkpoint', **data)
    assert isinstance(logger, ProgressCheckpointLogger)

    with pytest.raises(ValueError):
        train_logger_factory('invalid', **data)