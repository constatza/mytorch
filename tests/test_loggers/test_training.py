import pytest
from mytorch.loggers.training import TrainLogger, ProgressLogger, CheckpointLogger, ProgressCheckpointLogger, train_logger_factory

def test_TrainLogger():
    logger = TrainLogger(history_file='history.log', error_file='error.log')
    assert isinstance(logger, TrainLogger)
    assert str(logger.history_file) == 'history.log'
    assert str(logger.error_file) == 'error.log'

def test_ProgressLogger():
    logger = ProgressLogger(history_file='history.log', error_file='error.log', print_every=5)
    assert isinstance(logger, ProgressLogger)
    assert logger.print_every == 5

def test_CheckpointLogger():
    logger = CheckpointLogger(history_file='history.log', error_file='error.log', model=None, checkpoint_path='checkpoint.pth')
    assert isinstance(logger, CheckpointLogger)
    assert logger.checkpoint_path == 'checkpoint.pth'

def test_ProgressCheckpointLogger():
    logger = ProgressCheckpointLogger(history_file='history.log', error_file='error.log', model=None, checkpoint_path='checkpoint.pth', print_every=5)
    assert isinstance(logger, ProgressCheckpointLogger)
    assert logger.print_every == 5
    assert logger.checkpoint_path == 'checkpoint.pth'

def test_train_logger_factory():
    logger = train_logger_factory('progress', history_file='history.log', error_file='error.log', print_every=5)
    assert isinstance(logger, ProgressLogger)

    logger = train_logger_factory('checkpoint', history_file='history.log', error_file='error.log', model=None, checkpoint_path='checkpoint.pth')
    assert isinstance(logger, CheckpointLogger)

    logger = train_logger_factory('progress_checkpoint', history_file='history.log', error_file='error.log', model=None, checkpoint_path='checkpoint.pth', print_every=5)
    assert isinstance(logger, ProgressCheckpointLogger)

    with pytest.raises(ValueError):
        train_logger_factory('invalid', history_file='history.log', error_file='error.log')