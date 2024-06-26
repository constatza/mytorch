import pytest

from mytorch.io.loggers import TrainLogger, ProgressLogger, CheckpointLogger, ProgressCheckpointLogger, \
    train_logger_factory


def test_TrainLogger(logger_data):
    logger = TrainLogger(**logger_data)
    assert isinstance(logger, TrainLogger)
    assert str(logger.history_file) == 'history.log'
    assert str(logger.error_file) == 'error.log'


def test_ProgressLogger(logger_data):
    logger = ProgressLogger(**logger_data)
    assert isinstance(logger, ProgressLogger)
    assert logger.print_every == 5


def test_CheckpointLogger(logger_data):
    logger = CheckpointLogger(**logger_data)
    assert isinstance(logger, CheckpointLogger)
    assert str(logger.checkpoint_path) == 'checkpoint.pth'


def test_ProgressCheckpointLogger(logger_data):
    logger = ProgressCheckpointLogger(**logger_data)
    assert isinstance(logger, ProgressCheckpointLogger)
    assert logger.print_every == 5
    assert str(logger.checkpoint_path) == 'checkpoint.pth'


def test_train_logger_factory(logger_data):
    logger = train_logger_factory('progress', **logger_data)
    assert isinstance(logger, ProgressLogger)

    logger = train_logger_factory('checkpoint', **logger_data)
    assert isinstance(logger, CheckpointLogger)

    logger = train_logger_factory('progress_checkpoint', **logger_data)
    assert isinstance(logger, ProgressCheckpointLogger)

    with pytest.raises(ValueError):
        train_logger_factory('invalid', **logger_data)
