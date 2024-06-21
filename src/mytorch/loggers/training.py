from typing import Optional, Union, List, Tuple
import io
import sys
from pydantic import BaseModel
import torch
import pathlib
import numpy as np
import logging
from pydantic import FilePath, PositiveInt, ConfigDict

ArrayLike = Union[List, torch.Tensor, np.ndarray, Tuple]


def get_logger(history_file, error_file) -> logging.Logger:
    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)

    history_handler = logging.FileHandler(history_file)
    console_handler = logging.StreamHandler()
    error_handler = logging.FileHandler(error_file)

    history_handler.setLevel(logging.INFO)
    error_handler.setLevel(logging.ERROR)
    console_handler.setLevel(logging.INFO)

    format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_format = logging.Formatter('%(message)s')

    history_handler.setFormatter(format)
    error_handler.setFormatter(format)
    console_handler.setFormatter(console_format)

    logger.addHandler(history_handler)
    logger.addHandler(console_handler)
    logger.addHandler(error_handler)
    return logger



class TrainLogger(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    history_file: FilePath
    error_file: FilePath

    @property
    def logger(self) -> logging.Logger:
        if not hasattr(self, '_logger'):
            self._logger = get_logger(self.history_file, self.error_file)
        return self._logger


    def info(self, message: str) -> None:
        self.logger.info(message)

    def error(self, message: str) -> None:
        self.logger.error(message)

    def warning(self, message: str) -> None:
        self.logger.warning(message)

    def debug(self, message: str) -> None:
        self.logger.debug(message)



class ProgressLogger(TrainLogger):
    print_every: int = 1

    def log(self, epoch: int, train_loss: float, val_loss: ArrayLike) -> None:
        self.log_epoch(epoch, train_loss, val_loss[-1])
    def log_epoch(self, epoch: int, train_loss: float, val_loss: float) -> None:
        if epoch % self.print_every == 0:
            self.info(f'Epoch {epoch} | Train Loss: {train_loss:.5e} | Val Loss: {val_loss:.5e}')



class CheckpointLogger(TrainLogger):

    model: torch.nn.Module
    relative_tol: float = 0.7
    min_val_loss: float = float('inf')
    checkpoint_path: FilePath
    past_n: PositiveInt = 10

    def log(self, epoch: int, train_loss: float, val_loss: ArrayLike) -> None:
        self.checkpoint(epoch, val_loss)

    def checkpoint(self, epoch: int, val_loss: ArrayLike) -> None:
        mean_val_loss = self.mean(val_loss)
        relative_diff = self.relative_tolerance(mean_val_loss)
        if relative_diff > self.relative_tol:
            self.info(f"Saving model. Epoch: {epoch} | Mean Loss: {mean_val_loss:.5e} | Relative Diff: {relative_diff:.5e}")
            self.min_val_loss = mean_val_loss
            current_best_model = torch.jit.script(self.model)
            current_best_model.save(self.checkpoint_path)
    def relative_diff(self, mean: float) -> float:
        return  (self.min_loss - mean) / mean

    def mean(self, loss: ArrayLike) -> float:
        return sum(loss[-self.past_n-1:]) / self.past_n


class ProgressCheckpointLogger(ProgressLogger, CheckpointLogger):
    def log(self, epoch: int, train_loss: float, val_loss: ArrayLike) -> None:
        self.log_epoch(epoch, train_loss, val_loss[-1])
        self.checkpoint(epoch, val_loss)


def train_logger_factory(logger_type: str, **kwargs) -> TrainLogger:
    if logger_type == 'progress':
        return ProgressLogger(**kwargs)
    elif logger_type == 'checkpoint':
        return CheckpointLogger(**kwargs)
    elif logger_type == 'progress_checkpoint':
        return ProgressCheckpointLogger(**kwargs)
    else:
        raise ValueError(f"Invalid logger type: {logger_type}")
