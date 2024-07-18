from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
from lightning import LightningDataModule
from pydantic import FilePath
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

from mytorch.io.readers import read_array_as_numpy
from mytorch.pipeline import Pipeline


def create_datasets(
    stage: str,
    x_test: torch.Tensor | np.ndarray,
    x_train: Optional[np.ndarray] = None,
    y_train: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
) -> Tuple[Optional[TensorDataset], TensorDataset]:
    """
    Creates PyTorch Dataloaders for the training and validation data.

    Args:
        stage (str): The stage of the pipeline.
        x_train (torch.Tensor): The training data.
        x_test (torch.Tensor): The validation data.
        y_train (torch.Tensor, optional): The training labels. Defaults to None.
        y_test (torch.Tensor, optional): The validation labels. Defaults to None.

    Returns:
        tuple[DataLoader, DataLoader]: The Dataloaders for the training and validation data.

    """
    x_train, x_test, y_train, y_test = map(
        lambda x: torch.from_numpy(x).float() if x is not None else None,
        (x_train, x_test, y_train, y_test),
    )
    if stage == "fit":
        return TensorDataset(x_train, y_train), TensorDataset(x_test, y_test)

    if stage == "test":
        return None, TensorDataset(x_test, y_test)

    if stage == "predict":
        return None, TensorDataset(x_test)


class SupervisedDataModule(LightningDataModule):
    def __init__(
        self,
        data_path: Path,
        targets_path: Optional[Path] = None,
        batch_size: int = 32,
        test_size: float = 0.2,
        val_size: float = 0.2,
        shuffle: bool = True,
        input_shape: Optional[Tuple[int, ...]] = None,
        output_shape: Optional[Tuple[int, ...]] = None,
    ):
        super().__init__()
        self.data_path = data_path
        self.targets_path = targets_path or data_path
        self.batch_size = batch_size
        self.test_size = test_size
        self.val_size = val_size
        self.shuffle = shuffle
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        # To be implemented by child classes
        pass

    def setup(self, stage: Optional[str] = None):
        # To be implemented by child classes
        pass

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)


class FileDataset(LightningDataModule):

    def __init__(
        self,
        default_dir: Path,
        data_path: FilePath,
        targets_path: Optional[FilePath] = None,
        test_size: float = 0.2,
        val_size: float = 0.2,
        batch_size: int = 32,
        shuffle_test: bool = False,
        rebuild_dataset: bool = False,
        pipeline: Optional[Pipeline] = None,
        input_shape: Optional[Tuple[int, ...]] = None,
        output_shape: Optional[Tuple[int, ...]] = None,
    ):
        super().__init__()
        self.data_path = Path(data_path)
        self.targets_path = (
            Path(targets_path) if targets_path is not None else self.data_path
        )
        self.test_size = test_size
        self.train_size = 1 - test_size
        self.val_size = val_size
        self.batch_size = batch_size
        self.shuffle_test = shuffle_test
        # dir save the train/test split
        self.default_dir = Path(default_dir)
        self.default_dir.mkdir(parents=True, exist_ok=True)
        # load path for train/test data
        self.load_path = self.default_dir / "dataset.npz"
        self.rebuild_dataset = rebuild_dataset

        self.train = None
        self.val = None
        self.test = None
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.pipeline = pipeline

    def prepare_data(self) -> None:
        raise NotImplementedError
        if not self.load_path.exists() or self.rebuild_dataset:
            data_array = read_array_as_numpy(self.data_path)
            target_array = read_array_as_numpy(self.targets_path)

            dofs = self.data_path.parent.parent / "dofs" / "Pdofs.txt"
            dofs = np.loadtxt(dofs).astype(int)
            x = data_array[:, :, dofs].transpose(0, 2, 1)
            y = x

            x_train, x_test, y_train, y_test = train_test_split(
                x,
                y,
                test_size=self.test_size,
                shuffle=True,
            )

            x_train = self.pipeline.fit_transform(x_train)
            x_test = self.pipeline.transform(x_test)

            y_train = self.pipeline.fit_transform(y_train)
            y_test = self.pipeline.transform(y_test)

            np.save(self.data_path.parent / "p_solutions.npy", x)

            save_uncompressed(
                self.load_path,
                x_train=x_train,
                x_test=x_test,
                y_train=y_train,
                y_test=y_test,
            )
            print("Train/Test saved to", self.load_path)

    def setup(self, stage: str):
        data = np.load(self.load_path)
        if stage == "fit":
            x_train = data["x_train"]
            y_train = data["y_train"]
            # train/val split
            x_train, x_val, y_train, y_val = train_test_split(
                x_train, y_train, test_size=self.val_size, shuffle=True
            )

            self.train_dataset, self.val_dataset = create_datasets(
                stage=stage,
                x_train=x_train,
                x_test=x_val,
                y_train=y_train,
                y_test=y_val,
            )

        if stage == "test":
            x_test = data["x_test"]
            y_test = data["y_test"]
            _, self.test_dataset = create_datasets(
                stage=stage,
                x_test=x_test,
                y_test=y_test,
            )

        if stage == "predict":
            x_test = data["x_test"]
            y_test = data["y_test"]
            _, self.predict_dataset = create_datasets(
                stage=stage,
                x_test=x_test,
                y_test=y_test,
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset, batch_size=self.batch_size, shuffle=False
        )

    def encode_dataloader(self):
        return DataLoader(
            self.predict_dataset, batch_size=self.batch_size, shuffle=False
        )


def percentage_to_size(percentage: float, total_size: int) -> int:
    """Converts a percentage to a size."""
    return int(total_size * percentage)


def get_split_sizes(
    total_size: int, percentages: Tuple[float] | List[float]
) -> Tuple[int]:
    """Get the sizes for the training, validation, and test sets."""

    if not sum(percentages) == 1:
        raise ValueError("The percentages must add up to one.")
    if not all(0 <= percentage <= 1 for percentage in percentages):
        raise ValueError("The value must be a percentage.")

    sizes_gen = [
        percentage_to_size(percentage, total_size) for percentage in percentages
    ]
    sizes_gen[-1] = total_size - sum(sizes_gen[:-1])

    return tuple(sizes_gen)


def save_uncompressed(path, **kwargs):
    kwargs = {k: v.numpy() if isinstance(v, torch.Tensor) else v for k, v in kwargs}
    np.savez_compressed(path, **kwargs)
