from pathlib import Path
from typing import Optional, Callable, Iterable

import numpy as np
from lightning import LightningDataModule
from pydantic import FilePath, validate_call
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

from mytorch.io.logging import get_logger
from mytorch.io.readers import read_array_as_numpy
from mytorch.io.writers import savez_asarray
from mytorch.mytypes import TupleLike, CreateIfNotExistsDir
from mytorch.transforms import NumpyToTensor, Transformation

logger = get_logger(__name__)


# TODO: maybe use symbolic links to avoid copying the data
class FileDataModule(LightningDataModule):
    """Class to load data from files and prepare it for training, validation, and testing.

    Args:
        save_dir (CreateIfNotExistsDir): Directory to save the dataset and metadata.
        paths (TupleLike[FilePath]): Paths to the data files.
        preprocessors (TupleLike[Callable, ...], optional): Preprocessors for the data. Defaults to (lambda x: x, lambda x: x).
        transforms (TupleLike[Transformation, ...], optional): Transforms for the data. Defaults to (NumpyToTensor(), NumpyToTensor()).
        test_size (float, optional): Percentage of the whole dataset to use for testing. Defaults to 0.2.
        val_size (float, optional): Percentage of the test dataset to use for validation. Defaults to 0.5.
        batch_size (int, optional): Batch size for the dataloaders. Defaults to 64.
        rebuild_dataset (bool, optional): Whether to rebuild the dataset. Defaults to False.
        names (TupleLike[str], optional): Names of the data. Defaults to ("features", "targets").
        indices_path (Optional[FilePath], optional): Path to the metadata file. Defaults to None.

    """

    @validate_call(config={"arbitrary_types_allowed": True})
    def __init__(
        self,
        save_dir: CreateIfNotExistsDir,
        paths: TupleLike[FilePath],
        preprocessors: TupleLike[Callable, ...] = (lambda x: x, lambda x: x),
        transforms: TupleLike[Transformation, ...] = (NumpyToTensor(), NumpyToTensor()),
        test_size: float = 0.2,  # percentage of the whole dataset
        val_size: float = 0.5,  # percentage of the test dataset
        batch_size: int = 64,
        rebuild_dataset: bool = False,
        names: TupleLike[str] = ("features", "targets"),
        indices_path: Optional[FilePath] = None,
    ):
        super().__init__()
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        # find the least common number of files, names, transforms, and preprocessors
        zipped = zip(paths, names, transforms, preprocessors)
        self.paths, self.names, self.transforms, self.preprocessors = zip(*zipped)
        self.num_files = len(self.paths)
        if self.num_files != len(transforms) or self.num_files != len(
            self.preprocessors
        ):
            logger.warn(
                "The number of paths, names, transforms, and preprocessors are not equal."
            )

        self.targets_exist = False
        if self.num_files == 2:
            if self.paths[1] != self.paths[0]:
                # if the target path is different from the feature path
                self.targets_exist = True

        self.batch_size = batch_size
        self.test_size = test_size
        self.val_size = val_size
        self.indices_path = indices_path
        self._metadata = None
        self._dataset = None

        self.input_shape = None
        self.output_shape = None

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.x_test = None
        self.y_test = None

        self.rebuild_dataset = rebuild_dataset
        self.indices_path = indices_path

    def path_to(self, filename: str, suffix: str = ".npy") -> Path:
        return (self.save_dir / filename).with_suffix(suffix)

    def prepare_data(self):
        if not self.path_to("dataset").exists() or self.rebuild_dataset:
            logger.info("Preparing data.")
            data = self.get_data(self.paths, self.preprocessors)
            logger.info("Saving dataset.")
            self.save_data(data)
            logger.info("Saving metadata.")
            self.save_metadata(data)

    def save_data(self, data):
        savez_asarray(
            self.path_to("dataset"),
            **{name: data for name, data in zip(self.names, data)},
        )

    def get_train_val_test_indices(self, num_samples: int = None):
        # Save indices for train, val, test
        train_idx, val_idx, test_idx = train_val_test_split(
            indices=np.arange(num_samples),
            test_size=self.test_size,
            val_size=self.val_size,
        )
        return {
            "train_idx": train_idx,
            "val_idx": val_idx,
            "test_idx": test_idx,
            "num_samples": num_samples,
        }

    def save_metadata(self, data):
        """Save metadata to a file.
        Train/Val/Test indices are generated if not provided in indices_path.
        """
        num_samples = data[0].shape[0]
        if self.indices_path is None:
            logger.warn("Indices being split into train/validation/test.")
            indices = self.get_train_val_test_indices(num_samples)
        else:
            logger.info(
                f"Loading train/validation/test indices from {self.indices_path}."
            )
            metadata = np.load(self.indices_path)
            indices = {
                "train_idx": metadata["train_idx"],
                "val_idx": metadata["val_idx"],
                "test_idx": metadata["test_idx"],
            }

        metadata = {
            **indices,
            **{name + "_shape": data.shape for name, data in zip(self.names, data)},
        }

        savez_asarray(
            self.path_to("metadata"),
            **metadata,
        )

    @staticmethod
    def get_data(paths: tuple, preprocessors: Iterable) -> tuple[np.ndarray]:
        data_raw = (read_array_as_numpy(path) for path in paths)
        return tuple(
            preprocessor(data) for data, preprocessor in zip(data_raw, preprocessors)
        )

    @property
    def dataset(self):
        if self._dataset is None:
            self._dataset = np.load(self.path_to("dataset", ".npz"))
        return self._dataset

    @property
    def metadata(self):
        if self.indices_path is not None:
            return np.load(self.indices_path)
        return self._metadata or np.load(self.path_to("metadata", ".npz"))

    def setup(self, stage: str | None = None):
        # To be implemented by child classes
        self.input_shape = self.metadata["features_shape"]
        self.feature_transforms = self.transforms[0]
        if self.targets_exist:
            self.target_transforms = self.transforms[1]
            self.output_shape = self.metadata["targets_shape"]
        else:
            self.output_shape = self.input_shape
            self.target_transforms = None

        if stage == "fit":
            train_idx = self.metadata["train_idx"]
            val_idx = self.metadata["val_idx"]

            self.x_train = self.feature_transforms.fit_transform(
                self.dataset["features"][train_idx]
            )
            self.x_val = self.feature_transforms.transform(
                self.dataset["features"][val_idx]
            )

            if self.targets_exist:
                self.y_train = self.target_transforms.fit_transform(
                    self.dataset["targets"][train_idx]
                )
                self.y_val = self.target_transforms.transform(
                    self.dataset["targets"][val_idx]
                )
            else:
                self.y_train = self.x_train
                self.y_val = self.x_val

            self.train_dataset = TensorDataset(self.x_train, self.y_train)
            self.val_dataset = TensorDataset(self.x_val, self.y_val)

        elif stage == "test":
            test_idx = self.metadata["test_idx"]
            x_test = self.dataset["features"][test_idx]
            self.x_test = self.feature_transforms.transform(x_test)
            if self.targets_exist:
                y_test = self.dataset["targets"][test_idx]
                self.y_test = self.target_transforms.transform(y_test)
            else:
                self.y_test = self.x_test

            self.test_dataset = TensorDataset(self.x_test, self.y_test)

        elif stage == "predict":
            self.features = self.feature_transforms.transform(self.dataset["features"])
            if self.targets_exist:
                self.targets = self.target_transforms.transform(self.dataset["targets"])
            else:
                self.targets = self.features
            self.predict_dataset = TensorDataset(self.features, self.targets)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.predict_dataset, batch_size=len(self.predict_dataset), shuffle=False
        )


def train_val_test_split(indices=None, test_size=0.2, val_size=0.5):
    train, test_plus_val = train_test_split(
        indices,
        test_size=test_size,
        shuffle=True,
    )

    val, test = train_test_split(
        test_plus_val,
        test_size=val_size,
        shuffle=True,
    )

    return train, val, test
