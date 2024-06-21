import torch
from torch.utils.data import TensorDataset, DataLoader
from typing import Optional, Union


class OptionalTargetDataset(TensorDataset):
    """
    A PyTorch Dataset that can handle both autoencoder and regular labeled data.

    If y_data is provided, it is used as the target data. If y_data is not provided,
    the input data (x_data) is used as the target data, which is a characteristic of autoencoders.

    Attributes:
        x_data (torch.Tensor): The input data.
        y_data (torch.Tensor): The target data. If None, x_data is used as the target data.
        is_autoencoder (bool): Indicates whether the dataset is for an autoencoder.
    """

    def __init__(self, x_data: torch.Tensor, y_data: Optional[Union[torch.Tensor, None]] = None) -> None:
        """
        Initializes the dataset with the given input and target data.

        Args:
            x_data (torch.Tensor): The input data.
            y_data (Optional[torch.Tensor, None]): The target data. If None, x_data is used as the target data.
        """
        self.x_data = torch.tensor(x_data, dtype=torch.float32)
        if y_data is not None:
            self.y_data = torch.tensor(y_data, dtype=torch.float32)
            self.is_autoencoder = False
        else:
            self.y_data = self.x_data
            self.is_autoencoder = True

        self.x_shape = self.x_data.shape
        self.y_shape = self.y_data.shape

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.x_data)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the input and target data for the sample at the given index.

        Args:
            idx (int): The index of the sample.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The input and target data for the sample.
        """
        return self.x_data[idx], self.y_data[idx]


def create_dataloaders(x_train: torch.Tensor, x_test: torch.Tensor, y_train=None, y_test=None, batch_size=32):
    """
    Creates PyTorch Dataloaders for the training and validation data.

    Args:
        x_train (Optional[torch.Tensor, None]): The training input data.
        y_train (Optional[torch.Tensor, None]): The training target data. If None, x_train is used as the target data.
        x_test (Optional[torch.Tensor, None]): The validation input data.
        y_test (Optional[torch.Tensor, None]): The validation target data. If None, x_test is used as the target data.
        batch_size (int, optional): The batch size. Defaults to 32.

    Returns:
        tuple[DataLoader, DataLoader]: The Dataloaders for the training and validation data.
    """
    train_dataset = OptionalTargetDataset(x_train, y_data=y_train)
    val_dataset = OptionalTargetDataset(x_test, y_data=y_test)

    dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dataloader_val = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return dataloader_train, dataloader_val
