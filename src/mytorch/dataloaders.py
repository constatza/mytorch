from typing import Optional

import torch
from pydantic import FilePath
from torch.utils.data import TensorDataset, DataLoader

from mytorch.io.utils import smart_load_tensors


def create_dataloaders(
    x_train: torch.Tensor,
    x_test: torch.Tensor,
    y_train: Optional[torch.Tensor] = None,
    y_test: Optional[torch.Tensor] = None,
    batch_size: Optional[int] = 32,
):
    """
    Creates PyTorch Dataloaders for the training and validation data.

    Args:
        x_train (torch.Tensor): The training data.
        x_test (torch.Tensor): The validation data.
        y_train (torch.Tensor, optional): The training labels. Defaults to None.
        y_test (torch.Tensor, optional): The validation labels. Defaults to None.
        batch_size (int, optional): The batch size. Defaults to 32.

    Returns:
        tuple[DataLoader, DataLoader]: The Dataloaders for the training and validation data.
    """
    train_dataset = TensorDataset(x_train, y_train if y_train is not None else x_train)
    val_dataset = TensorDataset(x_test, y_test if y_test is not None else x_test)

    dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dataloader_val = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    return dataloader_train, dataloader_val


def create_dataloaders_from_path_config(
    x_train_path: FilePath,
    x_test_path: FilePath,
    y_train_path: FilePath,
    y_test_path: FilePath,
    convolution_dims: int,
):
    """get the train_loader and test_loader from the data and return them."""

    x_train = smart_load_tensors(x_train_path, convolution_dims)
    x_test = smart_load_tensors(x_test_path, convolution_dims)

    if x_train_path != y_train_path and x_test != y_test_path:
        y_train = smart_load_tensors(y_train_path, convolution_dims)
        y_test = smart_load_tensors(y_test_path, convolution_dims)
    else:
        y_train = None
        y_test = None

    dataloader_train, dataloader_val = create_dataloaders(
        x_train, x_test, y_train, y_test
    )
    return dataloader_train, dataloader_val
