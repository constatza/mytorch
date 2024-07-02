import pytest
import torch
from torch.nn import MSELoss
from torch.optim import SGD
from torch.utils.data import DataLoader

from mytorch.io.config import TrainingConfig, EstimatorsConfig
from mytorch.io.loggers import ProgressLogger


# Define pytest fixtures for the common setup code
@pytest.fixture
def mock_model():
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 1)

        def forward(self, x):
            return self.linear(x)

    return MockModel()


@pytest.fixture
def mock_dataloader():
    x = torch.randn(10, 1)
    y = torch.randn(10, 1)
    return DataLoader(list(zip(x, y)), batch_size=2)


@pytest.fixture
def mock_dataloader_input_only():
    x = torch.randn(10, 1)
    return DataLoader(x, batch_size=2)


@pytest.fixture
def training_config(mock_model, mock_dataloader):
    return TrainingConfig(
        optimizer=SGD,
        criterion=MSELoss,
        lr=0.01,
        train_loader=mock_dataloader,
        test_loader=mock_dataloader,
        num_epochs=5,
        batch_size=2,
        device=torch.device("cpu"),
        logger=ProgressLogger(console=False),
        unique_id="1234",
    )


@pytest.fixture
def training_config_autoencoder(mock_model, mock_dataloader_input_only):
    return TrainingConfig(
        optimizer=SGD,
        criterion=MSELoss,
        train_loader=mock_dataloader_input_only,
        test_loader=mock_dataloader_input_only,
        num_epochs=5,
        batch_size=2,
        lr=0.01,
        device=torch.device("cpu"),
        logger=ProgressLogger(console=False),
        unique_id="1234",
    )


@pytest.fixture
def estimator_config(mock_model):
    return EstimatorsConfig(
        model=mock_model,
        name="MockModel",
        kernel_size=3,
    )


# Use the fixtures in the tests
