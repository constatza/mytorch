import pytest
import torch
from torch.optim import SGD
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from mytorch.trainers import Trainer, AutoEncoderTrainer
from mytorch.config.core import TrainingConfig
from mytorch.config.loggers import ProgressLogger

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
        model=mock_model,
        optimizer=SGD(mock_model.parameters(), lr=0.01),
        criterion=MSELoss(),
        train_loader=mock_dataloader,
        test_loader=mock_dataloader,
        num_epochs=5,
        batch_size=2,
        device=torch.device('cpu'),
        logger= ProgressLogger(console=False),
        unique_id='1234',

    )

@pytest.fixture
def training_config_autoencoder(mock_model, mock_dataloader_input_only):
    return TrainingConfig(
        model=mock_model,
        optimizer=SGD(mock_model.parameters(), lr=0.01),
        criterion=MSELoss(),
        train_loader=mock_dataloader_input_only,
        test_loader=mock_dataloader_input_only,
        num_epochs=5,
        batch_size=2,
        device=torch.device('cpu'),
        logger= ProgressLogger(console=False),
        unique_id='1234'
    )

# Use the fixtures in the tests
def test_Trainer(training_config):
    trainer = Trainer(config=training_config)
    assert trainer.model == training_config.model
    assert trainer.optimizer == training_config.optimizer
    assert trainer.criterion == training_config.criterion
    assert trainer.train_loader == training_config.train_loader
    assert trainer.test_loader == training_config.test_loader

def test_AutoEncoderTrainer(training_config_autoencoder):
    trainer = AutoEncoderTrainer(config=training_config_autoencoder)
    assert trainer.model == training_config_autoencoder.model
    assert trainer.optimizer == training_config_autoencoder.optimizer
    assert trainer.criterion == training_config_autoencoder.criterion
    assert trainer.train_loader == training_config_autoencoder.train_loader
    assert trainer.test_loader == training_config_autoencoder.test_loader

# Test the training capabilities of the Trainer class
def test_Trainer_train(training_config):
    trainer = Trainer(config=training_config)
    train_losses, val_losses = trainer.train(5)
    assert len(train_losses) == 5
    assert len(val_losses) == 5


# Test the training capabilities of the AutoEncoderTrainer class
def test_AutoEncoderTrainer_train(training_config_autoencoder):
    trainer = AutoEncoderTrainer(config=training_config_autoencoder)
    train_losses, val_losses = trainer.train(5)
    assert len(train_losses) == 5
    assert len(val_losses) == 5