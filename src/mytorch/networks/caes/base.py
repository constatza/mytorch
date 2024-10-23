import torch
from lightning import LightningModule

from mytorch.metrics import normalized_rmse
from mytorch.networks.blocks import OptimizerSchedulerNetwork


class CAE(OptimizerSchedulerNetwork):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def encode(self, x):
        raise NotImplementedError

    def decode(self, x):
        raise NotImplementedError

    def forward(self, x, full=True):
        encoding = self.encode(x)
        if full:
            return self.decode(encoding)
        return encoding

    def training_step(self, batch, batch_idx):
        x = batch[0]
        x_hat = self.forward(x)
        loss = self.training_loss(x_hat, x)
        self.train_loss = loss
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[0]
        x_hat = self.forward(x)
        loss = self.training_loss(x_hat, x)
        self.val_loss = loss
        return loss

    def test_step(self, batch, batch_idx):
        x = batch[0]
        x_hat = self.forward(x)
        loss = self.test_loss(x_hat, x)
        self.test_loss = loss
        return loss

    def predict_step(self, batch, batch_idx):
        x = batch[0]
        encoding = self.encode(x)
        predictions = self.decode(encoding)
        return predictions, encoding

    @staticmethod
    def training_loss(x_hat, x):
        return torch.nn.functional.huber_loss(x_hat, x)

    @staticmethod
    def test_loss(x_hat, x):
        return normalized_rmse(x_hat, x)
