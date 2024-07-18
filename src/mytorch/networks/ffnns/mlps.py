import torch.nn as nn
from lightning import LightningModule
from torch.optim import RAdam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from mytorch.mytypes import ListLike


class FeedForwardNN(LightningModule):

    def __init__(
        self,
        layers: ListLike = None,
        activation: nn.functional = nn.functional.relu,
        lr: float = 1e-3,
    ):
        super(FeedForwardNN, self).__init__()
        self.save_hyperparameters()
        self.linear_layers = nn.ModuleList(
            nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)
        )
        self.activation = activation
        self.norms = nn.ModuleList(
            nn.BatchNorm1d(layers[i]) for i in range(len(layers) - 1)
        )

    def forward(self, x):
        for i, layer in enumerate(self.linear_layers):
            x = self.norms[i](x)
            x = layer(x)
            x = self.activation(x)
            x = nn.functional.dropout(x, p=0.5)
        return x

    def configure_optimizers(self):
        optimizer = RAdam(self.parameters(), lr=self.hparams.lr)
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    @staticmethod
    def loss(y_hat, y):
        return nn.functional.mse_loss(y_hat, y)

    def on_validation_epoch_end(self) -> None:
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", lr, on_step=False, on_epoch=True, prog_bar=True)


class ConstantHiddenSizeFFNN(FeedForwardNN):
    def __init__(
        self,
        input_size: int = None,
        output_size: int = None,
        hidden_size: int = None,
        num_layers: int = None,
        **kwargs,
    ):
        layers = [input_size] + [hidden_size] * num_layers + [output_size]
        super(ConstantHiddenSizeFFNN, self).__init__(layers, **kwargs)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)
        return x
