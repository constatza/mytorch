import torch.nn as nn
import torch.optim
from lightning import LightningModule

import mytorch.metrics
from mytorch.mytypes import TupleLike


class FeedForwardNN(LightningModule):

    def __init__(
        self,
        layers: TupleLike = None,
        activation: nn.Module = nn.GELU(),
        lr: float = 1e-3,
    ):
        super(FeedForwardNN, self).__init__()
        self.save_hyperparameters()
        self.num_layers = len(layers) - 1

        self.layers = nn.ModuleList()

        for i in range(self.num_layers - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
            self.layers.append(nn.LayerNorm(layers[i + 1]))
            self.layers.append(activation)

        self.layers.append(nn.Linear(layers[-2], layers[-1]))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)

    def configure_optimizers(self):
        optimizer = torch.optim.RAdam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.6,
            patience=60,
            min_lr=1e-3,
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
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.test_loss(y_hat, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        y_hat = self.forward(x)
        return y_hat

    @staticmethod
    def loss(y_hat, y):
        return nn.functional.huber_loss(y_hat, y)

    @staticmethod
    def test_loss(y_hat, y):
        return mytorch.metrics.normalized_rmse(y_hat, y)

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
