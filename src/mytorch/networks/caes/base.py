import torch
from lightning import LightningModule

from mytorch.metrics import normalized_rmse


class CAE(LightningModule):

    def __init__(self, encoder, decoder):
        super(CAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def training_step(self, batch, batch_idx):
        x = batch[0]
        x_hat = self.forward(x)
        loss = self.training_loss(x_hat, x)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[0]
        x_hat = self.forward(x)
        loss = self.training_loss(x_hat, x)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x = batch[0]
        x_hat = self.forward(x)
        loss = self.test_loss(x_hat, x)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx):
        x = batch[0]
        encoding = self.encode(x)
        predictions = self.decode(encoding)
        return predictions, encoding

    def configure_optimizers(self):
        return torch.optim.RAdam(self.parameters(), lr=self.hparams.lr)

    def on_validation_epoch_end(self) -> None:
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", lr, on_step=False, on_epoch=True, prog_bar=True)

    @staticmethod
    def training_loss(x_hat, x):
        return torch.nn.functional.huber_loss(x_hat, x)

    @staticmethod
    def test_loss(x_hat, x):
        return normalized_rmse(x_hat, x)
