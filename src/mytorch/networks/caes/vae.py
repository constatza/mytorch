from typing import Any

import torch
import torch.nn as nn
from lightning import LightningModule

from mytorch.networks.caes.cae1d import LinearChannelDescentLatent1d


class VAE(nn.Module):
    def __init__(self, cae, latent_size, activation=nn.GELU()):
        super(VAE, self).__init__()
        self.cae = cae
        self.latent_size = latent_size
        self.linear_mean = nn.Linear(latent_size, latent_size)
        self.linear_std = nn.Linear(latent_size, latent_size)
        self.linear_decode = nn.Linear(latent_size, latent_size)
        self.activation = activation

    def encode(self, x):
        x = self.cae.encode(x)
        x = self.activation(x)
        return self.linear_mean(x), self.linear_std(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = self.activation(z)
        z1 = self.activation(self.linear_decode(z))
        return self.cae.decode(z1)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class VAE2(LightningModule):
    def __init__(
        self,
        cae,
        latent_size: int,
        lr: float,
        beta: float,
        activation=nn.GELU(),
    ):
        super(VAE2, self).__init__()
        self.lr = lr
        self.cae = cae
        self.beta = beta
        hidden_size = cae.hidden_size
        self.linear_mean = nn.Linear(hidden_size, latent_size)
        self.linear_std = nn.Linear(hidden_size, latent_size)
        self.linear_decode = nn.Linear(latent_size, hidden_size)
        self.activation = activation
        self.example_input_array = torch.randn(*self.cae.input_shape)

    def encode(self, x):
        x = self.cae.encode(x)
        x = self.activation(x)
        return self.linear_mean(x), self.linear_std(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = self.activation(z)
        z1 = self.activation(self.linear_decode(z))
        return self.cae.decode(z1)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def training_step(self, batch, batch_idx):
        x = batch
        x_recon, mu, logvar = self(x)
        loss = self.loss(x, x_recon, mu, logvar)
        # Calculate loss (e.g., reconstruction loss + KL divergence)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        x_recon, mu, logvar = self(x)
        loss = self.loss(x, x_recon, mu, logvar)
        # Calculate loss (e.g., reconstruction loss + KL divergence)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x = batch
        x_recon, mu, logvar = self(x)
        loss = self.loss(x, x_recon, mu, logvar)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx) -> Any:
        x = batch
        latent_mean, _ = self.encode(x)
        return latent_mean

    def loss(self, x, x_recon, mu, logvar, kld_weight=0.005):
        recon_loss = torch.nn.functional.mse_loss(x_recon, x)
        kl_div = torch.mean(
            -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1), dim=0
        )
        loss = recon_loss + self.beta * kld_weight * kl_div
        return loss
        # return recon_loss

    def configure_optimizers(self):
        optimizer = torch.optim.RAdam(self.parameters(), lr=self.lr)
        return optimizer


class LinearChannelDescentVAE(VAE2):

    def __init__(
        self, input_shape, num_layers: int, kernel_size: int, beta: float, **kwargs
    ):
        base_cae = LinearChannelDescentLatent1d(
            input_shape, 2 * kwargs["latent_size"], num_layers, kernel_size
        )
        super(LinearChannelDescentVAE, self).__init__(
            cae=base_cae,
            beta=beta,
            **kwargs,
        )


if __name__ == "__main__":

    pass
