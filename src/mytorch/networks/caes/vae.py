import torch
import torch.nn as nn

from mytorch.networks.caes.cae1d import LinearChannelDescentLatent1d


class VAE(nn.Module):
    def __init__(self, cae, latent_size, activation=nn.ReLU()):
        super(VAE, self).__init__()
        self.cae = cae
        self.latent_size = latent_size
        self.linear_mean = nn.Linear(latent_size, latent_size)
        self.linear_std = nn.Linear(latent_size, latent_size)
        self.linear_decode = nn.Linear(latent_size, latent_size)
        self.activation = activation

    def encode(self, x):
        x = self.cae.encode(x)
        return self.linear_mean(x), self.linear_std(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z1 = self.activation(self.linear_decode(z))
        return self.cae.decode(z1)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class LinearChannelDescentVAE(VAE):

    def __init__(
        self,
        input_shape,
        latent_size: int = 20,
        num_layers: int = 4,
        kernel_size: int = 7,
    ):
        base_cae = LinearChannelDescentLatent1d(
            input_shape, latent_size, num_layers, kernel_size
        )
        super(LinearChannelDescentVAE, self).__init__(
            cae=base_cae,
            latent_size=latent_size,
        )
