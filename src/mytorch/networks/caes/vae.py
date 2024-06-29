import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, encoder, decoder, latent_size, activation=nn.ReLU()):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_size = latent_size
        self.linear_mean = nn.Linear(latent_size, latent_size)
        self.linear_std = nn.Linear(latent_size, latent_size)
        self.linear_decode = nn.Linear(latent_size, latent_size)

    def encode(self, x):
        x = self.encoder.encode(x)
        return self.linear_mean(x), self.linear_std(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z1 = self.activation(self.linear_decode(z))
        return self.decoder.decode(z1)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
