import torch
from torch.nn import functional as F

from mytorch.networks.utils import to_tensor


@to_tensor
def root_mean_square(tensor):
    return torch.sqrt(torch.mean(tensor**2))


@to_tensor
def rmse(predictions, targets):
    """Root Mean Square Error"""
    return root_mean_square(predictions - targets)


def ktl(predictions, targets):
    """Kalogeris Timeseries Loss"""
    norm = torch.norm(targets, dim=-1)
    diff = torch.norm(predictions - targets, dim=-1)
    return torch.mean(diff / norm)


def normalized_rmse(predictions, targets):
    """Normalized Root Mean Square Error"""
    return rmse(predictions, targets) / root_mean_square(targets)


# Reconstruction + KL divergence losses summed over all elements and batch


def mse_plus_kl_divergence(recon_x, x, mu, logvar):
    BCE = F.mse_loss(recon_x, x, reduction="sum")
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD
