import torch

from utils import to_tensor


@to_tensor
def root_mean_square(tensor):
    return torch.sqrt(torch.mean(tensor ** 2))


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
