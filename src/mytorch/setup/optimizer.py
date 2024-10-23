import torch.optim as optim


def initialize_optimizer(config, model):

    optimizer_name = config.pop("name", "Adam")
    optimizer_class = getattr(optim, optimizer_name, optim.Adam)
    optimizer = optimizer_class(model.parameters(), **config)
    return optimizer
