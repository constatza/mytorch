import numpy as np
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.tuner import Tuner
from torch.utils.data import DataLoader, TensorDataset

from mytorch.io.readers import read_study
from mytorch.networks.ffnns import FeedForwardNN

config_file = "../config/u-ffnn.toml"
config = read_study(config_file)

x_train = np.load(config.paths.x_train)
x_test = np.load(config.paths.x_test)
y_train = np.load(config.paths.y_train)
y_test = np.load(config.paths.y_test)

x_train = torch.from_numpy(x_train).float()
x_test = torch.from_numpy(x_test).float()
y_train = torch.from_numpy(y_train).float()
y_test = torch.from_numpy(y_test).float()


input_shape = x_train.shape[-1]
output_shape = y_train.shape[-1]
layers = [input_shape, 64, 128, 256, 128, 64, output_shape]
batch_size = 2**11
print(f"Latent space shape: {y_train.shape}")

train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


trainer = Trainer(
    default_root_dir=config.paths.output,
    max_epochs=10000,
    accelerator="gpu",
    callbacks=[TQDMProgressBar()],
    gradient_clip_val=0.5,
    enable_progress_bar=True,
    reload_dataloaders_every_n_epochs=10,
)

model = FeedForwardNN(layers=layers, activation=nn.functional.tanh)

tuner = Tuner(
    trainer,
)

lr_finder = tuner.lr_find(
    model,
    train_loader,
    val_loader,
    min_lr=1e-8,
    max_lr=1e-1,
    num_training=100,
    early_stop_threshold=50,
    # mode="linear",
)

fig = lr_finder.plot(suggest=True, show=True)


# Pick point based on plot, or get suggestion
new_lr = lr_finder.suggestion()

# update hparams of the model
model.hparams.lr = new_lr


trainer.fit(model, train_loader, val_loader)

trainer.test(model, val_loader)

# save the model
model_path = config.paths.output / "ffnn_model.ckpt"
trainer.save_checkpoint(model_path)
