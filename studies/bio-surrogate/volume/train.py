import torch
import torch.nn as nn
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.trainer import Trainer

from mytorch.datamodules import FileDataModule
from mytorch.io.readers import read_study
from mytorch.networks.ffnns import FeedForwardNN

torch.set_float32_matmul_precision("medium")

config_file = "./config.toml"
config = read_study(config_file)
paths = config.paths

data_module = FileDataModule(
    default_dir=paths.workdir,
    data_path=paths.data,
    targets_path=paths.targets,
)


layers = [3, 5, 1]

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

# tuner = Tuner(
#     trainer,
# )
#
# lr_finder = tuner.lr_find(
#     model,
#     train_loader,
#     val_loader,
#     min_lr=1e-8,
#     max_lr=1e-1,
#     num_training=100,
#     early_stop_threshold=50,
#     # mode="linear",
# )
#
# fig = lr_finder.plot(suggest=True, show=True)
#
#
# # Pick point based on plot, or get suggestion
# new_lr = lr_finder.suggestion()
#
# # update hparams of the model
# model.hparams.lr = new_lr
#
#
trainer.fit(model, datamodule=data_module)
#
# trainer.test(model, val_loader)

# save the model
model_path = config.paths.output / "ffnn_model.ckpt"
trainer.save_checkpoint(model_path)
