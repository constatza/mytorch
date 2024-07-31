import shutil

import numpy as np
import torch
from lightning.pytorch.callbacks import TQDMProgressBar, ModelCheckpoint, ModelSummary
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.trainer import Trainer

from mytorch.datamodules import FileDataModule
from mytorch.io.readers import read_study
from mytorch.networks.caes import LinearChannelDescentLatent1d
from mytorch.pipeline import Pipeline
from mytorch.transforms import StandardScaler, NumpyToTensor
from mytorch.utils import tune_lr


def main():
    num_epochs = 350
    num_layers = 2
    kernel_size = 5
    latent_size = 4
    reduced_channels = 10
    reduced_timesteps = 100
    lr = 1e-3
    tune = False
    continue_training = False
    rebuild_dataset = True

    torch.set_float32_matmul_precision("high")

    config_file = "./config.toml"
    config = read_study(config_file)
    paths = config.paths
    root_dir = paths.workdir

    logger = CSVLogger(paths.logs)

    callbacks = [
        ModelSummary(max_depth=2),
        TQDMProgressBar(),
        ModelCheckpoint(
            monitor="val_loss",
            dirpath=paths.checkpoints,
            filename="best-checkpoint",
            save_top_k=2,
            mode="min",
            save_last=True,
            every_n_epochs=10,
        ),
    ]

    x_transforms = Pipeline(
        NumpyToTensor(),
        StandardScaler(axis=(0, -1)),
    )

    data_module = FileDataModule(
        save_dir=paths.input,
        paths=paths.features,
        transforms=x_transforms,
        rebuild_dataset=rebuild_dataset,
        batch_size=64,
        test_size=0.2,
    )

    data_module.setup(stage="fit")

    trainer = Trainer(
        default_root_dir=root_dir,
        max_epochs=num_epochs,
        accelerator="gpu",
        callbacks=callbacks,
        # gradient_clip_val=0.5,
        enable_progress_bar=True,
        reload_dataloaders_every_n_epochs=10,
        logger=logger,
        # fast_dev_run=True,
    )
    model = LinearChannelDescentLatent1d(
        input_shape=data_module.input_shape,
        latent_size=latent_size,
        reduced_channels=reduced_channels,
        reduced_timesteps=reduced_timesteps,
        num_layers=num_layers,
        kernel_size=kernel_size,
        lr=lr,
    )

    model = tune_lr(trainer, model, data_module) if tune else model

    if continue_training:
        trainer.fit(model, datamodule=data_module, ckpt_path="last")
    else:
        # delete the checkpoint directory
        shutil.rmtree(paths.checkpoints, ignore_errors=True)
        shutil.rmtree(paths.logs, ignore_errors=True)
        trainer.fit(model, datamodule=data_module)

    data_module.setup(stage="test")
    test_score = trainer.test(model, datamodule=data_module)

    data_module.setup(stage="predict")
    output = trainer.predict(model, datamodule=data_module)
    predictions, latent = zip(*output)

    predictions = torch.cat(predictions, dim=0)
    predictions = data_module.feature_transforms.inverse_transform(predictions)
    latent = torch.cat(latent, dim=0).cpu().numpy()

    np.save(paths.predictions.with_suffix(".npy"), predictions)
    np.save(paths.latent.with_suffix(".npy"), latent)


def separate_dofs(features_raw, path_raw):
    dofs = path_raw / "dofs" / "Pdofs.txt"
    dofs = np.loadtxt(dofs).astype(int)
    return features_raw[:, :, dofs].transpose(0, 2, 1)


if __name__ == "__main__":
    main()
