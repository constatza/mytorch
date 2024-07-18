import shutil

import torch
from lightning.pytorch.callbacks import TQDMProgressBar, ModelCheckpoint, ModelSummary
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.tuner import Tuner

from mytorch.datamodules import FileDataset
from mytorch.io.readers import read_study
from mytorch.networks.caes import LinearChannelDescentLatent1d


def main():
    num_epochs = 500
    num_layers = 2
    kernel_size = 5
    latent_size = 3
    reduced_channels = 10
    reduced_timesteps = 100
    lr = 1e-3
    tune = False
    continue_training = False
    rebuild_dataset = False

    torch.set_float32_matmul_precision("high")

    config_file = "./config.toml"
    config = read_study(config_file)
    paths = config.paths
    root_dir = paths.output

    callbacks = [
        ModelSummary(max_depth=2),
        TQDMProgressBar(),
        ModelCheckpoint(
            monitor="val_loss",
            dirpath=paths.checkpoints_dir,
            filename="best-checkpoint",
            save_top_k=2,
            mode="min",
            save_last=True,
            every_n_epochs=10,
        ),
    ]

    data_module = FileDataset(
        default_dir=paths.workdir,
        data_path=paths.data,
        targets_path=paths.targets,
        rebuild_dataset=rebuild_dataset,
        input_shape=(10, 663, 738),
        output_shape=(10, 663, 738),
        batch_size=64,
        test_size=0.2,
    )

    data_module.prepare_data()
    data_module.setup(stage="fit")

    trainer = Trainer(
        default_root_dir=root_dir,
        max_epochs=num_epochs,
        accelerator="gpu",
        callbacks=callbacks,
        gradient_clip_val=0.5,
        enable_progress_bar=True,
        reload_dataloaders_every_n_epochs=10,
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
        shutil.rmtree(paths.checkpoints_dir, ignore_errors=True)
        shutil.rmtree(paths.logs_dir, ignore_errors=True)
        trainer.fit(model, datamodule=data_module)

    data_module.setup(stage="test")
    test_score = trainer.test(model, datamodule=data_module)

    print(test_score)


def tune_lr(trainer, model, data_module):
    tuner = Tuner(
        trainer,
    )

    lr_finder = tuner.lr_find(
        model,
        min_lr=1e-4,
        max_lr=1e-3,
        num_training=100,
        mode="linear",
        datamodule=data_module,
    )

    fig = lr_finder.plot(suggest=True, show=True)

    # Pick point based on plot, or get suggestion
    new_lr = lr_finder.suggestion()

    # update hparams of the model
    model.hparams.lr = new_lr
    return model


if __name__ == "__main__":
    main()
