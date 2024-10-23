import shutil

import numpy as np
import torch
from lightning.pytorch.callbacks import TQDMProgressBar, ModelCheckpoint, ModelSummary

from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.trainer import Trainer

from mytorch.datamodules import FileDataModule
from mytorch.io.readers import read_study
from mytorch.networks.caes import BasicCAE
from mytorch.pipeline import Pipeline
from mytorch.transforms import StandardScaler, NumpyToTensor
from mytorch.utils.system import tune_lr
from mytorch.utils.optuna import run_optuna_study


def training(hparams):

    torch.set_float32_matmul_precision("high")

    paths = hparams["paths"]
    root_dir = paths["workdir"]

    train_config = hparams["train"]
    model_config = hparams["model"]
    optimizer_config = hparams["optimizer"]
    info_config = hparams["info"]

    logger = TensorBoardLogger(save_dir=paths["output"], name="logs")

    callbacks = [
        ModelSummary(max_depth=2),
        TQDMProgressBar(),
        ModelCheckpoint(
            monitor="val_loss",
            dirpath=paths["checkpoints"],
            filename="best-checkpoint",
            save_top_k=2,
            mode="min",
            save_last=True,
            every_n_epochs=10,
        ),
    ]

    x_transforms = Pipeline(
        NumpyToTensor(),
        StandardScaler(dim=(0, -1)),
    )

    data_module = FileDataModule(
        save_dir=paths["output"],
        paths=paths["features"],
        transforms=x_transforms,
        test_size=0.3,
        val_size=0.5,
        batch_size=500,
    )

    data_module.prepare_data()
    data_module.setup(stage="fit")

    trainer = Trainer(
        default_root_dir=root_dir,
        max_epochs=train_config["num_epochs"],
        accelerator="gpu",
        callbacks=callbacks,
        # gradient_clip_val=0.5,
        enable_progress_bar=True,
        reload_dataloaders_every_n_epochs=10,
        logger=logger,
        # fast_dev_run=True,
    )
    model = BasicCAE(
        input_shape=data_module.input_shape,
        latent_size=model_config["latent_size"],
        num_layers=model_config["num_layers"],
        kernel_size=model_config["kernel_size"],
        reduced_channels=model_config["reduced_channels"],
        reduced_timesteps=model_config["reduced_timesteps"],
    )

    model = tune_lr(trainer, model, data_module) if info_config["tune"] else model

    if info_config["continue_training"]:
        trainer.fit(model, datamodule=data_module, ckpt_path="last")
    else:
        # delete the checkpoint directory
        shutil.rmtree(paths["checkpoints"], ignore_errors=True)
        shutil.rmtree(paths["logs"], ignore_errors=True)
        trainer.fit(model, datamodule=data_module)

    data_module.setup(stage="test")
    test_score = trainer.test(model, datamodule=data_module)

    return test_score


if __name__ == "__main__":
    config_file = r"./config.toml"
    config = read_study(config_file)
    run_optuna_study(config_file, training)
