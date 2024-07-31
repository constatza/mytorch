import lightning
import numpy as np
import torch
from lightning import Trainer

from mytorch.datamodules import FileDataModule
from mytorch.io.readers import read_study
from mytorch.networks.caes import LinearChannelDescentLatent1d
from mytorch.networks.ffnns import FeedForwardNN
from mytorch.pipeline import Pipeline
from mytorch.transforms import MinMaxScaler, NumpyToTensor, StandardScaler

batch_size = 64
rebuild_dataset = False


class ChainedModel(lightning.LightningModule):
    def __init__(self, *models):
        super(ChainedModel, self).__init__()
        self.models = models

    def forward(self, x):
        for model in self.models:
            x = model(x)
        return x

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        y = batch[1]
        y_hat = self.predict_step(batch, batch_idx, dataloader_idx)
        return self.loss(y_hat, y)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x = batch[0]
        y_hat = self.forward(x)
        return y_hat

    def loss(self, y_hat, y):
        return torch.nn.functional.huber_loss(y_hat, y)


def predict(paths):

    feature_transform = Pipeline(
        NumpyToTensor(),
        MinMaxScaler(),
    )
    target_transforms = Pipeline(
        NumpyToTensor(),
        StandardScaler(axis=(0, -1)),
    )

    data_module = FileDataModule(
        save_dir=paths.input,
        paths=(paths.features, paths.targets),
        transforms=(feature_transform, target_transforms),
        rebuild_dataset=False,
        batch_size=batch_size,
        test_size=0.3,
    )

    mynetwork = FeedForwardNN.load_from_checkpoint(
        paths.checkpoints_ffnn / "best-checkpoint.ckpt"
    )

    second_network = LinearChannelDescentLatent1d.load_from_checkpoint(
        paths.checkpoints_cae / "best-checkpoint.ckpt"
    )

    model = ChainedModel(mynetwork, second_network)

    model.eval()
    model.freeze()
    trainer = Trainer(
        default_root_dir=paths.output,
        accelerator="gpu",
    )

    data_module.setup(stage="fit")
    data_module.setup(stage="test")
    trainer.test(mynetwork, datamodule=data_module)
    predictions = trainer.predict(mynetwork, datamodule=data_module)

    stacked_predictions = torch.stack(predictions).squeeze().cpu().numpy()
    np.save(paths.output / "predictions.npy", stacked_predictions)
    return stacked_predictions


def main():
    config_file = "./config.toml"
    config = read_study(config_file)
    predictions = predict(config.paths)


if __name__ == "__main__":
    main()
