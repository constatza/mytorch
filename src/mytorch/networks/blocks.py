import lightning.pytorch as pl
from mytorch.setup.optimizer import initialize_optimizer
from mytorch.setup.scheduler import initialize_scheduler


class OptimizerSchedulerNetwork(pl.LightningModule):

    def __init__(self, optimizer_config=None, scheduler_config=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.val_loss = None
        self.train_loss = None
        self.test_loss = None

    def configure_optimizers(self):
        optimizer = initialize_optimizer(self.optimizer_config, self.parameters())
        scheduler = initialize_scheduler(self.scheduler_config, optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "frequency": 1,
            },
        }

    def on_train_epoch_end(self) -> None:
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log(
            "train_loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("lr", lr, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        self.log("val_loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
