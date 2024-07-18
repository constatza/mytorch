from pathlib import Path

import numpy as np
import torch
from lightning import Trainer
from torch.utils.data import DataLoader, TensorDataset

from mytorch.io.readers import read_toml
from mytorch.networks.caes import ForcedLatentSpace

num_layers = 4
kernel_size = 7
stride = 3
hidden_size = 3

config_path = "config.toml"
config = read_toml(config_path)
root_dir = config["paths"]["output"]
root_dir = Path(root_dir)
# predict
dataset_path = config["paths"]["dataset"]
u = torch.from_numpy(np.load(dataset_path)).float()


model_class = ForcedLatentSpace
test_dataset = TensorDataset(u)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
input_shape = u.shape

# load the model
model_path = root_dir / f"{model_class.__name__}.ckpt"
model_dict = torch.load(model_path)
model = model_class.load_from_checkpoint(
    model_path,
    input_shape=input_shape,
    hidden_size=hidden_size,
    num_layers=num_layers,
    kernel_size=kernel_size,
    stride=stride,
)

trainer = Trainer()
model.eval()
with torch.no_grad():
    latent_data = model.encode(u)
    trainer.test(model, dataloaders=test_loader)

# save
print(latent_data.shape)
np.save(root_dir / "latent_data.npy", latent_data.detach().numpy())
