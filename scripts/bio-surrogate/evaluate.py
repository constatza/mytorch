
import os
import random
import torch
import numpy as np

from metrics import normalized_rmse
from postprocessing import plot_parametric_predictions
from preprocessing import unscale_timeseries
from parsers import TOMLParser
from networks.blocks import StandardScaler, StandardizedModel



experiment = 'p-cae.toml'



parser = TOMLParser(os.path.join('scripts', 'bio-surrogate', 'config', experiment))
experiment_name = parser['name']
x_test_path = parser['paths']['input']['x-test']
means_path = parser['paths']['input']['means']
stds_path = parser['paths']['input']['stds']
figures_path = parser['paths']['output']['figures']


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = parser['test']['model']
model_name = model.split('.')[0]
model_path = os.path.join(parser['paths']['output']['models'], model)


targets = torch.load(x_test_path).float().squeeze()
means = torch.load(means_path).float()
stds = torch.load(stds_path).float()
targets = targets.to(device)

model = torch.load(model_path)

scaler = StandardScaler().fit(means=means, stds=stds)
model = StandardizedModel(model, scaler, normalize_in=False, denormalize_out=True)
model = model.to(device)
model.eval()

with torch.no_grad():
    predictions = model(targets).detach()
    latent_data = model.base_model.encode(targets).detach()
    # denormalize the predictions
    targets = model.inverse_transform(targets).float().detach()
    print(f"Original space shape: {targets.shape[2]}x{targets.shape[3]}")
    print(f"Lattent space shape: {latent_data.shape[0]}x{latent_data.shape[1]}")

print("Normalized RMSE:", normalized_rmse(predictions, targets))

sample = random.randint(0, targets.shape[0]-1)
dof = random.randint(0, targets.shape[2]-1)
# sample, dof = 24, 17
y = targets[sample, 0, dof, :]
y_pred = predictions[sample, 0, dof, :]

title = f"Sample: {sample}, DoF: {dof}"
fig = plot_parametric_predictions(y_pred, y, x_label='Timestep', y_label=parser['variable'], title=title)


os.makedirs(figures_path, exist_ok=True)
fig.savefig(os.path.join(figures_path, f'{model_name}_{sample}_{dof}.png'), dpi=400)



