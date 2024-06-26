
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



parser = TOMLParser(os.path.join('scenarios', 'bio-surrogate', 'io', experiment))
experiment_name = parser['name']
x_test_path = parser['paths']['input']['x-test']
means_path = parser['paths']['input']['means']
stds_path = parser['paths']['input']['stds']
figures_path = parser['paths']['output']['figures']


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = parser['test']['model']
model_name = model.split('.')[0]
model_path = os.path.join(parser['paths']['output']['models'], model)


targets = np.load(x_test_path, allow_pickle=False).squeeze()
means = np.load(means_path)
stds = np.load(stds_path)

targets = torch.tensor(targets).float().squeeze()
means = torch.tensor(means).float().view(1, -1, 1)
stds = torch.tensor(stds).float().view(1, -1, 1)


targets = targets.to(device)
means = means.to(device)
stds = stds.to(device)

model = torch.load(model_path)

scaler = StandardScaler().fit(means=means, stds=stds)
model = model.to(device)
model.eval()

with torch.no_grad():
    predictions = model(targets).detach()
    latent_data = model.encode(targets).detach()
    targets = scaler.inverse_transform(targets)
    predictions = scaler.inverse_transform(predictions)
    error = normalized_rmse(predictions, targets)
    # denormalize the predictions

print(f"Original space shape: {targets.shape[1]}x{targets.shape[2]}")
print(f"Lattent space shape: {latent_data.shape[0]}x{latent_data.shape[1]}")

print("Normalized RMSE:", error)

sample = random.randint(0, targets.shape[0]-1)
dof = random.randint(0, targets.shape[1]-1)
# sample, dof = 24, 17
y = targets[sample, dof, :]
y_pred = predictions[sample, dof, :]

title = f"Sample: {sample}, DoF: {dof}"
fig = plot_parametric_predictions(y_pred, y, x_label='Timestep', y_label=parser['variable'], title=title)


os.makedirs(figures_path, exist_ok=True)
fig.savefig(os.path.join(figures_path, f'{model_name}_{sample}_{dof}.png'), dpi=400)



