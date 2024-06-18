
import os
import random
import torch
import numpy as np

from metrics import normalized_rmse
from postprocessing import plot_parametric_predictions
from preprocessing import unscale_timeseries
from parsers import TOMLParser
from networks.blocks import StandardScaler, StandardizedModel



experiment = 'config/u-ffnn.toml'

path_to_config = os.path.join('scripts', 'bio-surrogate', 'config', experiment)
path_to_config = experiment


parser = TOMLParser(path_to_config)
experiment_name = parser['name']
x_test_path = parser['paths']['input']['x-test']
y_test_path = parser['paths']['input']['y-test']
figures_path = parser['paths']['output']['figures']


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = parser['test']['model']
model_name = model.split('.')[0]
model_path = os.path.join(parser['paths']['output']['models'], model)


y_test = np.load(y_test_path, allow_pickle=False)
x_test = np.load(x_test_path, allow_pickle=False)

targets = torch.tensor(y_test).float().squeeze()
inputs = torch.tensor(x_test).float().squeeze()
targets = targets.to(device)
inputs = inputs.to(device)

model = torch.load(model_path)
model = model.to(device)
model.eval()

with torch.no_grad():
    predictions = model(targets).detach()



print("Normalized RMSE:", normalized_rmse(predictions, targets))

dof = 0
y = targets[:, dof].cpu()
y_pred = predictions[:, dof].cpu()
x = x_test[:, 0]

title = f'{model_name} - {experiment_name} - {dof}'
style = {'linestyle': '', 'marker': 'o'}
fig = plot_parametric_predictions(y, y_pred, parameter=x, title=title, kwargs_true=style, kwargs_pred=style)



os.makedirs(figures_path, exist_ok=True)
fig.savefig(os.path.join(figures_path, f'{model_name}_{dof}.png'), dpi=400)


import matplotlib.pyplot as plt
plt.scatter(x_test[:, 1], x_test[:, 2], c=x_test[:, 0], cmap='viridis')
plt.show()
