import os

import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from parsers import TOMLParser
config_file = './config/p-ffnn-train-test-split.toml'
parser = TOMLParser(config_file)

parameters_path = parser['paths']['raw']['parameters']
latent_space_path = parser['paths']['input']['latent-space']

x = np.load(parameters_path)

y = torch.load(latent_space_path).cpu().numpy()
print(f"Latent space shape: {y.shape}")

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)

x_scaler = StandardScaler()
x_train_scaled = x_scaler.fit_transform(x_train)
x_test_scaled = x_scaler.transform(x_test)
y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train)
y_test_scaled = y_scaler.transform(y_test)



os.makedirs(parser['paths']['output']['root'], exist_ok=True)
torch.save(x_train, parser['paths']['output']['x-train'])
torch.save(x_test, parser['paths']['output']['x-test'])
torch.save(y_train, parser['paths']['output']['y-train'])
torch.save(y_test, parser['paths']['output']['y-test'])
torch.save(x_train_scaled, parser['paths']['output']['x-train-scaled'])
torch.save(x_test_scaled, parser['paths']['output']['x-test-scaled'])
torch.save(y_train_scaled, parser['paths']['output']['y-train-scaled'])
torch.save(y_test_scaled, parser['paths']['output']['y-test-scaled'])

import matplotlib.pyplot as plt
plt.scatter(x_train[:, 0], y_train[:, 1])
plt.show()