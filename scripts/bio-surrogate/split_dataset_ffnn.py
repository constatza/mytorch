import os

import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from parsers import TOMLParser
config_file = './config/u-ffnn-train-test-split.toml'
parser = TOMLParser(config_file)
save_with = 'numpy'

parameters_path = parser['paths']['raw']['parameters']
latent_space_path = parser['paths']['input']['latent-space']

x = np.load(parameters_path)
y = np.load(latent_space_path)


print(f"Latent space shape: {y.shape}")

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)

x_scaler = StandardScaler()
x_train_scaled = x_scaler.fit_transform(x_train)
x_test_scaled = x_scaler.transform(x_test)
y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train)
y_test_scaled = y_scaler.transform(y_test)

os.makedirs(parser['paths']['output']['root'], exist_ok=True)

if save_with=="torch":
    torch.save(x_train, parser['paths']['output']['x-train'])
    torch.save(x_test, parser['paths']['output']['x-test'])
    torch.save(y_train, parser['paths']['output']['y-train'])
    torch.save(y_test, parser['paths']['output']['y-test'])
    torch.save(x_train_scaled, parser['paths']['output']['x-train-scaled'])
    torch.save(x_test_scaled, parser['paths']['output']['x-test-scaled'])
    torch.save(y_train_scaled, parser['paths']['output']['y-train-scaled'])
    torch.save(y_test_scaled, parser['paths']['output']['y-test-scaled'])
elif save_with=="numpy":
    np.save(parser['paths']['output']['x-train'], x_train, allow_pickle=False)
    np.save(parser['paths']['output']['x-test'], x_test, allow_pickle=False)
    np.save(parser['paths']['output']['y-train'], y_train)
    np.save(parser['paths']['output']['y-test'], y_test, allow_pickle=False)
    np.save(parser['paths']['output']['x-train-scaled'], x_train_scaled, allow_pickle=False)
    np.save(parser['paths']['output']['x-test-scaled'], x_test_scaled, allow_pickle=False)
    np.save(parser['paths']['output']['y-train-scaled'], y_train_scaled, allow_pickle=False)
    np.save(parser['paths']['output']['y-test-scaled'], y_test_scaled, allow_pickle=False)

# plot 3d scatter plot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(y_train_scaled[:, 0], y_train_scaled[:, 1], y_train_scaled[:, 2], c=x_train_scaled[:, 1], cmap='viridis')
fig, ax = plt.subplots()
ax.scatter(x_train_scaled[:, 0], x_train_scaled[:, 2], c=x_train_scaled[:, 1], cmap='viridis')
plt.show()