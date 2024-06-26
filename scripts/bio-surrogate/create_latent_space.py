
import os
import torch

from metrics import normalized_rmse
from parsers import TOMLParser
from networks.blocks import StandardScaler, StandardizedModel
import matplotlib.pyplot as plt

config_file = './config/p-cae.toml'

parser = TOMLParser(config_file)

model = parser['test']['model']
model_name = model.split('.')[0]
model_path = os.path.join(parser['paths']['output']['models'], model)
data_path = parser['paths']['input']['dataset']
means_path = parser['paths']['input']['means']
stds_path = parser['paths']['input']['stds']
latent_path = parser['paths']['input']['latent']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

targets = torch.load(data_path)
targets = torch.from_numpy(targets).float().squeeze()
means = torch.load(means_path).float().squeeze(1)
stds = torch.load(stds_path).float().squeeze(1)
targets = targets.to(device)
print(f"Means shape: {means.shape}")
print(f"Stds shape: {stds.shape}")
print(f"Targets shape: {targets.shape}")


model = torch.load(model_path)


scaler = StandardScaler().fit(means=means, stds=stds)
scaler = scaler.to(device)

model = model.to(device)
model.eval()
with torch.no_grad():
    targets_normalized = (targets - means) / stds

    predictions = model(targets_normalized).detach().squeeze()
    latent_data = model.encode(targets_normalized).detach()
    predictions = predictions * stds + means

    error = normalized_rmse(predictions[:100], targets[:100])


print("Original space shape:", targets.shape)
print("Latent space shape:", latent_data.shape)
print(f"predictions shape: {predictions.shape}")
print("Normalized RMSE:", error)

# Save the predictions
os.makedirs(os.path.dirname(latent_path), exist_ok=True)
torch.save(latent_data, latent_path)

plt.plot(targets[0, 0, :].cpu().numpy())
plt.plot(predictions[0, 0, :].cpu().numpy())
plt.show()
