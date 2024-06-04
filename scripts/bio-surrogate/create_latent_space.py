
import os
import torch

from metrics import normalized_rmse
from parsers import TOMLParser
from networks.blocks import StandardScaler, StandardizedModel

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

scaler = StandardScaler(means=means, stds=stds)

cae = torch.load(model_path)
cae = cae.to(device)
cae.eval()
with torch.no_grad():
    input = scaler(targets)
    latent_data = cae.encoder(input).detach()

    predictions = cae.decoder(latent_data).detach()[:, :, :730]

    predictions = scaler.inverse(predictions).detach()
    error = normalized_rmse(predictions[:100], targets[:100])


print("Normalized RMSE:", error)

# reduce to vector
latent_data = latent_data.view(latent_data.size(0), -1)
# Save the predictions
os.makedirs(os.path.dirname(latent_path), exist_ok=True)
torch.save(latent_data, latent_path)


