import matplotlib.pyplot as plt
import numpy as np
import torch

from networks.caes import CAE1dLinear
from postprocessing import plot_losses, plot_parametric_predictions, load_model

# Load the model


x_test = torch.load('../data/solutions500/x_test.npy')
train_loss = np.load('../data/models/train_loss.npy')
val_loss = np.load('../data/models/val_loss.npy')

input_shape = x_test.shape[2:]
model = CAE1dLinear(input_shape, num_layers=8)
model_name = model.__class__.__name__
model = load_model(model, f'../data/models/{model_name}.pth')

criterion = lambda x, y: torch.mean(torch.abs(x - y) / torch.abs(y))

model.eval()
with torch.no_grad():
    x_recon = model(x_test)
    loss = criterion(x_recon, x_test)
    print(f'Test loss: {loss.item():.6e}')

plot_losses(train_loss, val_loss)

which = 20
sample = 60

plot_parametric_predictions(x_recon[sample, 0, which, :], x_test[sample, 0, which, :], torch.arange(x_test.shape[-1]))
plt.show()
