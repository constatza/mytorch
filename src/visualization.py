
import numpy as np
import torch
from src.postprocessing import plot_losses, plot_parametric_predictions, load_model
from src.blocks import CAE2d
import matplotlib.pyplot as plt


# Load the model


x_test = torch.load('../data/solutions/x_test.pt')
train_loss = np.load('../data/models/train_loss.npy')
val_loss = np.load('../data/models/val_loss.npy')


model = CAE2d(x_test.shape[2:], 1000)
model = load_model(model, '../data/models/CAE_1000.pth')

criterion = lambda x, y: torch.mean(torch.abs(x - y)/torch.abs(y))

model.eval()
with torch.no_grad():
    x_recon = model(x_test)
    loss = criterion(x_recon, x_test)
    print(f'Test loss: {loss.item():.6e}')

plot_losses(train_loss, val_loss)
plot_parametric_predictions(x_recon[0, 0, 0, :], x_test[0, 0, 0, :], torch.arange(x_test.shape[-1]))
plt.show()