import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader


def reshaper(func):
    """Decorator that scales data with format (num_samples, num_channels, num_dofs, num_timesteps)."""

    def wrapper(dataset, *args, **kwargs):

        # Reshape data to (num_samples, num_channels*num_dofs* num_timesteps)

        original_shape = dataset.shape
        num_features = np.prod(original_shape[1:])
        reshaped_dataset = dataset.reshape(original_shape[0], num_features)
        # Scale data
        reshaped_dataset = func(reshaped_dataset, *args, **kwargs)
        # Reshape data back to (num_samples, num_channels, num_dofs, num_timesteps)
        if isinstance(reshaped_dataset, tuple) or isinstance(reshaped_dataset, list):
            output = [d.reshape(-1, *original_shape[1:]) for d in reshaped_dataset]
        else:
            output = reshaped_dataset.reshape(original_shape)
        return output

    return wrapper


@reshaper
def scale_data(dataset, scaler=StandardScaler()):
    """Scale the data using a standard scaler."""
    # Scale data
    scaler = StandardScaler()
    dataset = scaler.fit_transform(dataset)
    return dataset


@reshaper
def split_data(dataset, *args, **kwargs):
    """Split the data into training and testing sets."""
    # Split data
    return train_test_split(dataset, *args, **kwargs)


def create_dataloaders(*data, batch_size=32):
    """Create dataloaders from the data."""
    dataloaders = []
    for datum in data:
        if not isinstance(datum, torch.Tensor):
            raise ValueError('Data must be torch tensors.')
        else:
            dataloaders.append(DataLoader(datum, batch_size=batch_size, shuffle=True))
    return dataloaders


def training_autoencoder(model, x_train, x_val, optimizer, criterion=nn.MSELoss(), num_epochs=100, batch_size=32):
    """Training loop for the model with both training and validation loss."""
    # use the GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} for training.')

    # Create dataloaders
    dataloader_train, dataloader_val = create_dataloaders(x_train, x_val, batch_size=batch_size)

    # Initialize lists to store training and validation losses
    train_losses = []
    val_losses = []
    model.to(device)
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for data in dataloader_train:
            data = data.to(device)
            optimizer.zero_grad()
            recon = model(data)
            loss = criterion(recon, data)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_losses.append(train_loss / len(dataloader_train))

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in dataloader_val:
                data = data.to(device)
                recon = model(data)
                loss = criterion(recon, data)
                val_loss += loss.item()
            val_losses.append(val_loss / len(dataloader_val))
        # print with scientific notation and 6 decimal places
        print(f'Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_losses[-1]:.6e} | Val Loss: {val_losses[-1]:.6e}')

    return train_losses, val_losses


def format_data(solutions_path, dofs_to_keep_path):
    """Import the data from the path and keep only the dofs specified."""
    # Load the dofs to keep
    import os
    dirname = os.path.dirname(solutions_path)
    solutions = np.load(solutions_path).T
    # dofs to keep
    dofs = np.loadtxt(dofs_to_keep_path).astype(int)

    solutions = np.transpose(solutions, (2, 0, 1))
    solutions = solutions[:, :, :-8]
    solutions = solutions[:, dofs, :]
    solutions = solutions[:, np.newaxis, :, :]
    np.save(os.path.join(dirname, 'formatted_solutions.npy'), solutions)
    print(solutions.shape)
    return solutions


if __name__ == "__main__":
    # load data
    solutions_path = '../data/solutions500/porousSolutions.npy'
    dofs_to_keep_path = '../data/dofs/pInnerRemainderfacedofs.txt'
    dataset = format_data(solutions_path, dofs_to_keep_path)
