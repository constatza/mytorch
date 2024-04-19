import matplotlib.pyplot as plt
import torch
from preprocessing import reshaper

def plot_losses(train_losses, val_losses):
    """Plot the training and validation losses."""

    fig, ax = plt.subplots()
    ax.plot(train_losses, label='Training loss')
    ax.plot(val_losses, label='Validation loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    # set the scale to log
    ax.set_yscale('log')
    ax.legend()
    return fig


def save_model(model, path):
    """Save the model to the path."""
    torch.save(model.state_dict(), path)

def load_model(model, path):
    """Load the model from the path."""
    model.load_state_dict(torch.load(path))
    return model

def plot_parametric_predictions(y_pred, y_true, parameter):
    """Plot the predictions of the model."""
    fig, ax = plt.subplots()
    ax.scatter(parameter, y_true, label='True', marker='+')
    ax.scatter(parameter, y_pred, label='Predicted', marker='o')
    ax.set_xlabel('Parameter')
    ax.set_ylabel('Output')
    ax.legend()
    return fig

