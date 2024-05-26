import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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
