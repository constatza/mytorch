import os
from parsers import TOMLParser
from preprocessing import format_data, split_data, scale_timeseries
import torch
from sklearn.model_selection import train_test_split


config_file = './config/p-cae.toml'

parser = TOMLParser(config_file)

processed_dir = parser['paths']['input']['root']
os.makedirs(processed_dir, exist_ok=True)

dofs_to_keep_path = parser['paths']['raw']['dofs']
solutions_path = parser['paths']['raw']['data']

dataset = format_data(solutions_path, dofs_to_keep_path).squeeze()

# x_train, x_test, y_train, y_test = train_test_split(dataset, test_size=0.2, shuffle=True)

train_data, test_data = split_data(dataset, test_size=0.2, shuffle=True)
x_train, x_train_means, x_train_stds = scale_timeseries(train_data)
x_test, _, _ = scale_timeseries(test_data, means=x_train_means, stds=x_train_stds)


# torch.save(train_data, parser['paths']['input']['x-train'])
# torch.save(test_data, parser['paths']['input']['x-test'])
torch.save(x_train, parser['paths']['input']['x-train'])
torch.save(x_test, parser['paths']['input']['x-test'])
torch.save(x_train_means, parser['paths']['input']['means'])
torch.save(x_train_stds, parser['paths']['input']['stds'])
torch.save(dataset, parser['paths']['input']['dataset'])
