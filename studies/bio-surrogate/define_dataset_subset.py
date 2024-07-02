import os
import torch
import numpy as np

from parsers import TOMLParser
from preprocessing import format_data, split_data, scale_timeseries
from sklearn.model_selection import train_test_split


config_file = './config/p-cae.toml'

parser = TOMLParser(config_file)

processed_dir = parser['paths']['input']['root']
os.makedirs(processed_dir, exist_ok=True)

dofs_to_keep_path = parser['paths']['raw']['dofs']
solutions_path = parser['paths']['raw']['data']

dataset = format_data(solutions_path, dofs_to_keep_path).squeeze()


train_data, test_data = split_data(dataset, test_size=0.2, shuffle=True)
x_train, x_train_means, x_train_stds = scale_timeseries(train_data)
x_test, _, _ = scale_timeseries(test_data, means=x_train_means, stds=x_train_stds)


x_train = x_train.squeeze().float().cpu().numpy()
x_test = x_test.squeeze().float().cpu().numpy()
x_train_stds = x_train_stds.squeeze().float().cpu().numpy()
x_train_means = x_train_means.squeeze().float().cpu().numpy()


os.makedirs(parser['paths']['input']['root'], exist_ok=True)
np.save(parser['paths']['input']['y-train'], train_data, allow_pickle=False)
np.save(parser['paths']['input']['y-test'], test_data, allow_pickle=False)
np.save(parser['paths']['input']['x-train'], x_train, allow_pickle=False)
np.save(parser['paths']['input']['x-test'], x_test, allow_pickle=False)
np.save(parser['paths']['input']['means'], x_train_means, allow_pickle=False)
np.save(parser['paths']['input']['stds'], x_train_stds, allow_pickle=False)
np.save(parser['paths']['input']['dataset'], dataset, allow_pickle=False)
