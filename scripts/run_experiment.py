import sys
import os
import torch
from parsers import TOMLParser
from experiment import Analysis

args = sys.argv
config_path = 'U.toml'
if len(args) > 1:
    config_path = args[1]

parser = TOMLParser(config_path)

optimizer = torch.optim.Adam
criterion = torch.nn.MSELoss()

analysis = Analysis(parser, optimizer, criterion, new=True)
analysis.run()
