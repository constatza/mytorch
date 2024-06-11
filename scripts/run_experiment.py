import sys
import os
import torch
from parsers import TOMLParser, Logger
from experiment import Analysis, AnalysisLoader

config_path = 'scripts/bio-surrogate/config/p-cae.toml'
delete_old = True
args = sys.argv
if len(args) > 1:
    config_path = args[1]

parser = TOMLParser(config_path)

optimizer = torch.optim.Adam
criterion = torch.nn.MSELoss()


analysis_loader = AnalysisLoader(parser, delete_old=delete_old, convolution_dims=1)
logger = Logger(parser.config['paths'])
analysis = Analysis(analysis_loader, logger, optimizer, criterion)
analysis.run()
