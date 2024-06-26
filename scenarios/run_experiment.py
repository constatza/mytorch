import sys
import os
import torch
from parsers import TOMLParser, Logger
from experiment import Analysis, AnalysisLoader
from metrics import mse_plus_kl_divergence

config_path = 'scenarios/bio-surrogate/io/u-ffnn.toml'
delete_old = True
args = sys.argv
if len(args) > 1:
    config_path = args[1]

parser = TOMLParser(config_path)

optimizer = torch.optim.Adam
criterion = mse_plus_kl_divergence


analysis_loader = AnalysisLoader(parser, delete_old=delete_old, convolution_dims=1)
logger = Logger(parser.config['paths'])
analysis = Analysis(analysis_loader, logger, optimizer, criterion)
analysis.run()
