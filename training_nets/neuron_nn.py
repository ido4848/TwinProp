from __future__ import print_function
import os
import copy
import pathlib
import h5py
import platform
import sys
import numpy as np
from scipy.stats import norm
from scipy import sparse
import pickle
import time
import argparse
import logging
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix, explained_variance_score
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
import confidenceinterval
from torch.nn import functional as F
from torch.nn import init
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import wandb

sys.path.append(str(pathlib.Path(__file__).parent.parent.absolute()))

from utils.roc_utils import window_roc_curve
from utils.utils import setup_logger, str2bool, ArgumentSaver, AddDefaultInformationAction, AddOutFileAction, TeeAll
from utils.slurm_job import get_job_args
from training_nets.expressive_leaky_memory_neuron import ELM

logger = logging.getLogger(__name__)

def get_nn_from_file(nn_path, map_location):
    nn_dict = torch.load(nn_path, map_location=map_location)
    args = nn_dict['args']
    in_chans = nn_dict['in_chans']

    if args.use_elm:
        model_config = dict()
        model_config["input_to_synapse_routing"] = "neuronio_routing"

        # some legacy params
        model_config["learn_memory_tau"] = args.elm_learn_memory_tau if hasattr(args, 'elm_learn_memory_tau') else False
        model_config["memory_tau_max"] = args.elm_memory_tau_max if hasattr(args, 'elm_memory_tau_max') else 150.0
        model_config["memory_tau_min"] = args.elm_memory_tau_min if hasattr(args, 'elm_memory_tau_min') else 1.0
        
        model_config["mlp_activation"] = "silu"
        model_config["num_input"] = in_chans
        model_config["num_memory"] = args.elm_num_memory
        model_config["num_output"] = 2

        if args.elm_branch_mode:
            model_config["num_branch"] = 45
            model_config["num_synapse_per_branch"] = 100
        else:
            model_config["num_branch"] = None
            model_config["num_synapse_per_branch"] = 1

        model = ELM(**model_config).double()

    else:
        raise ValueError("No model specified")

    if 'use_data_parallel' in args and args.use_data_parallel:
        model = nn.DataParallel(model)        

    model.load_state_dict(nn_dict['model_state_dict'])

    return model, args

