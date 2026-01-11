import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal
import pickle
import time
import sys
import pathlib
from training_nets.neuron_nn import get_nn_from_file

sys.path.append(str(pathlib.Path(__file__).parent.parent.absolute()))

from utils.utils import setup_logger

import logging
logger = logging.getLogger(__name__)


class NeuronNnWrapper(nn.Module):
    # TODO: 300 might not be long enough for some nets
    def __init__(self, neuron_nn_file, threshold=0.9, default_network_temporal_extent=300, return_all=False):
        super().__init__()

        self.neuron_nn_file = neuron_nn_file
        self.nn, self.nn_args = get_nn_from_file(neuron_nn_file, torch.device("cpu"))
        test_name = "test"
        if self.nn_args.use_valid_as_test:
            test_name = "valid"

        if 'module' in dir(self.nn):
            self.the_nn = self.nn.module
        else:
            self.the_nn = self.nn


        if self.nn_args.use_elm:
            self.count_axons = list(self.the_nn.parameters())[0].shape[0]
        else:
            self.count_axons = self.the_nn.convs[0].in_channels
        self.threshold = threshold
        
        try:
            result_dict = pickle.load(open(f"{neuron_nn_file}_normalized_{test_name}_results.pkl", "rb"))
            self.network_temporal_extent = result_dict['network_temporal_extent']
            self.normalized_auc = result_dict['auc']
        except:
            self.network_temporal_extent = default_network_temporal_extent

            logger.info(f"Could not load normalized {test_name} results, using the default network_temporal_extent = {default_network_temporal_extent}")
            self.normalized_auc = -1

        logger.info(f"Loaded neuron_nn of {self.count_axons} axons, {self.network_temporal_extent} temporal extent, with {self.normalized_auc} normalized {test_name} auc, from {neuron_nn_file}")
        logger.info(f"Using threshold of {threshold} (default is 0.9)")

        self.return_all = return_all


    def forward(self, x):
        if self.nn_args.use_elm:
            x = x.transpose(1,2)

        outputs = self.nn(x)

        if self.nn_args.use_elm:
            out_v = outputs[..., 1].unsqueeze(-1)
            out_sp = outputs[..., 0].unsqueeze(-1)
        else:
            out_v, out_sp = outputs

        if self.return_all:
            return out_v, out_sp
        
        # using spikes, removing last dimension
        out = out_sp.squeeze(2)

        # don't remove network_temporal_extent from left, the user will do that
        # out = out[:, self.network_temporal_extent:]

        # apply sigmoid to output
        out = torch.sigmoid(out)

        return out

    def get_model_shape(self):
        return ((self.count_axons,), True, (1,), -self.network_temporal_extent, self.threshold)