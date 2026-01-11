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
import os
import copy
from scipy import sparse

sys.path.append(str(pathlib.Path(__file__).parent.absolute()))
sys.path.append(str(pathlib.Path(__file__).parent.parent.absolute()))

from create_dataset_from_input_folder import get_create_dataset_from_input_folder_parser, create_dataset_from_input_folder

import logging
logger = logging.getLogger(__name__)

class NeuronSimulationWrapper(nn.Module):
    def __init__(self, neuron_model_folder, neuron_simulation_wrapper_folder, args=None, nseg=None, default_nseg=1041, max_segment_length=None):
        super().__init__()

        self.neuron_model_folder = neuron_model_folder
        
        neuron_model_folder_fuller_path = os.path.join("simulating_neurons", neuron_model_folder)
        nseg_file_path = os.path.join(neuron_model_folder_fuller_path, "nseg.txt")

        if nseg is not None:
            self.nseg = nseg
        else:
            if os.path.exists(nseg_file_path):
                # read nseg from this txt file
                with open(nseg_file_path, 'r') as f:
                    self.nseg = int(f.read())
            else:
                self.nseg = default_nseg

        # assumed two synapses per segment (one excitatory, one inhibitory)
        self.count_input_synapses = 2 * self.nseg
            
        self.neuron_simulation_wrapper_folder = neuron_simulation_wrapper_folder
        self.run_counter = 0

        explicit_args = ['--neuron_model_folder', neuron_model_folder]
        if max_segment_length is not None:
            explicit_args += ['--max_segment_length', str(max_segment_length)]

        parser = get_create_dataset_from_input_folder_parser()
        self.create_dataset_args = parser.parse_args(explicit_args)

        if args is not None:
            self.create_dataset_args.__dict__.update(args.__dict__)
            
        # these are relevant only for train / valid (doesn't matter for test)
        self.create_dataset_args.simple_stimulation = True
        self.create_dataset_args.simulation_duration_in_seconds = 2
        self.create_dataset_args.count_simulations_for_train = 1
        self.create_dataset_args.count_simulations_for_valid = 10

        self.create_dataset_args.return_output_spike_times = True
        self.create_dataset_args.save_plots = False
        self.create_dataset_args.add_explicit_padding_for_initialization = False
        self.create_dataset_args.use_finishfile = True
        
        self.simulation_initialization_duration_in_ms = self.create_dataset_args.simulation_initialization_duration_in_ms

    def forward(self, x):
        current_create_dataset_args = copy.deepcopy(self.create_dataset_args)

        current_run_folder = os.path.join(self.neuron_simulation_wrapper_folder, 'run_{}'.format(self.run_counter))
        os.makedirs(current_run_folder, exist_ok=True)

        current_create_dataset_args.input_folder = os.path.join(current_run_folder, 'input_folder')
        os.makedirs(current_create_dataset_args.input_folder, exist_ok=True)

        test_folder = os.path.join(current_create_dataset_args.input_folder, 'test')
        os.makedirs(test_folder, exist_ok=True)

        # save to disk
        x_numpy = x.detach().cpu().numpy()
        window_size = x_numpy.shape[2]
        for i in range(x_numpy.shape[0]):
            all_weighted_spikes = x_numpy[i]
            current_item_folder = os.path.join(test_folder, '{}'.format(i))
            os.makedirs(current_item_folder, exist_ok=True)
            sparse.save_npz(f'{current_item_folder}/all_weighted_spikes.npz', sparse.csr_matrix(all_weighted_spikes))

        current_create_dataset_args.dataset_folder = os.path.join(current_run_folder, 'dataset_folder')
        os.makedirs(current_create_dataset_args.dataset_folder, exist_ok=True)

        # TODO: avoid so much stdout logs from this function
        output_spike_times, create_dataset_total_duration_in_seconds = create_dataset_from_input_folder(current_create_dataset_args)
        
        test_output_spike_times = output_spike_times['test']

        y = np.zeros((x_numpy.shape[0], window_size))
        for i in range(x_numpy.shape[0]):
            current_output_spike_times = test_output_spike_times[i]

            output_spikes_for_window = np.zeros(window_size)
            output_spikes_for_window[current_output_spike_times.astype(int)] = 1

            y[i] = output_spikes_for_window

        # remove input folder now that we have the dataset
        logger.info(f'Removing {current_create_dataset_args.input_folder} after creating dataset')
        os.system(f'rm -rf {current_create_dataset_args.input_folder}')

        # don't remove simulation_initialization_duration_in_ms from left, the user will do that
        # y = y[:, self.simulation_initialization_duration_in_ms:]

        self.run_counter += 1

        return torch.from_numpy(y).float()

    def get_model_shape(self):
        return ((self.count_input_synapses,), True, (1,), -self.simulation_initialization_duration_in_ms, 0.5)