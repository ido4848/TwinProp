from __future__ import print_function
import pathlib
import importlib
import copy
import os
import peakutils
import h5py
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import signal
import pickle
import time
import argparse
from scipy import sparse

sys.path.append(str(pathlib.Path(__file__).parent.parent.absolute()))
sys.path.append(str(pathlib.Path(__file__).parent.absolute()))

from utils.utils import setup_logger, str2bool, ArgumentSaver, AddOutFileAction, TeeAll, AddDefaultInformationAction
from utils.slurm_job import get_job_args
from training_nets.constrained_linear import ConstrainedLinear
from simulating_neurons.neuron_plotter import NeuronPlotter, MAX_CM, MIN_CM

import logging
logger = logging.getLogger(__name__)

# mock
neuron = None
h = None
gui = None

def generate_input_spike_rates_for_simulation(args, sim_duration_ms, count_exc_netcons, count_inh_netcons):
    auxiliary_information = {}

    # randomly sample inst rate (with some uniform noise) smoothing sigma
    keep_inst_rate_const_for_ms = args.inst_rate_sampling_time_interval_options_ms[np.random.randint(len(args.inst_rate_sampling_time_interval_options_ms))]
    keep_inst_rate_const_for_ms += int(2 * args.inst_rate_sampling_time_interval_jitter_range * np.random.rand() - args.inst_rate_sampling_time_interval_jitter_range)
    
    # randomly sample smoothing sigma (with some uniform noise)
    temporal_inst_rate_smoothing_sigma = args.temporal_inst_rate_smoothing_sigma_options_ms[np.random.randint(len(args.temporal_inst_rate_smoothing_sigma_options_ms))]
    temporal_inst_rate_smoothing_sigma += int(2 * args.temporal_inst_rate_smoothing_sigma_jitter_range * np.random.rand() - args.temporal_inst_rate_smoothing_sigma_jitter_range)
    
    count_inst_rate_samples = int(np.ceil(float(sim_duration_ms) / keep_inst_rate_const_for_ms))
    
    # create the coarse inst rates with units of "total spikes per tree per 100 ms"
    count_exc_spikes_per_100ms   = np.random.uniform(low=args.effective_count_exc_spikes_per_synapse_per_100ms_range[0] * count_exc_netcons,
     high=args.effective_count_exc_spikes_per_synapse_per_100ms_range[1] * count_exc_netcons, size=(1,count_inst_rate_samples))

    if args.adaptive_inh:
        count_inh_spikes_per_synapse_per_100ms_low_range = np.maximum(0, count_exc_spikes_per_100ms / count_exc_netcons + args.effective_adaptive_inh_additive_factor_per_synapse_per_100ms_range[0])
        count_inh_spikes_per_synapse_per_100ms_high_range = count_exc_spikes_per_100ms / count_exc_netcons + args.effective_adaptive_inh_additive_factor_per_synapse_per_100ms_range[1]
        count_inh_spikes_per_100ms = np.random.uniform(low=count_inh_spikes_per_synapse_per_100ms_low_range * count_inh_netcons,
         high=count_inh_spikes_per_synapse_per_100ms_high_range * count_inh_netcons, size=(1,count_inst_rate_samples))
    else:
        count_inh_spikes_per_100ms  = np.random.uniform(low=args.effective_count_inh_spikes_per_synapse_per_100ms_range[0] * count_inh_netcons, high=args.effective_count_inh_spikes_per_synapse_per_100ms_range[1] * count_inh_netcons, size=(1,count_inst_rate_samples))

    # convert to units of "per_netcon_per_1ms"
    exc_spike_rate_per_netcon_per_1ms   = count_exc_spikes_per_100ms   / (count_exc_netcons  * 100.0)
    inh_spike_rate_per_netcon_per_1ms  = count_inh_spikes_per_100ms  / (count_inh_netcons  * 100.0)
            
    # kron by space (uniform distribution across branches per tree)
    exc_spike_rate_per_netcon_per_1ms   = np.kron(exc_spike_rate_per_netcon_per_1ms  , np.ones((count_exc_netcons,1)))
    inh_spike_rate_per_netcon_per_1ms  = np.kron(inh_spike_rate_per_netcon_per_1ms , np.ones((count_inh_netcons,1)))
        
    # vstack basal and apical
    exc_spike_rate_per_netcon_per_1ms  = np.vstack((exc_spike_rate_per_netcon_per_1ms))
    inh_spike_rate_per_netcon_per_1ms = np.vstack((inh_spike_rate_per_netcon_per_1ms))

    exc_spatial_multiplicative_randomness_delta = np.random.uniform(args.exc_spatial_multiplicative_randomness_delta_range[0], args.exc_spatial_multiplicative_randomness_delta_range[1])
    if np.random.rand() < args.same_exc_inh_spatial_multiplicative_randomness_delta_prob:
        inh_spatial_multiplicative_randomness_delta = exc_spatial_multiplicative_randomness_delta
    else:
        inh_spatial_multiplicative_randomness_delta = np.random.uniform(args.inh_spatial_multiplicative_randomness_delta_range[0], args.inh_spatial_multiplicative_randomness_delta_range[1])

    # add some spatial multiplicative randomness (that will be added to the sampling noise)
    if args.spatial_multiplicative_randomness and np.random.rand() < args.exc_spatial_multiplicative_randomness_delta_prob:
        exc_spike_rate_per_netcon_per_1ms  = np.random.uniform(low=1 - exc_spatial_multiplicative_randomness_delta, high=1 + exc_spatial_multiplicative_randomness_delta, size=exc_spike_rate_per_netcon_per_1ms.shape) * exc_spike_rate_per_netcon_per_1ms
    if args.spatial_multiplicative_randomness and np.random.rand() < args.inh_spatial_multiplicative_randomness_delta_prob:
        inh_spike_rate_per_netcon_per_1ms = np.random.uniform(low=1 - inh_spatial_multiplicative_randomness_delta, high=1 + inh_spatial_multiplicative_randomness_delta, size=inh_spike_rate_per_netcon_per_1ms.shape) * inh_spike_rate_per_netcon_per_1ms

    # kron by time (crop if there are leftovers in the end) to fill up the time to 1ms time bins
    exc_spike_rate_per_netcon_per_1ms  = np.kron(exc_spike_rate_per_netcon_per_1ms , np.ones((1,keep_inst_rate_const_for_ms)))[:,:sim_duration_ms]
    inh_spike_rate_per_netcon_per_1ms = np.kron(inh_spike_rate_per_netcon_per_1ms, np.ones((1,keep_inst_rate_const_for_ms)))[:,:sim_duration_ms]
    
    # filter the inst rates according to smoothing sigma
    smoothing_window = signal.gaussian(1.0 + args.temporal_inst_rate_smoothing_sigma_mult * temporal_inst_rate_smoothing_sigma, std=temporal_inst_rate_smoothing_sigma)[np.newaxis,:]
    smoothing_window /= smoothing_window.sum()
    netcon_inst_rate_exc_smoothed  = signal.convolve(exc_spike_rate_per_netcon_per_1ms,  smoothing_window, mode='same')
    netcon_inst_rate_inh_smoothed = signal.convolve(inh_spike_rate_per_netcon_per_1ms, smoothing_window, mode='same')

    # add synchronization if necessary
    if np.random.rand() < args.synchronization_prob:
        time_ms = np.arange(0, sim_duration_ms)

        exc_synchronization_period = np.random.randint(args.exc_synchronization_period_range[0], args.exc_synchronization_period_range[1])
        if np.random.rand() < args.same_exc_inh_synchronization_prob:
            inh_synchronization_period = np.random.randint(args.inh_synchronization_period_range[0], args.inh_synchronization_period_range[1])
        else:
            inh_synchronization_period = exc_synchronization_period

        exc_synchronization_profile_mult = np.random.uniform(args.exc_synchronization_profile_mult_range[0], args.exc_synchronization_profile_mult_range[1])
        if np.random.rand() < args.same_exc_inh_synchronization_profile_mult_prob:
            inh_synchronization_profile_mult = np.random.uniform(args.inh_synchronization_profile_mult_range[0], args.inh_synchronization_profile_mult_range[1])
        else:
            inh_synchronization_profile_mult = exc_synchronization_profile_mult

        exc_temporal_profile = exc_synchronization_profile_mult * np.sin(2 * np.pi * time_ms / exc_synchronization_period) + 1.0
        inh_temporal_profile = inh_synchronization_profile_mult * np.sin(2 * np.pi * time_ms / inh_synchronization_period) + 1.0
        
        temp_exc_mult_factor = np.tile(exc_temporal_profile[np.newaxis], (netcon_inst_rate_exc_smoothed.shape[0], 1))
        temp_inh_mult_factor = np.tile(inh_temporal_profile[np.newaxis], (netcon_inst_rate_inh_smoothed.shape[0], 1))

        if np.random.rand() >= args.no_exc_synchronization_prob:
            netcon_inst_rate_exc_smoothed  = temp_exc_mult_factor * netcon_inst_rate_exc_smoothed
            auxiliary_information['exc_synchronization_period'] = exc_synchronization_period

        if np.random.rand() >= args.no_inh_synchronization_prob:
            netcon_inst_rate_inh_smoothed = temp_inh_mult_factor * netcon_inst_rate_inh_smoothed
            auxiliary_information['inh_synchronization_period'] = inh_synchronization_period

        logger.info(f'on synchronization mode, with {exc_synchronization_period=}, {inh_synchronization_period=}')

    # remove inhibition if necessary
    if np.random.rand() < args.remove_inhibition_prob:
        # reduce inhibition to zero
        netcon_inst_rate_inh_smoothed[:] = 0

        # reduce average excitatory firing rate
        excitation_mult_factor = np.random.uniform(args.remove_inhibition_exc_mult_range[0], args.remove_inhibition_exc_mult_range[1]) + np.random.uniform(args.remove_inhibition_exc_mult_jitter_range[0], args.remove_inhibition_exc_mult_jitter_range[1]) * np.random.rand()
        netcon_inst_rate_exc_smoothed = excitation_mult_factor * netcon_inst_rate_exc_smoothed

        logger.info(f'on remove inhibition mode with {excitation_mult_factor=}')

    # randomly deactivate part of the synapses
    if np.random.rand() < args.deactivate_synapses_prob:
        count_exc_synapses_to_deactivate = int(np.random.uniform(args.exc_deactivate_synapses_ratio_range[0] * count_exc_netcons, args.exc_deactivate_synapses_ratio_range[1] * count_exc_netcons))
        count_inh_synapses_to_deactivate = int(np.random.uniform(args.inh_deactivate_synapses_ratio_range[0] * count_inh_netcons, args.inh_deactivate_synapses_ratio_range[1] * count_inh_netcons))

        if netcon_inst_rate_exc_smoothed.shape == netcon_inst_rate_inh_smoothed.shape and np.random.rand() < args.same_exc_inh_deactivation_count:
            count_inh_synapses_to_deactivate = count_exc_synapses_to_deactivate

        exc_synapses_to_deactivate = np.random.choice(range(netcon_inst_rate_exc_smoothed.shape[0]), count_exc_synapses_to_deactivate, replace=False)
        inh_synapses_to_deactivate = np.random.choice(range(netcon_inst_rate_inh_smoothed.shape[0]), count_inh_synapses_to_deactivate, replace=False)

        if netcon_inst_rate_exc_smoothed.shape == netcon_inst_rate_inh_smoothed.shape and np.random.rand() < args.same_exc_inh_deactivations:
            inh_synapses_to_deactivate = exc_synapses_to_deactivate

        if np.random.rand() >= args.no_inh_deactivation_prob:
            netcon_inst_rate_inh_smoothed[inh_synapses_to_deactivate] = 0

        if np.random.rand() >= args.no_exc_deactivation_prob:
            netcon_inst_rate_exc_smoothed[exc_synapses_to_deactivate] = 0

        logger.info(f'on deactivate synapses mode, with {count_exc_synapses_to_deactivate} exc synapses to deactivate and {count_inh_synapses_to_deactivate} inh synapses to deactivate')

    # add random spatial clustering througout the entire simulation
    if np.random.rand() < args.spatial_clustering_prob:
        exc_cluster_sizes = np.random.uniform(args.exc_spatial_cluster_size_ratio_range[0] * count_exc_netcons, args.exc_spatial_cluster_size_ratio_range[1] * count_exc_netcons, count_exc_netcons).astype(int)
        exc_cluster_sizes = exc_cluster_sizes[:np.argmax(np.cumsum(exc_cluster_sizes)>count_exc_netcons)+1]

        if np.random.rand() < args.random_exc_spatial_clusters_prob:
            exc_curr_clustering_row = np.array(list(range(count_exc_netcons)))
            all_indices = np.array(list(range(count_exc_netcons)))
            for i, exc_cluster_size in enumerate(exc_cluster_sizes):
                if len(all_indices) == 0:
                    break
                if len(all_indices) < exc_cluster_size:
                    exc_cluster_size = len(all_indices)
                chosen_indices_indices = np.random.choice(range(len(all_indices)), exc_cluster_size, replace=False)
                exc_curr_clustering_row[all_indices[chosen_indices_indices]] = i
                all_indices = np.delete(all_indices, chosen_indices_indices)
        else:
            exc_curr_clustering_row = np.array(sum([[i for _ in range(t+1)] for i, t in enumerate(exc_cluster_sizes)], []))[:count_exc_netcons]

        exc_count_spatial_clusters = np.unique(exc_curr_clustering_row).shape[0]
        exc_count_active_clusters = int(exc_count_spatial_clusters * np.random.uniform(args.active_exc_spatial_cluster_ratio_range[0], args.active_exc_spatial_cluster_ratio_range[1]))
        exc_active_clusters = np.random.choice(np.unique(exc_curr_clustering_row), size=exc_count_active_clusters, replace=False)
        exc_spatial_mult_factor = np.tile(np.isin(exc_curr_clustering_row, exc_active_clusters)[:,np.newaxis], (1, netcon_inst_rate_exc_smoothed.shape[1]))
        
        if np.random.rand() >= args.no_exc_spatial_clustering_prob:
            auxiliary_information['exc_curr_clustering_row'] = exc_curr_clustering_row
            auxiliary_information['exc_count_spatial_clusters'] = exc_count_spatial_clusters
            auxiliary_information['exc_count_active_clusters'] = exc_count_active_clusters
            auxiliary_information['exc_active_clusters'] = exc_active_clusters
            auxiliary_information['exc_spatial_mult_factor'] = exc_spatial_mult_factor
            try:
                netcon_inst_rate_exc_smoothed  = exc_spatial_mult_factor * netcon_inst_rate_exc_smoothed
            except:
                logger.info(f"failed to multiply exc spatial mult factor {exc_spatial_mult_factor.shape} "+
                f"with netcon_inst_rate_exc_smoothed {netcon_inst_rate_exc_smoothed.shape}, exc_count_spatial_clusters {exc_count_spatial_clusters}, "+
                 f"exc_count_active_clusters {exc_count_active_clusters}, exc_active_clusters {exc_active_clusters.shape}")

        if netcon_inst_rate_exc_smoothed.shape == netcon_inst_rate_inh_smoothed.shape and np.random.rand() < args.same_exc_inh_spatial_clustering_prob:
            inh_curr_clustering_row = exc_curr_clustering_row
            inh_count_spatial_clusters = exc_count_spatial_clusters
            inh_count_active_clusters = exc_count_active_clusters
            inh_active_clusters = exc_active_clusters
            inh_spatial_mult_factor = exc_spatial_mult_factor
        else:
            inh_cluster_sizes = np.random.uniform(args.inh_spatial_cluster_size_ratio_range[0] * count_inh_netcons, args.inh_spatial_cluster_size_ratio_range[1] * count_inh_netcons, count_inh_netcons).astype(int)
            inh_cluster_sizes = inh_cluster_sizes[:np.argmax(np.cumsum(inh_cluster_sizes)>count_inh_netcons)+1]
                
            if np.random.rand() < args.random_inh_spatial_clusters_prob:
                inh_curr_clustering_row = np.array(list(range(count_inh_netcons)))
                all_indices = np.array(list(range(count_inh_netcons)))
                for i, inh_cluster_size in enumerate(inh_cluster_sizes):
                    if len(all_indices) == 0:
                        break
                    if len(all_indices) < inh_cluster_size:
                        inh_cluster_size = len(all_indices)
                    chosen_indices_indices = np.random.choice(range(len(all_indices)), inh_cluster_size, replace=False)
                    inh_curr_clustering_row[all_indices[chosen_indices_indices]] = i
                    all_indices = np.delete(all_indices, chosen_indices_indices)
            else:
                inh_curr_clustering_row = np.array(sum([[i for _ in range(t+1)] for i, t in enumerate(inh_cluster_sizes)], []))[:count_inh_netcons]

            inh_count_spatial_clusters = np.unique(inh_curr_clustering_row).shape[0]
            inh_count_active_clusters = int(inh_count_spatial_clusters * np.random.uniform(args.active_inh_spatial_cluster_ratio_range[0], args.active_inh_spatial_cluster_ratio_range[1]))
            inh_active_clusters = np.random.choice(np.unique(inh_curr_clustering_row), size=inh_count_active_clusters, replace=False)
            inh_spatial_mult_factor = np.tile(np.isin(inh_curr_clustering_row, inh_active_clusters)[:,np.newaxis], (1, netcon_inst_rate_inh_smoothed.shape[1]))

        if np.random.rand() >= args.no_inh_spatial_clustering_prob:
            auxiliary_information['inh_curr_clustering_row'] = inh_curr_clustering_row
            auxiliary_information['inh_active_clusters'] = inh_active_clusters
            auxiliary_information['inh_count_spatial_clusters'] = inh_count_spatial_clusters
            auxiliary_information['inh_count_active_clusters'] = inh_count_active_clusters
            auxiliary_information['inh_spatial_mult_factor'] = inh_spatial_mult_factor

            try:
                netcon_inst_rate_inh_smoothed  = inh_spatial_mult_factor * netcon_inst_rate_inh_smoothed
            except Exception as e:
                logger.info(f"failed to multiply inh spatial mult factor {inh_spatial_mult_factor.shape} "+
                f"with netcon_inst_rate_inh_smoothed {netcon_inst_rate_inh_smoothed.shape}, inh_count_spatial_clusters {inh_count_spatial_clusters}, "+
                 f"inh_count_active_clusters {inh_count_active_clusters}, inh_active_clusters {inh_active_clusters.shape}")
        
        logger.info(f'on spatial clustering mode, with {exc_count_active_clusters} active exc clusters out of {exc_count_spatial_clusters} total, and {inh_count_active_clusters} active inh clusters out of {inh_count_spatial_clusters} total')

    if netcon_inst_rate_exc_smoothed.shape == netcon_inst_rate_inh_smoothed.shape and np.random.rand() < args.same_exc_inh_inst_rate_prob:
        netcon_inst_rate_inh_smoothed = netcon_inst_rate_exc_smoothed

    return netcon_inst_rate_exc_smoothed, netcon_inst_rate_inh_smoothed, auxiliary_information

def sample_spikes_from_rates(args, netcon_inst_rate_ex, netcon_inst_rate_inh):
    # sample the instantanous spike prob and then sample the actual spikes
    exc_inst_spike_prob = np.random.exponential(scale=netcon_inst_rate_ex)
    exc_spikes_bin      = np.random.rand(exc_inst_spike_prob.shape[0], exc_inst_spike_prob.shape[1]) < exc_inst_spike_prob
    
    inh_inst_spike_prob = np.random.exponential(scale=netcon_inst_rate_inh)
    inh_spikes_bin      = np.random.rand(inh_inst_spike_prob.shape[0], inh_inst_spike_prob.shape[1]) < inh_inst_spike_prob
    
    # This accounts also for shared connections
    same_exc_inh_spikes_bin_prob = args.same_exc_inh_spikes_bin_prob
    if args.exc_weights_ratio_range[0] < args.exc_weights_ratio_range[1] or args.inh_weights_ratio_range[0] < args.inh_weights_ratio_range[1]:
        same_exc_inh_spikes_bin_prob *= args.same_exc_inh_spikes_bin_prob_weighted_multiply

    if exc_spikes_bin.shape == inh_spikes_bin.shape and np.random.rand() < same_exc_inh_spikes_bin_prob:
        logger.info("on same_exc_inh_spikes_bin mode")
        inh_spikes_bin = exc_spikes_bin

    return exc_spikes_bin, inh_spikes_bin

class MoreThanOneEventPerMsException(Exception):
    pass

class ForceNumberOfSegmentsIsNotAMultipleOfNumberOfSegments(Exception):
    pass

def generate_input_spike_trains_for_simulation_new(args, sim_duration_ms, count_exc_netcons, count_inh_netcons):
    auxiliary_information = {}

    inst_rate_exc, inst_rate_inh, original_spike_rates_information = generate_input_spike_rates_for_simulation(args, sim_duration_ms, count_exc_netcons, count_inh_netcons)
    
    auxiliary_information['original_spike_rates_information'] = original_spike_rates_information

    special_interval_added_edge_indicator = np.zeros(sim_duration_ms)
    for k in range(args.count_special_intervals):
        special_interval_high_dur_ms = min(args.special_interval_high_dur_ms, sim_duration_ms // 2)
        special_interval_low_dur_ms = min(args.special_interval_low_dur_ms, sim_duration_ms // 4)
        special_interval_start_ind = np.random.randint(sim_duration_ms - special_interval_high_dur_ms - args.special_interval_offset_ms)
        special_interval_duration_ms = np.random.randint(special_interval_low_dur_ms, special_interval_high_dur_ms)
        special_interval_final_ind = special_interval_start_ind + special_interval_duration_ms

        curr_special_interval_inst_rate_exc, curr_special_interval_inst_rate_inh, special_aux_info = generate_input_spike_rates_for_simulation(args, special_interval_duration_ms, count_exc_netcons, count_inh_netcons)

        auxiliary_information[f'special_interval_start_ind_{k}'] = special_interval_start_ind
        auxiliary_information[f'special_interval_duration_ms_{k}'] = special_interval_duration_ms
        auxiliary_information[f'spike_rates_information_special_interval_{k}'] = special_aux_info

        inst_rate_exc[:,special_interval_start_ind:special_interval_final_ind] = curr_special_interval_inst_rate_exc
        inst_rate_inh[:,special_interval_start_ind:special_interval_final_ind] = curr_special_interval_inst_rate_inh
        special_interval_added_edge_indicator[special_interval_start_ind] = 1
        special_interval_added_edge_indicator[special_interval_final_ind] = 1

    smoothing_window = signal.gaussian(1.0 + args.special_interval_transition_dur_ms_gaussian_mult * args.special_interval_transition_dur_ms, std=args.special_interval_transition_dur_ms)
    special_interval_added_edge_indicator = signal.convolve(special_interval_added_edge_indicator,  smoothing_window, mode='same') > args.special_interval_transition_threshold

    smoothing_window /= smoothing_window.sum()
    inst_rate_exc_smoothed = signal.convolve(inst_rate_exc, smoothing_window[np.newaxis,:], mode='same')
    inst_rate_inh_smoothed = signal.convolve(inst_rate_inh, smoothing_window[np.newaxis,:], mode='same')

    # build the final rates matrices
    inst_rate_exc_final = inst_rate_exc.copy()
    inst_rate_inh_final = inst_rate_inh.copy()

    inst_rate_exc_final[:,special_interval_added_edge_indicator] = inst_rate_exc_smoothed[:,special_interval_added_edge_indicator]
    inst_rate_inh_final[:,special_interval_added_edge_indicator] = inst_rate_inh_smoothed[:,special_interval_added_edge_indicator]

    # correct any minor mistakes
    inst_rate_exc_final[inst_rate_exc_final <= 0] = 0
    inst_rate_inh_final[inst_rate_inh_final <= 0] = 0

    exc_spikes_bin, inh_spikes_bin = sample_spikes_from_rates(args, inst_rate_exc_final, inst_rate_inh_final)

    for spikes_bin in exc_spikes_bin:
        spike_times = np.nonzero(spikes_bin)[0]
        if len(list(spike_times)) != len(set(spike_times)):
            raise MoreThanOneEventPerMsException("there is more than one event per ms!")

    for spikes_bin in inh_spikes_bin:
        spike_times = np.nonzero(spikes_bin)[0]
        if len(list(spike_times)) != len(set(spike_times)):
            raise MoreThanOneEventPerMsException("there is more than one event per ms!")

    return exc_spikes_bin, inh_spikes_bin, auxiliary_information

def generate_spike_times_and_weights_for_kernel_based_weights(args, syns, simulation_duration_in_ms):
    exc_netcons = syns.exc_netcons
    inh_netcons = syns.inh_netcons

    if args.zero_padding_for_initialization:
        raise NotImplementedError("zero_padding_for_initialization is not implemented for kernel based weights")

    auxiliary_information = {}

    multiple_connections = np.random.rand() < args.multiple_connections_prob
    multiply_count_initial_synapses_per_super_synapse = np.random.rand() < args.multiply_count_initial_synapses_per_super_synapse_prob
    
    auxiliary_information['multiple_connections'] = multiple_connections
    auxiliary_information['multiply_count_initial_synapses_per_super_synapse'] = multiply_count_initial_synapses_per_super_synapse
    auxiliary_information['seg_lens'] = syns.seg_lens

    if multiply_count_initial_synapses_per_super_synapse:
        count_exc_initial_synapses_per_super_synapse = np.ceil([seg_len * np.random.uniform(args.count_exc_initial_synapses_per_super_synapse_mult_factor_range[0], args.count_exc_initial_synapses_per_super_synapse_mult_factor_range[1]) for seg_len in syns.seg_lens]).astype(int)
        count_inh_initial_synapses_per_super_synapse = np.ceil([seg_len * np.random.uniform(args.count_inh_initial_synapses_per_super_synapse_mult_factor_range[0], args.count_inh_initial_synapses_per_super_synapse_mult_factor_range[1]) for seg_len in syns.seg_lens]).astype(int)
        logger.info("on multiply_count_initial_synapses_per_super_synapse mode")
    else:
        count_exc_initial_synapses_per_super_synapse = np.ceil(syns.seg_lens).astype(int)
        count_inh_initial_synapses_per_super_synapse = np.ceil(syns.seg_lens).astype(int)

    if np.random.rand() < args.same_exc_inh_count_initial_synapses_per_super_synapse_prob:
        count_inh_initial_synapses_per_super_synapse = count_exc_initial_synapses_per_super_synapse

        logger.info("on same_exc_inh_count_initial_synapses_per_super_synapse mode")

    if args.force_count_initial_synapses_per_super_synapse is not None:
        count_exc_initial_synapses_per_super_synapse = np.array([args.force_count_initial_synapses_per_super_synapse for _ in count_exc_initial_synapses_per_super_synapse])
        count_inh_initial_synapses_per_super_synapse = np.array([args.force_count_initial_synapses_per_super_synapse for _ in count_inh_initial_synapses_per_super_synapse])

    if args.force_count_initial_synapses_per_tree is not None:
        average_number_of_initial_synapses_per_super_synapse = args.force_count_initial_synapses_per_tree // len(count_exc_initial_synapses_per_super_synapse)
        auxiliary_information['average_number_of_initial_synapses_per_super_synapse'] = average_number_of_initial_synapses_per_super_synapse
        count_exc_initial_synapses_per_super_synapse = np.array([average_number_of_initial_synapses_per_super_synapse for _ in count_exc_initial_synapses_per_super_synapse])
        for _ in range(args.force_count_initial_synapses_per_tree % len(count_exc_initial_synapses_per_super_synapse)):
            count_exc_initial_synapses_per_super_synapse[np.random.randint(len(count_exc_initial_synapses_per_super_synapse))] += 1
        
        count_inh_initial_synapses_per_super_synapse = count_exc_initial_synapses_per_super_synapse

    auxiliary_information['count_exc_initial_synapses_per_super_synapse'] = count_exc_initial_synapses_per_super_synapse
    auxiliary_information['count_inh_initial_synapses_per_super_synapse'] = count_inh_initial_synapses_per_super_synapse

    original_count_exc_initial_neurons = count_exc_initial_neurons = np.sum(count_exc_initial_synapses_per_super_synapse)
    original_count_inh_initial_neurons = count_inh_initial_neurons = np.sum(count_inh_initial_synapses_per_super_synapse)

    if multiple_connections:
        average_exc_multiple_connections = min(args.exc_multiple_connections_upperbound, max(args.average_exc_multiple_connections_avg_std_min[2], abs(np.random.normal(args.average_exc_multiple_connections_avg_std_min[0], args.average_exc_multiple_connections_avg_std_min[1]))))
        if np.random.rand() < args.same_exc_inh_average_multiple_connections_prob:
            logger.info("on same_exc_inh_average_multiple_connections mode")
            average_inh_multiple_connections = average_exc_multiple_connections
        else:
            average_inh_multiple_connections = min(args.inh_multiple_connections_upperbound, max(args.average_inh_multiple_connections_avg_std_min[2], abs(np.random.normal(args.average_inh_multiple_connections_avg_std_min[0], args.average_inh_multiple_connections_avg_std_min[1]))))
    
        count_exc_initial_neurons = max(int(count_exc_initial_neurons / average_exc_multiple_connections), 1)
        count_inh_initial_neurons = max(int(count_inh_initial_neurons / average_inh_multiple_connections), 1)
        auxiliary_information['average_exc_multiple_connections'] = average_exc_multiple_connections
        auxiliary_information['average_inh_multiple_connections'] = average_inh_multiple_connections

        logger.info(f"on multiple_connections mode, average_exc_multiple_connections is {average_exc_multiple_connections}, average_inh_multiple_connections is {average_inh_multiple_connections}")

    if args.force_multiply_count_spikes_per_synapse_per_100ms_range_by_average_segment_length:
        average_segment_length = np.mean(syns.seg_lens)
        args.effective_count_exc_spikes_per_synapse_per_100ms_range = [args.count_exc_spikes_per_synapse_per_100ms_range[0] * average_segment_length, args.count_exc_spikes_per_synapse_per_100ms_range[1] * average_segment_length]
        args.effective_count_inh_spikes_per_synapse_per_100ms_range = [args.count_inh_spikes_per_synapse_per_100ms_range[0] * average_segment_length, args.count_inh_spikes_per_synapse_per_100ms_range[1] * average_segment_length]
        args.effective_adaptive_inh_additive_factor_per_synapse_per_100ms_range = [args.adaptive_inh_additive_factor_per_synapse_per_100ms_range[0] * average_segment_length, args.adaptive_inh_additive_factor_per_synapse_per_100ms_range[1] * average_segment_length]
    else:
        args.effective_count_exc_spikes_per_synapse_per_100ms_range = args.count_exc_spikes_per_synapse_per_100ms_range
        args.effective_count_inh_spikes_per_synapse_per_100ms_range = args.count_inh_spikes_per_synapse_per_100ms_range
        args.effective_adaptive_inh_additive_factor_per_synapse_per_100ms_range = args.adaptive_inh_additive_factor_per_synapse_per_100ms_range

    exc_initial_neurons_spikes_bin, inh_initial_neurons_spikes_bin, initial_neurons_aux_info = generate_input_spike_trains_for_simulation_new(args, simulation_duration_in_ms, count_exc_initial_neurons, count_inh_initial_neurons)
    
    auxiliary_information['initial_neurons_spike_trains_information'] = initial_neurons_aux_info
    auxiliary_information['exc_initial_neurons_spikes_bin'] = exc_initial_neurons_spikes_bin
    auxiliary_information['inh_initial_neurons_spikes_bin'] = inh_initial_neurons_spikes_bin

    exc_initial_neurons_spikes_bin = np.array(exc_initial_neurons_spikes_bin)
    inh_initial_neurons_spikes_bin = np.array(inh_initial_neurons_spikes_bin)

    logger.info(f"exc_initial_neurons_spikes_bin.shape is {exc_initial_neurons_spikes_bin.shape}")
    logger.info(f"inh_initial_neurons_spikes_bin.shape is {inh_initial_neurons_spikes_bin.shape}")

    # Done generating input spikes, now to weights and weighted spikes

    # This is the default, but gets turned off when simple_stimulation is on (or if explicitly turned off)
    if args.generate_weights_using_constrained_linear:
        exc_input_dim = count_exc_initial_neurons
        inh_input_dim = count_inh_initial_neurons
        ds_input_dim = (exc_input_dim, inh_input_dim)

        exc_output_dim = len(exc_netcons)
        inh_output_dim = len(inh_netcons)
        model_input_dim = (exc_output_dim, inh_output_dim)

        wiring_bias = False # not relevant, but required, so it'll work
        functional_only_wiring = False
        positive_wiring = True

        exc_weight_init_mean = (args.exc_weights_ratio_range[0] + args.exc_weights_ratio_range[1]) / 2
        exc_weight_init_bound = (args.exc_weights_ratio_range[1] - args.exc_weights_ratio_range[0]) / 2
        inh_weight_init_mean = (args.inh_weights_ratio_range[0] + args.inh_weights_ratio_range[1]) / 2
        inh_weight_init_bound = (args.inh_weights_ratio_range[1] - args.inh_weights_ratio_range[0]) / 2
        wiring_weight_init_mean = (exc_weight_init_mean, inh_weight_init_mean)
        wiring_weight_init_bound = (exc_weight_init_bound, inh_weight_init_bound)

        average_exc_axons_per_segment = original_count_exc_initial_neurons / len(exc_netcons)
        average_inh_axons_per_segment = original_count_inh_initial_neurons /len(inh_netcons)

        average_exc_row_density = average_exc_axons_per_segment / count_exc_initial_neurons
        average_inh_row_density = average_inh_axons_per_segment / count_inh_initial_neurons

        max_exc_synapses_per_super_synapse = np.max(count_exc_initial_synapses_per_super_synapse)
        max_inh_synapses_per_super_synapse = np.max(count_inh_initial_synapses_per_super_synapse)

        exc_weight_init_sparsity = average_exc_row_density
        inh_weight_init_sparsity = average_inh_row_density
        wiring_weight_init_sparsity = (exc_weight_init_sparsity, inh_weight_init_sparsity)

        wiring_zero_smaller_than = 0.0

        exc_keep_max_k_from_input = args.exc_multiple_connections_upperbound
        inh_keep_max_k_from_input = args.inh_multiple_connections_upperbound
        wiring_keep_max_k_from_input = (exc_keep_max_k_from_input, inh_keep_max_k_from_input)

        logger.info(f"exc_keep_max_k_from_input is {exc_keep_max_k_from_input}, inh_keep_max_k_from_input is {inh_keep_max_k_from_input}")

        exc_keep_max_k_to_output = max_exc_synapses_per_super_synapse
        inh_keep_max_k_to_output = max_inh_synapses_per_super_synapse
        wiring_keep_max_k_to_output = (exc_keep_max_k_to_output, inh_keep_max_k_to_output)

        logger.info(f"exc_keep_max_k_to_output is {exc_keep_max_k_to_output}, inh_keep_max_k_to_output is {inh_keep_max_k_to_output}")

        wiring_dales_law = True

        wiring_layer = ConstrainedLinear(ds_input_dim, model_input_dim, bias=wiring_bias,
                    diagonals_only=functional_only_wiring, positive_weight=positive_wiring,
                    weight_init_mean=wiring_weight_init_mean, weight_init_bound=wiring_weight_init_bound,
                    weight_init_sparsity=wiring_weight_init_sparsity, zero_smaller_than=wiring_zero_smaller_than,
                        keep_max_k_from_input=wiring_keep_max_k_from_input, keep_max_k_to_output=wiring_keep_max_k_to_output,
                        dales_law=wiring_dales_law)

        wiring_matrix = wiring_layer.get_weights().detach().numpy()

        exc_wiring_matrix = wiring_matrix[:exc_output_dim, :exc_input_dim]
        inh_wiring_matrix = wiring_matrix[exc_output_dim:, exc_input_dim:]

        logger.info(f"wiring_matrix.shape is {wiring_matrix.shape}, exc_wiring_matrix.shape is {exc_wiring_matrix.shape}, inh_wiring_matrix.shape is {inh_wiring_matrix.shape}")

        all_initial_neurons_spikes_bin = np.concatenate((exc_initial_neurons_spikes_bin, inh_initial_neurons_spikes_bin), axis=0)

        weighted_spikes = wiring_matrix @ all_initial_neurons_spikes_bin

        exc_initial_neurons_weights = None
        inh_initial_neurons_weights = None
        exc_initial_neuron_connection_counts = None
        inh_initial_neuron_connection_counts = None

        ret = generate_spike_times_and_weights_from_weighted_spikes(args, syns, simulation_duration_in_ms, weighted_spikes)
        
        exc_ncon_to_input_spike_times, inh_ncon_to_input_spike_times, exc_weighted_spikes, inh_weighted_spikes, count_exc_spikes, count_inh_spikes = ret

    else:
        if multiple_connections:
            exc_initial_neuron_connection_counts = np.zeros(count_exc_initial_neurons)
        else:
            exc_initial_neuron_connection_counts = None
        exc_super_synapse_kernels = []
        exc_weighted_spikes = np.zeros((len(exc_netcons), simulation_duration_in_ms))
        exc_ncon_to_input_spike_times = {}
        count_exc_spikes = 0
        count_weighted_exc_spikes = 0
        exc_initial_neurons_weights = []
        for exc_netcon_index, exc_netcon in enumerate(exc_netcons):
            if multiple_connections:
                kernel_density = (count_exc_initial_synapses_per_super_synapse[exc_netcon_index] + 0.0)  / count_exc_initial_neurons
                get_random_exc_weight_ratio = lambda s : np.random.uniform(args.exc_weights_ratio_range[0], args.exc_weights_ratio_range[1], s)
                exc_super_synapse_random_kernel = sparse.random(1, count_exc_initial_neurons, density=kernel_density, data_rvs=get_random_exc_weight_ratio).A
                initial_exc_super_synapse_random_kernel_usage = np.nonzero(exc_super_synapse_random_kernel)[1]

                for i in initial_exc_super_synapse_random_kernel_usage:
                    if exc_initial_neuron_connection_counts[i] > args.exc_multiple_connections_upperbound:
                        new_index = np.random.choice(np.intersect1d(np.where(exc_initial_neuron_connection_counts < args.exc_multiple_connections_upperbound), np.where(exc_super_synapse_random_kernel == 0)[1]))
                        exc_super_synapse_random_kernel[0, new_index] = exc_super_synapse_random_kernel[0, i]
                        exc_super_synapse_random_kernel[0, i] = 0
                        exc_initial_neuron_connection_counts[new_index] += 1        
                    else:
                        exc_initial_neuron_connection_counts[i] += 1
                
                exc_super_synapse_kernels.append(exc_super_synapse_random_kernel)
                exc_initial_neurons_weights += list(exc_super_synapse_random_kernel[exc_super_synapse_random_kernel!=0])
                weighted_spikes = np.dot(exc_super_synapse_random_kernel, exc_initial_neurons_spikes_bin).flatten()
                count_weighted_exc_spikes += np.sum(weighted_spikes)
            else:
                relevant_exc_initial_neurons_spikes_bin = exc_initial_neurons_spikes_bin[np.sum(count_exc_initial_synapses_per_super_synapse[:exc_netcon_index]):np.sum(count_exc_initial_synapses_per_super_synapse[:exc_netcon_index+1])]
                exc_super_synapse_random_kernel = np.random.uniform(low=args.exc_weights_ratio_range[0], high=args.exc_weights_ratio_range[1], size=(1, relevant_exc_initial_neurons_spikes_bin.shape[0]))
                exc_super_synapse_kernels.append(exc_super_synapse_random_kernel)
                exc_initial_neurons_weights += list(exc_super_synapse_random_kernel[exc_super_synapse_random_kernel!=0])
                weighted_spikes = np.dot(exc_super_synapse_random_kernel, relevant_exc_initial_neurons_spikes_bin).flatten()
                count_weighted_exc_spikes += np.sum(weighted_spikes)

            exc_weighted_spikes[exc_netcon_index, :] = weighted_spikes
            exc_ncon_to_input_spike_times[exc_netcon] = np.nonzero(weighted_spikes)[0]
            count_exc_spikes += len(exc_ncon_to_input_spike_times[exc_netcon])

        exc_wiring_matrix = np.zeros((exc_super_synapse_kernels[0].shape[1], len(exc_super_synapse_kernels)))

        same_exc_inh_kernels_possible = (count_exc_initial_neurons == count_inh_initial_neurons) and (args.inh_multiple_connections_upperbound == args.exc_multiple_connections_upperbound)

        same_exc_inh_all_kernels = False
        if same_exc_inh_kernels_possible and np.random.rand() < args.same_exc_inh_all_kernels_prob:
            same_exc_inh_all_kernels = True
            logger.info("on same_exc_inh_all_kernels mode")

        if multiple_connections:
            inh_initial_neuron_connection_counts = np.zeros(count_inh_initial_neurons)
        else:
            inh_initial_neuron_connection_counts = None
        inh_super_synapse_kernels = []
        inh_weighted_spikes = np.zeros((len(inh_netcons), simulation_duration_in_ms))
        inh_ncon_to_input_spike_times = {}
        count_inh_spikes = 0
        count_weighted_inh_spikes = 0
        inh_initial_neurons_weights = []
        for inh_netcon_index, inh_netcon in enumerate(inh_netcons):    
            if multiple_connections:
                kernel_density = (count_inh_initial_synapses_per_super_synapse[inh_netcon_index] + 0.0)  / count_inh_initial_neurons
                get_random_inh_weight_ratio = lambda s : np.random.uniform(args.inh_weights_ratio_range[0], args.inh_weights_ratio_range[1], s)

                same_exc_inh_kernels = False
                if same_exc_inh_kernels_possible and np.random.rand() < args.same_exc_inh_kernels_prob:
                    logger.info(f"on same_exc_inh_kernels mode for netcon_index {inh_netcon_index}")
                    same_exc_inh_kernels = True

                if same_exc_inh_all_kernels or same_exc_inh_kernels:
                    inh_super_synapse_random_kernel = exc_super_synapse_kernels[inh_netcon_index]
                else:
                    inh_super_synapse_random_kernel = sparse.random(1, count_inh_initial_neurons, density=kernel_density, data_rvs=get_random_inh_weight_ratio).A
                    initial_inh_super_synapse_random_kernel_usage = np.nonzero(inh_super_synapse_random_kernel)[1]

                    for i in initial_inh_super_synapse_random_kernel_usage:
                        if inh_initial_neuron_connection_counts[i] > args.inh_multiple_connections_upperbound:
                            new_index = np.random.choice(np.intersect1d(np.where(inh_initial_neuron_connection_counts < args.inh_multiple_connections_upperbound), np.where(inh_super_synapse_random_kernel == 0)[1]))
                            inh_super_synapse_random_kernel[0, new_index] = inh_super_synapse_random_kernel[0, i]
                            inh_super_synapse_random_kernel[0, i] = 0
                            inh_initial_neuron_connection_counts[new_index] += 1        
                        else:
                            inh_initial_neuron_connection_counts[i] += 1

                inh_super_synapse_kernels.append(inh_super_synapse_random_kernel)
                inh_initial_neurons_weights += list(inh_super_synapse_random_kernel[inh_super_synapse_random_kernel!=0])
                weighted_spikes = np.dot(inh_super_synapse_random_kernel, inh_initial_neurons_spikes_bin).flatten()
                count_weighted_inh_spikes += np.sum(weighted_spikes)
            else:
                relevant_inh_initial_neurons_spikes_bin = inh_initial_neurons_spikes_bin[np.sum(count_inh_initial_synapses_per_super_synapse[:inh_netcon_index]):np.sum(count_inh_initial_synapses_per_super_synapse[:inh_netcon_index+1])]
                inh_super_synapse_random_kernel = np.random.uniform(low=args.inh_weights_ratio_range[0], high=args.inh_weights_ratio_range[1], size=(1, relevant_inh_initial_neurons_spikes_bin.shape[0]))
                inh_super_synapse_kernels.append(inh_super_synapse_random_kernel)
                inh_initial_neurons_weights += list(inh_super_synapse_random_kernel[inh_super_synapse_random_kernel!=0])
                weighted_spikes = np.dot(inh_super_synapse_random_kernel, relevant_inh_initial_neurons_spikes_bin).flatten()
                count_weighted_inh_spikes += np.sum(weighted_spikes)

            inh_weighted_spikes[inh_netcon_index, :] = weighted_spikes
            inh_ncon_to_input_spike_times[inh_netcon] = np.nonzero(weighted_spikes)[0]
            count_inh_spikes += len(inh_ncon_to_input_spike_times[inh_netcon])

        inh_wiring_matrix = np.zeros((inh_super_synapse_kernels[0].shape[1], len(inh_super_synapse_kernels)))

    auxiliary_information = save_more_auxiliary_information(args, syns, simulation_duration_in_ms, exc_weighted_spikes, inh_weighted_spikes, \
        count_exc_spikes, count_inh_spikes, auxiliary_information, exc_initial_neurons_weights=exc_initial_neurons_weights, \
            inh_initial_neurons_weights=inh_initial_neurons_weights, \
                exc_initial_neuron_connection_counts=exc_initial_neuron_connection_counts, \
                    inh_initial_neuron_connection_counts=inh_initial_neuron_connection_counts, \
                        exc_wiring_matrix=exc_wiring_matrix, inh_wiring_matrix=inh_wiring_matrix)

    return simulation_duration_in_ms, exc_ncon_to_input_spike_times, inh_ncon_to_input_spike_times, exc_weighted_spikes, inh_weighted_spikes, auxiliary_information

def generate_spike_times_and_weights_from_input_file(args, syns):
    exc_netcons = syns.exc_netcons
    inh_netcons = syns.inh_netcons

    auxiliary_information = {}
    auxiliary_information["input_file"] = args.input_file

    weighted_spikes = sparse.load_npz(args.input_file).A

    simulation_duration_in_ms = 0
    if args.add_explicit_padding_for_initialization:
        simulation_duration_in_ms = args.simulation_initialization_duration_in_ms
    simulation_duration_in_ms += weighted_spikes.shape[1]

    if args.add_explicit_padding_for_initialization:
        if args.zero_padding_for_initialization:
            zero_exc_weighted_spikes = np.zeros((len(exc_netcons), args.simulation_initialization_duration_in_ms))
            zero_inh_weighted_spikes = np.zeros((len(inh_netcons), args.simulation_initialization_duration_in_ms))

            padding_exc_weighted_spikes = zero_exc_weighted_spikes
            padding_inh_weighted_spikes = zero_inh_weighted_spikes
        else:
            _, _, _, noise_padding_exc_weighted_spikes, noise_padding_inh_weighted_spikes, noise_padding_auxiliary_information = generate_spike_times_and_weights_for_kernel_based_weights(args, syns, args.simulation_initialization_duration_in_ms)
            auxiliary_information['noise_padding_auxiliary_information'] = noise_padding_auxiliary_information

            padding_exc_weighted_spikes = noise_padding_exc_weighted_spikes
            padding_inh_weighted_spikes = noise_padding_inh_weighted_spikes
    else:
        padding_exc_weighted_spikes = None
        padding_inh_weighted_spikes = None

    ret = generate_spike_times_and_weights_from_weighted_spikes(args, syns, simulation_duration_in_ms,
     weighted_spikes, padding_exc_weighted_spikes=padding_exc_weighted_spikes, padding_inh_weighted_spikes=padding_inh_weighted_spikes)
     
    exc_ncon_to_input_spike_times, inh_ncon_to_input_spike_times, exc_weighted_spikes, inh_weighted_spikes, count_exc_spikes, count_inh_spikes = ret

    auxiliary_information = save_more_auxiliary_information(args, syns, simulation_duration_in_ms, exc_weighted_spikes, inh_weighted_spikes, \
        count_exc_spikes, count_inh_spikes, auxiliary_information)

    return simulation_duration_in_ms, exc_ncon_to_input_spike_times, inh_ncon_to_input_spike_times, exc_weighted_spikes, inh_weighted_spikes, auxiliary_information

def generate_spike_times_and_weights_from_weighted_spikes(args, syns, simulation_duration_in_ms, weighted_spikes,
 padding_exc_weighted_spikes=None, padding_inh_weighted_spikes=None):
    exc_netcons = syns.exc_netcons
    inh_netcons = syns.inh_netcons

    if weighted_spikes.min() < 0:
        raise ValueError("weighted_spikes contains negative values")        

    exc_weighted_spikes = np.zeros((len(exc_netcons), simulation_duration_in_ms))
    exc_ncon_to_input_spike_times = {}
    count_exc_spikes = 0
    for exc_netcon_index, exc_netcon in enumerate(exc_netcons):
        if padding_exc_weighted_spikes is not None:
            cur_exc_weighted_spikes = np.concatenate((padding_exc_weighted_spikes[exc_netcon_index,:], weighted_spikes[exc_netcon_index,:]))
        else:
            cur_exc_weighted_spikes = weighted_spikes[exc_netcon_index,:]
        exc_weighted_spikes[exc_netcon_index, :] = cur_exc_weighted_spikes
        exc_ncon_to_input_spike_times[exc_netcon] = np.nonzero(cur_exc_weighted_spikes)[0]
        count_exc_spikes += len(exc_ncon_to_input_spike_times[exc_netcon])

    inh_weighted_spikes = np.zeros((len(inh_netcons), simulation_duration_in_ms))
    inh_ncon_to_input_spike_times = {}
    count_inh_spikes = 0
    for inh_netcon_index, inh_netcon in enumerate(inh_netcons):
        if padding_inh_weighted_spikes is not None:
            cur_inh_weighted_spikes = np.concatenate((padding_inh_weighted_spikes[inh_netcon_index,:], weighted_spikes[len(exc_netcons) + inh_netcon_index,:]))
        else:
            cur_inh_weighted_spikes = weighted_spikes[len(exc_netcons) + inh_netcon_index,:]
        inh_weighted_spikes[inh_netcon_index, :] = cur_inh_weighted_spikes
        inh_ncon_to_input_spike_times[inh_netcon] = np.nonzero(cur_inh_weighted_spikes)[0]
        count_inh_spikes += len(inh_ncon_to_input_spike_times[inh_netcon])

    return exc_ncon_to_input_spike_times, inh_ncon_to_input_spike_times, exc_weighted_spikes, inh_weighted_spikes, count_exc_spikes, count_inh_spikes

def save_more_auxiliary_information(args, syns, simulation_duration_in_ms, exc_weighted_spikes, inh_weighted_spikes, \
    count_exc_spikes, count_inh_spikes, auxiliary_information, exc_initial_neurons_weights=None, inh_initial_neurons_weights=None, \
        exc_initial_neuron_connection_counts=None, inh_initial_neuron_connection_counts=None, exc_wiring_matrix=None, inh_wiring_matrix=None):

    exc_netcons = syns.exc_netcons
    inh_netcons = syns.inh_netcons

    # exc
    
    average_exc_spikes_per_second = count_exc_spikes / (simulation_duration_in_ms / 1000)
    count_exc_spikes_per_super_synapse = count_exc_spikes / (len(exc_netcons) + 0.0)
    average_exc_spikes_per_super_synapse_per_second = count_exc_spikes_per_super_synapse / (simulation_duration_in_ms/1000.0)

    auxiliary_information['count_exc_spikes'] = count_exc_spikes
    auxiliary_information['average_exc_spikes_per_second'] = average_exc_spikes_per_second
    auxiliary_information['count_exc_spikes_per_super_synapse'] = count_exc_spikes_per_super_synapse
    auxiliary_information['average_exc_spikes_per_super_synapse_per_second'] = average_exc_spikes_per_super_synapse_per_second

    logger.info(f'average of exc spikes per second is {average_exc_spikes_per_second:.3f}, which is {average_exc_spikes_per_super_synapse_per_second:.3f} average exc spikes per exc netcon per second')

    count_weighted_exc_spikes = np.sum(exc_weighted_spikes)

    average_weighted_exc_spikes_per_second = count_weighted_exc_spikes / (simulation_duration_in_ms / 1000)
    count_weighted_exc_spikes_per_super_synapse = count_weighted_exc_spikes / (len(exc_netcons) + 0.0)
    average_weighted_exc_spikes_per_super_synapse_per_second = count_weighted_exc_spikes_per_super_synapse / (simulation_duration_in_ms/1000.0)
    
    auxiliary_information['count_weighted_exc_spikes'] = count_weighted_exc_spikes
    auxiliary_information['average_weighted_exc_spikes_per_second'] = average_weighted_exc_spikes_per_second
    auxiliary_information['count_weighted_exc_spikes_per_super_synapse'] = count_weighted_exc_spikes_per_super_synapse
    auxiliary_information['average_weighted_exc_spikes_per_super_synapse_per_second'] = average_weighted_exc_spikes_per_super_synapse_per_second

    logger.info(f'average of weighted exc spikes per second is {average_weighted_exc_spikes_per_second:.3f}, which is {average_weighted_exc_spikes_per_super_synapse_per_second:.3f} average weighted exc spikes per exc netcon per second')

    if exc_initial_neurons_weights is None:
        exc_initial_neurons_weights = [0.0]
    average_exc_initial_neuron_weight = np.mean(exc_initial_neurons_weights)
    auxiliary_information['exc_initial_neurons_weights'] = exc_initial_neurons_weights
    auxiliary_information['average_exc_initial_neuron_weight'] = average_exc_initial_neuron_weight

    logger.info(f'average exc initial neuron weight is {average_exc_initial_neuron_weight:.3f}')

    if exc_initial_neuron_connection_counts is not None:
        logger.info(f'min, max, avg, std, med exc initial neuron connection count are {np.min(exc_initial_neuron_connection_counts):.3f}, {np.max(exc_initial_neuron_connection_counts):.3f}, {np.mean(exc_initial_neuron_connection_counts):.3f}, {np.std(exc_initial_neuron_connection_counts):.3f}, {np.median(exc_initial_neuron_connection_counts):.3f}')
        auxiliary_information['exc_initial_neuron_connection_counts'] = exc_initial_neuron_connection_counts

    auxiliary_information['exc_wiring_matrix'] = exc_wiring_matrix

    # inh
   
    average_inh_spikes_per_second = count_inh_spikes / (simulation_duration_in_ms / 1000)
    count_inh_spikes_per_super_synapse = count_inh_spikes / (len(inh_netcons) + 0.0)
    average_inh_spikes_per_super_synapse_per_second = count_inh_spikes_per_super_synapse / (simulation_duration_in_ms/1000.0)

    auxiliary_information['count_inh_spikes'] = count_inh_spikes
    auxiliary_information['average_inh_spikes_per_second'] = average_inh_spikes_per_second
    auxiliary_information['count_inh_spikes_per_super_synapse'] = count_inh_spikes_per_super_synapse
    auxiliary_information['average_inh_spikes_per_super_synapse_per_second'] = average_inh_spikes_per_super_synapse_per_second

    logger.info(f'average number of inh spikes per second is {average_inh_spikes_per_second:.3f}, which is {average_inh_spikes_per_super_synapse_per_second:.3f} average inh spikes per inh netcon per second')

    count_weighted_inh_spikes = np.sum(inh_weighted_spikes)

    average_weighted_inh_spikes_per_second = count_weighted_inh_spikes / (simulation_duration_in_ms / 1000)
    count_weighted_inh_spikes_per_super_synapse = count_weighted_inh_spikes / (len(inh_netcons) + 0.0)
    average_weighted_inh_spikes_per_super_synapse_per_second = count_weighted_inh_spikes_per_super_synapse / (simulation_duration_in_ms/1000.0)
    
    auxiliary_information['count_weighted_inh_spikes'] = count_weighted_inh_spikes
    auxiliary_information['average_weighted_inh_spikes_per_second'] = average_weighted_inh_spikes_per_second
    auxiliary_information['count_weighted_inh_spikes_per_super_synapse'] = count_weighted_inh_spikes_per_super_synapse
    auxiliary_information['average_weighted_inh_spikes_per_super_synapse_per_second'] = average_weighted_inh_spikes_per_super_synapse_per_second

    logger.info(f'average number of weighted inh spikes per second is {average_weighted_inh_spikes_per_second:.3f}, which is {average_weighted_inh_spikes_per_super_synapse_per_second:.3f} average weighted inh spikes per inh netcon per second')

    if inh_initial_neurons_weights is None:
        inh_initial_neurons_weights = [0.0]
    average_inh_initial_neuron_weight = np.mean(inh_initial_neurons_weights)
    auxiliary_information['inh_initial_neurons_weights'] = inh_initial_neurons_weights
    auxiliary_information['average_inh_initial_neuron_weight'] = average_inh_initial_neuron_weight

    logger.info(f'average inh initial neuron weight is {average_inh_initial_neuron_weight:.3f}')

    if inh_initial_neuron_connection_counts is not None:
        logger.info(f'min, max, avg, std, med inh initial neuron connection count are {np.min(inh_initial_neuron_connection_counts):.3f}, {np.max(inh_initial_neuron_connection_counts):.3f}, {np.mean(inh_initial_neuron_connection_counts):.3f}, {np.std(inh_initial_neuron_connection_counts):.3f}, {np.median(inh_initial_neuron_connection_counts):.3f}')
        auxiliary_information['inh_initial_neuron_connection_counts'] = inh_initial_neuron_connection_counts
    
    auxiliary_information['inh_wiring_matrix'] = inh_wiring_matrix

    return auxiliary_information

def generate_spike_times_and_weights(args, syns):
    if args.input_file is not None:
        return generate_spike_times_and_weights_from_input_file(args, syns)

    simulation_duration_in_seconds = args.simulation_duration_in_seconds

    simulation_duration_in_ms = 0
    if args.add_explicit_padding_for_initialization:
        simulation_duration_in_ms = args.simulation_initialization_duration_in_ms
    simulation_duration_in_ms += simulation_duration_in_seconds * 1000

    return generate_spike_times_and_weights_for_kernel_based_weights(args, syns, simulation_duration_in_ms)

def create_neuron_model(args):
    logger.info("About to import neuron module...")
    tm = importlib.import_module(f'{args.neuron_model_folder.replace("/",".")}.get_standard_model')
    logger.info("neuron module imported fine.")

    logger.info("About to create cell...")
    if args.max_segment_length is not None:
        cell, syns = tm.create_cell(max_segment_length=args.max_segment_length)
    else:
        cell, syns = tm.create_cell()
    logger.info("cell created fine.")

    if args.count_segments_to_stimulate is not None:
        syns = syns[:args.count_segments_to_stimulate]
        logger.info(f"Chosen {args.count_segments_to_stimulate} first segments to stimulate.")

    if args.force_number_of_segments is not None:
        logger.info(f"Currently have {len(syns)} segments and force_number_of_segments is {args.force_number_of_segments}.")
        if args.force_number_of_segments % len(syns) != 0:
            raise ForceNumberOfSegmentsIsNotAMultipleOfNumberOfSegments(f"force_number_of_segments {args.force_number_of_segments} is not a multiple of number of segments {len(syns)}.")

        multiple = int(args.force_number_of_segments / len(syns))
        logger.info(f"Multiple is {multiple}.")
        if multiple == 1:
            logger.info(f"No need to multiple segments.")
        else:
            original_number_of_segments = len(syns)
            new_values = []
            for ind in range(original_number_of_segments):
                orig_row = syns.iloc[ind]
                line = pd.DataFrame(orig_row).T
                new_values.append(line)
                for i in range(multiple - 1):
                    r = orig_row.copy()
                    r.exc_netcons = h.NetCon(None, r.exc_synapses)
                    r.inh_netcons = h.NetCon(None, r.inh_synapses)
                    line = pd.DataFrame(r).T
                    new_values.append(line)
            syns = pd.concat(new_values).reset_index(drop=True)
        logger.info(f"Now have {len(syns)} segments.")

    np_segment_lengths = np.array(syns.seg_lens)
    logger.info(f'min, max, avg, std, med segment length are {np.min(np_segment_lengths):.3f}, {np.max(np_segment_lengths):.3f}, {np.mean(np_segment_lengths):.3f}, {np.std(np_segment_lengths):.3f}, {np.median(np_segment_lengths):.3f}')

    if args.save_plots:
        plt.hist(np_segment_lengths, bins=10)
        plt.savefig(f"{args.simulation_folder}/segment_lengths.png")
        plt.close('all')

        plotter = NeuronPlotter(cell, list(syns['segments']),
                                cmap=plt.get_cmap('hot', MAX_CM - MIN_CM)(np.arange(0, MAX_CM - MIN_CM)))
        fig = plt.figure(figsize=(6, 10))
        ax = fig.subplots(1,1)
        ax.set_title(os.path.basename(args.neuron_model_folder))
        plotter.plot_shape(ax=ax)
        ax.set_axis_off()
        plt.savefig(f"{args.simulation_folder}/morphology.png")
        plt.close('all')

    return cell, syns


input_exc_sptimes = {}
input_inh_sptimes = {}

def run_neuron_model(args, cell, syns, simulation_duration_in_ms, exc_ncon_to_input_spike_times, inh_ncon_to_input_spike_times, exc_weighted_spikes, inh_weighted_spikes, auxiliary_information):
    # how to implement time dependent weights:
    # 1) create a new netcon for each event, and set the weight to the weight of the netcon at the time of the event + saving some netcons with same weight
    # 2) an alternative option that goes through python on each 1ms is using StateTransitionEvent, but do we want to go through python on each 1ms?
    # 3) reimplement NetCon to support a time dependent weight, but do we want to recompile NEURON? (TODO)
    # I chose option 1, but it might be heavy on memory, so I'm saving some netcons with roughly the same weight (rounded to some precision)

    total_number_of_netcons_after_saving = 0
    total_number_of_netcons = 0

    alt_exc_ncon_to_input_spike_times = {}
    for j, exc_ncon_and_spike_times in enumerate(exc_ncon_to_input_spike_times.items()):
        exc_netcon = exc_ncon_and_spike_times[0]
        spike_times = exc_ncon_and_spike_times[1]
        weight_to_alt_ncon = {}
        used_weights = []
        orig_exc_netcon_weight = exc_netcon.weight[0]
        orig_exc_netcon_used = False
        for sptime in spike_times:
            used_weights.append(exc_weighted_spikes[j][sptime])
            rounded_weight = round(exc_weighted_spikes[j][sptime], args.weight_rounding_precision)
            if args.use_rounded_weight:
                exc_weighted_spikes[j][sptime] = rounded_weight
            if rounded_weight in weight_to_alt_ncon:
                # reuse existing netcon, with a specific weight
                new_netcon = weight_to_alt_ncon[rounded_weight]
                alt_exc_ncon_to_input_spike_times[new_netcon] = (exc_netcon, np.concatenate((alt_exc_ncon_to_input_spike_times[new_netcon][1], np.array([sptime]))))
            else:
                if not orig_exc_netcon_used:
                    new_netcon = exc_netcon
                    orig_exc_netcon_used = True
                else:
                    new_netcon = h.NetCon(None, syns.exc_synapses[j])
                # setting the weight of the new netcon
                new_netcon.weight[0] = orig_exc_netcon_weight * (rounded_weight if args.use_rounded_weight else exc_weighted_spikes[j][sptime])
                weight_to_alt_ncon[rounded_weight] = new_netcon
                alt_exc_ncon_to_input_spike_times[new_netcon] = (exc_netcon, np.array([sptime]))

        total_number_of_netcons_after_saving += len(weight_to_alt_ncon.keys())
        total_number_of_netcons += len(used_weights)

    alt_inh_ncon_to_input_spike_times = {}
    for j, inh_ncon_and_spike_times in enumerate(inh_ncon_to_input_spike_times.items()):
        inh_netcon = inh_ncon_and_spike_times[0]
        spike_times = inh_ncon_and_spike_times[1]
        weight_to_alt_ncon = {}
        used_weights = []
        orig_inh_netcon_weight = inh_netcon.weight[0]
        orig_inh_netcon_used = False
        for sptime in spike_times:
            used_weights.append(inh_weighted_spikes[j][sptime])
            rounded_weight = round(inh_weighted_spikes[j][sptime], args.weight_rounding_precision)
            if args.use_rounded_weight:
                inh_weighted_spikes[j][sptime] = rounded_weight
            if rounded_weight in weight_to_alt_ncon:
                # reuse existing netcon, with a specific weight
                new_netcon = weight_to_alt_ncon[rounded_weight]
                alt_inh_ncon_to_input_spike_times[new_netcon] = (inh_netcon, np.concatenate((alt_inh_ncon_to_input_spike_times[new_netcon][1], np.array([sptime]))))
            else:
                if not orig_inh_netcon_used:
                    new_netcon = inh_netcon
                    orig_inh_netcon_used = True
                else:
                    new_netcon = h.NetCon(None, syns.inh_synapses[j])
                # setting the weight of the new netcon
                new_netcon.weight[0] = orig_inh_netcon_weight * (rounded_weight if args.use_rounded_weight else inh_weighted_spikes[j][sptime])
                weight_to_alt_ncon[rounded_weight] = new_netcon
                alt_inh_ncon_to_input_spike_times[new_netcon] = (inh_netcon, np.array([sptime]))

        total_number_of_netcons_after_saving += len(weight_to_alt_ncon.keys())
        total_number_of_netcons += len(used_weights)

    logger.info(f"There are {total_number_of_netcons_after_saving} netcons after saving {total_number_of_netcons-total_number_of_netcons_after_saving} out of {total_number_of_netcons}, using {args.weight_rounding_precision} precision")

    global input_exc_sptimes, input_inh_sptimes
    input_exc_sptimes = {}
    input_inh_sptimes = {}

    def apply_input_spike_times():
        logger.info("About to apply input spike times...")
        global input_exc_sptimes, input_inh_sptimes
        count_exc_events = 0
        count_inh_events = 0

        for alt_exc_netcon, exc_ncon_and_spike_times in alt_exc_ncon_to_input_spike_times.items():
            exc_netcon = exc_ncon_and_spike_times[0]
            spike_times = exc_ncon_and_spike_times[1]
            for sptime in spike_times:
                alt_exc_netcon.event(sptime)
                count_exc_events += 1
            input_exc_sptimes[exc_netcon] = spike_times

        for alt_inh_netcon, inh_ncon_and_spike_times in alt_inh_ncon_to_input_spike_times.items():
            inh_netcon = inh_ncon_and_spike_times[0]
            spike_times = inh_ncon_and_spike_times[1]
            for sptime in spike_times:
                alt_inh_netcon.event(sptime)
                count_inh_events += 1
            input_inh_sptimes[inh_netcon] = spike_times

        for exc_ncon, spike_times in exc_ncon_to_input_spike_times.items():
            input_exc_sptimes[exc_ncon] = spike_times
        for inh_ncon, spike_times in inh_ncon_to_input_spike_times.items():
            input_inh_sptimes[inh_ncon] = spike_times

        logger.info(f"Input spike applied fine, there were {count_exc_events} exc spikes and {count_inh_events} inh spikes.")

    # run sim
    cvode = h.CVode()
    if args.use_cvode:
        cvode.active(1)
    else:
        h.dt = args.dt
    h.tstop = simulation_duration_in_ms
    h.v_init = args.v_init
    fih = h.FInitializeHandler(apply_input_spike_times)
    somatic_voltage_vec = h.Vector().record(cell.soma[0](0.5)._ref_v)
    time_vec = h.Vector().record(h._ref_t)

    if args.record_dendritic_voltages:
        dendritic_voltage_vecs = []
        for segment in syns.segments:
            dendritic_voltage_vec = h.Vector()
            dendritic_voltage_vec.record(segment._ref_v)
            dendritic_voltage_vecs.append(dendritic_voltage_vec)

    if args.record_synaptic_traces:
        exc_i_AMPA_vecs = []
        exc_i_NMDA_vecs = []
        exc_g_AMPA_vecs = []
        exc_g_NMDA_vecs = []
        
        inh_i_GABAA_vecs = []
        inh_i_GABAB_vecs = []
        inh_g_GABAA_vecs = []
        inh_g_GABAB_vecs = []
        
        for exc_synapse in syns.exc_synapses:
            exc_i_AMPA_vec = h.Vector()
            exc_i_AMPA_vec.record(exc_synapse._ref_i_AMPA)
            exc_i_AMPA_vecs.append(exc_i_AMPA_vec)

            exc_i_NMDA_vec = h.Vector()
            exc_i_NMDA_vec.record(exc_synapse._ref_i_NMDA)
            exc_i_NMDA_vecs.append(exc_i_NMDA_vec)

            exc_g_AMPA_vec = h.Vector()
            exc_g_AMPA_vec.record(exc_synapse._ref_g_AMPA)
            exc_g_AMPA_vecs.append(exc_g_AMPA_vec)

            exc_g_NMDA_vec = h.Vector()
            exc_g_NMDA_vec.record(exc_synapse._ref_g_NMDA)
            exc_g_NMDA_vecs.append(exc_g_NMDA_vec)

        for inh_synapse in syns.inh_synapses:
            inh_i_GABAA_vec = h.Vector()
            inh_i_GABAA_vec.record(inh_synapse._ref_i_GABAA)
            inh_i_GABAA_vecs.append(inh_i_GABAA_vec)

            inh_i_GABAB_vec = h.Vector()
            inh_i_GABAB_vec.record(inh_synapse._ref_i_GABAB)
            inh_i_GABAB_vecs.append(inh_i_GABAB_vec)

            inh_g_GABAA_vec = h.Vector()
            inh_g_GABAA_vec.record(inh_synapse._ref_g_GABAA)
            inh_g_GABAA_vecs.append(inh_g_GABAA_vec)

            inh_g_GABAB_vec = h.Vector()
            inh_g_GABAB_vec.record(inh_synapse._ref_g_GABAB)
            inh_g_GABAB_vecs.append(inh_g_GABAB_vec)         

    logger.info("Going to h.run()...")
    h_run_start_time = time.time()
    h.run()
    h_run_duration_in_seconds = time.time() - h_run_start_time
    logger.info(f"h.run() finished!, it took {h_run_duration_in_seconds/60.0:.3f} minutes")

    np_somatic_voltage_vec = np.array(somatic_voltage_vec)
    np_time_vec = np.array(time_vec)

    recording_time_low_res = np.arange(0, simulation_duration_in_ms)
    somatic_voltage_low_res = np.interp(recording_time_low_res, np_time_vec, np_somatic_voltage_vec)

    recording_time_high_res = np.arange(0, simulation_duration_in_ms, 1.0/args.count_samples_for_high_res)
    somatic_voltage_high_res = np.interp(recording_time_high_res, np_time_vec, np_somatic_voltage_vec)

    if args.record_dendritic_voltages:
        dendritic_voltages_low_res = np.zeros((len(dendritic_voltage_vecs), recording_time_low_res.shape[0]))
        dendritic_voltages_high_res = np.zeros((len(dendritic_voltage_vecs), recording_time_high_res.shape[0]))
        for segment_index, dendritic_voltage_vec in enumerate(dendritic_voltage_vecs):
            dendritic_voltages_low_res[segment_index,:] = np.interp(recording_time_low_res, np_time_vec, np.array(dendritic_voltage_vec.as_numpy()))
            dendritic_voltages_high_res[segment_index,:] = np.interp(recording_time_high_res, np_time_vec, np.array(dendritic_voltage_vec.as_numpy()))
    else:
        dendritic_voltages_low_res = None
        dendritic_voltages_high_res = None

    if args.record_synaptic_traces:
        exc_i_AMPA_low_res = np.zeros((len(exc_i_AMPA_vecs), recording_time_low_res.shape[0]))
        exc_i_NMDA_low_res = np.zeros((len(exc_i_NMDA_vecs), recording_time_low_res.shape[0]))
        exc_g_AMPA_low_res = np.zeros((len(exc_g_AMPA_vecs), recording_time_low_res.shape[0]))
        exc_g_NMDA_low_res = np.zeros((len(exc_g_NMDA_vecs), recording_time_low_res.shape[0]))

        exc_i_AMPA_high_res = np.zeros((len(exc_i_AMPA_vecs), recording_time_high_res.shape[0]))
        exc_i_NMDA_high_res = np.zeros((len(exc_i_NMDA_vecs), recording_time_high_res.shape[0]))
        exc_g_AMPA_high_res = np.zeros((len(exc_g_AMPA_vecs), recording_time_high_res.shape[0]))
        exc_g_NMDA_high_res = np.zeros((len(exc_g_NMDA_vecs), recording_time_high_res.shape[0]))

        inh_i_GABAA_low_res = np.zeros((len(inh_i_GABAA_vecs), recording_time_low_res.shape[0]))
        inh_i_GABAB_low_res = np.zeros((len(inh_i_GABAB_vecs), recording_time_low_res.shape[0]))
        inh_g_GABAA_low_res = np.zeros((len(inh_g_GABAA_vecs), recording_time_low_res.shape[0]))
        inh_g_GABAB_low_res = np.zeros((len(inh_g_GABAB_vecs), recording_time_low_res.shape[0]))

        inh_i_GABAA_high_res = np.zeros((len(inh_i_GABAA_vecs), recording_time_high_res.shape[0]))
        inh_i_GABAB_high_res = np.zeros((len(inh_i_GABAB_vecs), recording_time_high_res.shape[0]))
        inh_g_GABAA_high_res = np.zeros((len(inh_g_GABAA_vecs), recording_time_high_res.shape[0]))
        inh_g_GABAB_high_res = np.zeros((len(inh_g_GABAB_vecs), recording_time_high_res.shape[0]))

        for i, exc_i_AMPA_vec in enumerate(exc_i_AMPA_vecs):
            exc_i_AMPA_low_res[i,:] = np.interp(recording_time_low_res, np_time_vec, np.array(exc_i_AMPA_vec.as_numpy()))
            exc_i_AMPA_high_res[i,:] = np.interp(recording_time_high_res, np_time_vec, np.array(exc_i_AMPA_vec.as_numpy()))

        for i, exc_i_NMDA_vec in enumerate(exc_i_NMDA_vecs):
            exc_i_NMDA_low_res[i,:] = np.interp(recording_time_low_res, np_time_vec, np.array(exc_i_NMDA_vec.as_numpy()))
            exc_i_NMDA_high_res[i,:] = np.interp(recording_time_high_res, np_time_vec, np.array(exc_i_NMDA_vec.as_numpy()))

        for i, exc_g_AMPA_vec in enumerate(exc_g_AMPA_vecs):
            exc_g_AMPA_low_res[i,:] = np.interp(recording_time_low_res, np_time_vec, np.array(exc_g_AMPA_vec.as_numpy()))
            exc_g_AMPA_high_res[i,:] = np.interp(recording_time_high_res, np_time_vec, np.array(exc_g_AMPA_vec.as_numpy()))

        for i, exc_g_NMDA_vec in enumerate(exc_g_NMDA_vecs):
            exc_g_NMDA_low_res[i,:] = np.interp(recording_time_low_res, np_time_vec, np.array(exc_g_NMDA_vec.as_numpy()))
            exc_g_NMDA_high_res[i,:] = np.interp(recording_time_high_res, np_time_vec, np.array(exc_g_NMDA_vec.as_numpy()))

        for i, inh_i_GABAA_vec in enumerate(inh_i_GABAA_vecs):
            inh_i_GABAA_low_res[i,:] = np.interp(recording_time_low_res, np_time_vec, np.array(inh_i_GABAA_vec.as_numpy()))
            inh_i_GABAA_high_res[i,:] = np.interp(recording_time_high_res, np_time_vec, np.array(inh_i_GABAA_vec.as_numpy()))

        for i, inh_i_GABAB_vec in enumerate(inh_i_GABAB_vecs):
            inh_i_GABAB_low_res[i,:] = np.interp(recording_time_low_res, np_time_vec, np.array(inh_i_GABAB_vec.as_numpy()))
            inh_i_GABAB_high_res[i,:] = np.interp(recording_time_high_res, np_time_vec, np.array(inh_i_GABAB_vec.as_numpy()))

        for i, inh_g_GABAA_vec in enumerate(inh_g_GABAA_vecs):
            inh_g_GABAA_low_res[i,:] = np.interp(recording_time_low_res, np_time_vec, np.array(inh_g_GABAA_vec.as_numpy()))
            inh_g_GABAA_high_res[i,:] = np.interp(recording_time_high_res, np_time_vec, np.array(inh_g_GABAA_vec.as_numpy()))

        for i, inh_g_GABAB_vec in enumerate(inh_g_GABAB_vecs):
            inh_g_GABAB_low_res[i,:] = np.interp(recording_time_low_res, np_time_vec, np.array(inh_g_GABAB_vec.as_numpy()))
            inh_g_GABAB_high_res[i,:] = np.interp(recording_time_high_res, np_time_vec, np.array(inh_g_GABAB_vec.as_numpy()))

    else:
        exc_i_AMPA_low_res = None
        exc_i_NMDA_low_res = None
        exc_g_AMPA_low_res = None
        exc_g_NMDA_low_res = None
        exc_i_AMPA_high_res = None
        exc_i_NMDA_high_res = None
        exc_g_AMPA_high_res = None
        exc_g_NMDA_high_res = None
        inh_i_GABAA_low_res = None
        inh_i_GABAB_low_res = None
        inh_g_GABAA_low_res = None
        inh_g_GABAB_low_res = None
        inh_i_GABAA_high_res = None
        inh_i_GABAB_high_res = None
        inh_g_GABAA_high_res = None
        inh_g_GABAB_high_res = None    

    recordings = {}
    recordings['recording_time_low_res'] = recording_time_low_res
    recordings['somatic_voltage_low_res'] = somatic_voltage_low_res
    recordings['recording_time_high_res'] = recording_time_high_res
    recordings['somatic_voltage_high_res'] = somatic_voltage_high_res
    recordings['dendritic_voltages_low_res'] = dendritic_voltages_low_res
    recordings['dendritic_voltages_high_res'] = dendritic_voltages_high_res
    recordings['exc_i_AMPA_low_res'] = exc_i_AMPA_low_res
    recordings['exc_i_NMDA_low_res'] = exc_i_NMDA_low_res
    recordings['exc_g_AMPA_low_res'] = exc_g_AMPA_low_res
    recordings['exc_g_NMDA_low_res'] = exc_g_NMDA_low_res
    recordings['exc_i_AMPA_high_res'] = exc_i_AMPA_high_res
    recordings['exc_i_NMDA_high_res'] = exc_i_NMDA_high_res
    recordings['exc_g_AMPA_high_res'] = exc_g_AMPA_high_res
    recordings['exc_g_NMDA_high_res'] = exc_g_NMDA_high_res
    recordings['inh_i_GABAA_low_res'] = inh_i_GABAA_low_res
    recordings['inh_i_GABAB_low_res'] = inh_i_GABAB_low_res
    recordings['inh_g_GABAA_low_res'] = inh_g_GABAA_low_res
    recordings['inh_g_GABAB_low_res'] = inh_g_GABAB_low_res
    recordings['inh_i_GABAA_high_res'] = inh_i_GABAA_high_res
    recordings['inh_i_GABAB_high_res'] = inh_i_GABAB_high_res
    recordings['inh_g_GABAA_high_res'] = inh_g_GABAA_high_res
    recordings['inh_g_GABAB_high_res'] = inh_g_GABAB_high_res

    output_spike_indexes = peakutils.indexes(somatic_voltage_high_res, thres=args.spike_threshold_for_computation, thres_abs=True)
    output_spike_times = recording_time_high_res[output_spike_indexes].astype(int)

    output_data = {}

    if args.record_dendritic_voltages:
        output_data['len_dendritic_voltage_vecs'] = len(dendritic_voltage_vecs)
    if args.record_synaptic_traces:
        output_data['len_exc_i_AMPA_vecs'] = len(exc_i_AMPA_vecs)
        output_data['len_exc_i_NMDA_vecs'] = len(exc_i_NMDA_vecs)
        output_data['len_exc_g_AMPA_vecs'] = len(exc_g_AMPA_vecs)
        output_data['len_exc_g_NMDA_vecs'] = len(exc_g_NMDA_vecs)
        output_data['len_inh_i_GABAA_vecs'] = len(inh_i_GABAA_vecs)
        output_data['len_inh_i_GABAB_vecs'] = len(inh_i_GABAB_vecs)
        output_data['len_inh_g_GABAA_vecs'] = len(inh_g_GABAA_vecs)
        output_data['len_inh_g_GABAB_vecs'] = len(inh_g_GABAB_vecs)

    return output_spike_times, somatic_voltage_low_res, recordings, output_data
    
def run_actual_simulation(args, create_model_function=create_neuron_model, run_model_function=run_neuron_model):
    cell, syns = create_model_function(args)

    simulation_duration_in_ms, exc_ncon_to_input_spike_times, inh_ncon_to_input_spike_times, exc_weighted_spikes, inh_weighted_spikes, auxiliary_information = generate_spike_times_and_weights(args, syns)

    output_spike_times, somatic_voltage_low_res, recordings, output_data = run_model_function(args, cell, syns, simulation_duration_in_ms, exc_ncon_to_input_spike_times, inh_ncon_to_input_spike_times, exc_weighted_spikes, inh_weighted_spikes, auxiliary_information)

    # relevant when using a non NEURON model
    recordings['somatic_voltage_low_res'] = somatic_voltage_low_res

    output_firing_rate = len(output_spike_times)/(simulation_duration_in_ms/1000.0)
    output_isi = np.diff(output_spike_times)
    
    output_spike_times_after_initialization = output_spike_times[output_spike_times > args.simulation_initialization_duration_in_ms]
    output_firing_rate_after_initialization = len(output_spike_times_after_initialization)/((simulation_duration_in_ms - args.simulation_initialization_duration_in_ms)/1000.0)
    output_isi_after_initialization = np.diff(output_spike_times)

    average_somatic_voltage = np.mean(somatic_voltage_low_res)

    clipped_somatic_voltage_low_res = np.copy(somatic_voltage_low_res)
    clipped_somatic_voltage_low_res[clipped_somatic_voltage_low_res>args.spike_threshold] = args.spike_threshold
    average_clipped_somatic_voltage = np.mean(clipped_somatic_voltage_low_res)

    output_data['args'] = args

    output_data['len_exc_netcons'] = len(syns.exc_netcons)
    output_data['len_inh_netcons'] = len(syns.inh_netcons)

    output_data['input_count_exc_spikes'] = auxiliary_information['count_exc_spikes']
    output_data['input_average_exc_spikes_per_second'] = auxiliary_information['average_exc_spikes_per_second']
    output_data['input_count_exc_spikes_per_super_synapse'] = auxiliary_information['count_exc_spikes_per_super_synapse']
    output_data['input_average_exc_spikes_per_super_synapse_per_second'] = auxiliary_information['average_exc_spikes_per_super_synapse_per_second']
    output_data['input_count_weighted_exc_spikes'] = auxiliary_information['count_weighted_exc_spikes']
    output_data['input_average_weighted_exc_spikes_per_second'] = auxiliary_information['average_weighted_exc_spikes_per_second']
    output_data['input_count_weighted_exc_spikes_per_super_synapse'] = auxiliary_information['count_weighted_exc_spikes_per_super_synapse']
    output_data['input_average_weighted_exc_spikes_per_super_synapse_per_second'] = auxiliary_information['average_weighted_exc_spikes_per_super_synapse_per_second']
    output_data['input_count_inh_spikes'] = auxiliary_information['count_inh_spikes']
    output_data['input_average_inh_spikes_per_second'] = auxiliary_information['average_inh_spikes_per_second']
    output_data['input_count_inh_spikes_per_super_synapse'] = auxiliary_information['count_inh_spikes_per_super_synapse']
    output_data['input_average_inh_spikes_per_super_synapse_per_second'] = auxiliary_information['average_inh_spikes_per_super_synapse_per_second']
    output_data['input_count_weighted_inh_spikes'] = auxiliary_information['count_weighted_inh_spikes']
    output_data['input_average_weighted_inh_spikes_per_second'] = auxiliary_information['average_weighted_inh_spikes_per_second']
    output_data['input_count_weighted_inh_spikes_per_super_synapse'] = auxiliary_information['count_weighted_inh_spikes_per_super_synapse']
    output_data['input_average_weighted_inh_spikes_per_super_synapse_per_second'] = auxiliary_information['average_weighted_inh_spikes_per_super_synapse_per_second']

    output_data['average_exc_initial_neuron_weight'] = auxiliary_information['average_exc_initial_neuron_weight']
    output_data['average_inh_initial_neuron_weight'] = auxiliary_information['average_inh_initial_neuron_weight']
    
    output_data['output_spike_times'] = output_spike_times

    output_data['output_firing_rate'] = output_firing_rate
    output_data['output_isi'] = output_isi
    output_data['output_spike_times_after_initialization'] = output_spike_times_after_initialization
    output_data['output_firing_rate_after_initialization'] = output_firing_rate_after_initialization
    output_data['output_isi_after_initialization'] = output_isi_after_initialization
    
    output_data['simulation_duration_in_ms'] = simulation_duration_in_ms
    output_data['average_somatic_voltage'] = average_somatic_voltage
    output_data['average_clipped_somatic_voltage'] = average_clipped_somatic_voltage
    
    if args.save_auxiliary_information:
        output_data['auxiliary_information'] = auxiliary_information

    return output_data, exc_weighted_spikes, inh_weighted_spikes, recordings, auxiliary_information, cell, syns

def run_simulation(args, create_model_function=create_neuron_model, run_model_function=run_neuron_model):
    logger.info("Going to run simulation with args:")
    logger.info("{}".format(args))
    logger.info("...")

    if args.simple_stimulation:
        args.generate_weights_using_constrained_linear = False
        args.multiple_connections_prob = args.simple_stimulation_multiple_connections_prob

        args.multiply_count_initial_synapses_per_super_synapse_prob = 0.0
        args.same_exc_inh_count_initial_synapses_per_super_synapse_prob = 0.0

        # I onced use these, but they give unnatural behavior, as no more than one spike per one ms in one branch is possible!
        # args.force_count_initial_synapses_per_super_synapse = 1
        # args.force_multiply_count_spikes_per_synapse_per_100ms_range_by_average_segment_length = True

        args.synchronization_prob = 0.0
        args.remove_inhibition_prob = 0.0
        args.deactivate_synapses_prob = 0.0
        args.spatial_clustering_prob = 0.0
        args.same_exc_inh_inst_rate_prob = 0.0
        args.same_exc_inh_spikes_bin_prob = 0.0
        args.same_exc_inh_all_kernels_prob = 0.0
        args.same_exc_inh_kernels_prob = 0.0

        args.count_special_intervals = 0

        args.spatial_multiplicative_randomness = args.simple_stimulation_spatial_multiplicative_randomness

    if args.default_weighted:
        args.exc_weights_ratio_range = [0.0, 2.0]
        args.inh_weights_ratio_range = [0.0, 2.0]

    if args.wide_weighted is not None:
        args.exc_weights_ratio_range = [0.0, args.wide_weighted]
        if args.wide_weighted_inh_multiplicative_factor:
            args.inh_weights_ratio_range = [0.0, args.wide_weighted * args.wide_weighted_inh_multiplicative_factor]
        else:
            args.inh_weights_ratio_range = [0.0, args.wide_weighted]

    if args.wide_fr:
        args.count_exc_spikes_per_synapse_per_100ms_range = [args.wide_fr_base * 0.1, (args.wide_fr_base + args.wide_fr) * 0.1]
        args.count_inh_spikes_per_synapse_per_100ms_range = [(args.wide_fr_base + args.wide_fr_inh_additive_factor) * 0.1,
         (args.wide_fr_base + args.wide_fr + args.wide_fr_inh_additive_factor) * 0.1]
        if args.wide_fr_inh_adaptive_additive_factor:
            args.adaptive_inh = True
            args.adaptive_inh_additive_factor_per_synapse_per_100ms_range = [args.wide_fr_inh_adaptive_additive_factor[0] * 0.1, args.wide_fr_inh_adaptive_additive_factor[1] * 0.1]

    logger.info("After shortcuts, args are:")
    logger.info("{}".format(args))

    os.makedirs(args.simulation_folder, exist_ok=True)

    run_simulation_start_time = time.time()

    random_seed = args.random_seed
    if random_seed is None:
        random_seed = int(time.time())
    logger.info(f"seeding with random_seed={random_seed}")
    np.random.seed(random_seed)

    if args.neuron_model_folder is not None:
        # trying to fix neuron crashes
        time.sleep(1 + 30*np.random.random())

        logger.info("About to import neuron...")
        logger.info(f"current dir: {pathlib.Path(__file__).parent.absolute()}")
        
        global neuron
        global h
        global gui
        import neuron
        from neuron import h
        from neuron import gui
        logger.info("Neuron imported fine.")

    simulation_trial = 0
    output_data, exc_weighted_spikes, inh_weighted_spikes, recordings, auxiliary_information, cell, syns = run_actual_simulation(args, create_model_function=create_model_function, run_model_function=run_model_function)
    output_firing_rate = output_data['output_firing_rate']
    output_firing_rate_after_initialization = output_data['output_firing_rate_after_initialization']
    simulation_trial += 1

    while output_firing_rate <= 0.0 and simulation_trial < args.count_trials_for_nonzero_output_firing_rate:
        logger.info(f"Firing rate is {output_firing_rate:.3f}, Firing rate after initialization is {output_firing_rate_after_initialization:.3f}")
        logger.info(f"Retrying simulation, {simulation_trial} trial")
        output_data, exc_weighted_spikes, inh_weighted_spikes, recordings, auxiliary_information, cell, syns = run_actual_simulation(args, create_model_function=create_model_function, run_model_function=run_model_function)
        output_firing_rate = output_data['output_firing_rate']
        output_firing_rate_after_initialization = output_data['output_firing_rate_after_initialization']
        simulation_trial += 1

    logger.info(f"Firing rate is {output_firing_rate:.3f}, Firing rate after initialization is {output_firing_rate_after_initialization:.3f}")
    logger.info(f"output_spike_times are {output_data['output_spike_times']}")
    logger.info(f"Simulation finished after {simulation_trial} trials")

    pickle.dump(output_data, open(f'{args.simulation_folder}/summary.pkl','wb'), protocol=-1)

    sparse.save_npz(f'{args.simulation_folder}/exc_weighted_spikes.npz', sparse.csr_matrix(exc_weighted_spikes))
    sparse.save_npz(f'{args.simulation_folder}/inh_weighted_spikes.npz', sparse.csr_matrix(inh_weighted_spikes))

    f = h5py.File(f'{args.simulation_folder}/voltage.h5','w')
    f.create_dataset('somatic_voltage', data=recordings['somatic_voltage_low_res'])
    if args.record_dendritic_voltages:
        f.create_dataset('dendritic_voltage', data=recordings['dendritic_voltages_low_res'])
    if args.record_synaptic_traces:
        f.create_dataset('exc_i_AMPA', data=recordings['exc_i_AMPA_low_res'])
        f.create_dataset('exc_i_NMDA', data=recordings['exc_i_NMDA_low_res'])
        f.create_dataset('exc_g_AMPA', data=recordings['exc_g_AMPA_low_res'])
        f.create_dataset('exc_g_NMDA', data=recordings['exc_g_NMDA_low_res'])
        f.create_dataset('inh_i_GABAA', data=recordings['inh_i_GABAA_low_res'])
        f.create_dataset('inh_i_GABAB', data=recordings['inh_i_GABAB_low_res'])
        f.create_dataset('inh_g_GABAA', data=recordings['inh_g_GABAA_low_res'])
        f.create_dataset('inh_g_GABAB', data=recordings['inh_g_GABAB_low_res'])
    f.close()

    if args.save_plots:

        # io plot
        ws = np.vstack((exc_weighted_spikes, inh_weighted_spikes))
        half_syn = exc_weighted_spikes.shape[0]
        count_spikes = len(output_data['output_spike_times'])
        name = os.path.basename(args.simulation_folder)
        avg_exc = output_data['input_average_weighted_exc_spikes_per_super_synapse_per_second']
        avg_inh = output_data['input_average_weighted_inh_spikes_per_super_synapse_per_second']

        if 'recording_time_high_res' in recordings and 'somatic_voltage_high_res' in recordings:
            recording_time_high_res = recordings['recording_time_high_res']
            somatic_voltage_high_res = recordings['somatic_voltage_high_res']
        else:
            # for non NEURON models
            somatic_voltage_high_res = recordings['somatic_voltage_low_res']
            recording_time_high_res = np.array(range(somatic_voltage_high_res.shape[0]))

        max_weight = ws.max()

        is_weighted = args.exc_weights_ratio_range[0] < args.exc_weights_ratio_range[1]\
             or args.inh_weights_ratio_range[0] < args.inh_weights_ratio_range[1]\
                 or max_weight > 1 or args.default_weighted or args.wide_weighted is not None

        plot_input_spikes = is_weighted and 'exc_initial_neurons_spikes_bin' in auxiliary_information and 'inh_initial_neurons_spikes_bin' in auxiliary_information

        if plot_input_spikes:
            fig = plt.figure(figsize=(25,15))
            axs = fig.subplots(3,1, sharex=True)
        else:
            fig = plt.figure(figsize=(25,10))
            axs = fig.subplots(2,1, sharex=True)

        fig.suptitle(f'{name}\nAverage input per segment: {avg_exc:.3f} exc Hz, {avg_inh:.3f} inh Hz\n'+
        f'Output: {count_spikes} spikes ({output_firing_rate:.3f} Hz)', fontsize=20)

        if plot_input_spikes:
            input_spikes = np.vstack((auxiliary_information['exc_initial_neurons_spikes_bin'], auxiliary_information['inh_initial_neurons_spikes_bin']))
            half_axon = auxiliary_information['exc_initial_neurons_spikes_bin'].shape[0]
            axs[0].matshow(input_spikes, cmap='binary', aspect='auto')
            axs[0].xaxis.set_ticks_position("bottom")
            axs[0].set_xlabel('Time (ms)')
            axs[0].set_ylabel('Axon')
            axs[0].axhline(half_axon, color='green')
            axs[0].set_title('Input Spikes')

        if max_weight > 1:
            first = int(128 / (max_weight-1))
            colors1 = plt.cm.binary(np.linspace(0., 1, first))
            colors2 = plt.cm.hot(np.linspace(0, 0.8, 256-first))
            colors = np.vstack((colors1, colors2))
            mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
            chosen_cmap = mymap
        else:
            chosen_cmap = plt.cm.binary

        # divider = make_axes_locatable(axs[0])
        # cax = divider.append_axes('right', size='5%', pad=0.05)

        # axs[0].matshow(ws, cmap='Reds', vmin=0, vmax=5)
        # axs[0].matshow(ws, cmap='binary', vmin=0, vmax=1)

        ax_index = 1 if plot_input_spikes else 0

        im = axs[ax_index].matshow(ws, cmap=chosen_cmap, vmin=0, vmax=max_weight, aspect='auto')
        if is_weighted:
            cax = fig.add_axes([0.27, 0.80, 0.5, 0.05])
            fig.colorbar(im, cax=cax, orientation='horizontal')
        # fig.colorbar(im, cax=cax, orientation='vertical')

        axs[ax_index].axhline(half_syn, color='green')
        axs[ax_index].xaxis.set_ticks_position("bottom")
        axs[ax_index].set_xlabel('Time (ms)')
        axs[ax_index].set_ylabel('Segment')

        if is_weighted:
            axs[ax_index].set_title('Input Weighted spikes')
        else:
            axs[ax_index].set_title('Input Spikes')

        ax_index += 1
        axs[ax_index].plot(recording_time_high_res, somatic_voltage_high_res)
        axs[ax_index].set_xlabel('Time (ms)')
        axs[ax_index].set_ylabel('Voltage (mV)')
        # axs[ax_index].set_xlim(0, 3000)
        axs[ax_index].set_xlim(0, exc_weighted_spikes.shape[1])
        axs[ax_index].set_title('Output Somatic voltage')
        if is_weighted:
            plt.subplots_adjust(bottom=0.2, top=0.73, hspace=0.35)
        else:
            plt.subplots_adjust(bottom=0.2, top=0.82, hspace=0.3)
        fig.savefig(f'{args.simulation_folder}/io.png')
        plt.close('all')

        exc_wiring_matrix = auxiliary_information['exc_wiring_matrix']
        inh_wiring_matrix = auxiliary_information['inh_wiring_matrix']
        if exc_wiring_matrix is not None and inh_wiring_matrix is not None:
            fig = plt.figure(figsize=(10,10))
            axs = fig.subplots(1,2)
            axs[0].matshow(exc_wiring_matrix, cmap='hot', aspect='auto')
            exc_stats_string = f"avg per row is {np.mean(np.sum(exc_wiring_matrix, axis=1)):.3f}, avg per col is {np.mean(np.sum(exc_wiring_matrix, axis=0)):.3f}"
            axs[0].set_title(f'Exc wiring matrix\n{exc_stats_string}')
            axs[0].xaxis.set_ticks_position('bottom')
            axs[0].set_xlabel("axon")
            axs[0].set_ylabel("segment")

            axs[1].matshow(inh_wiring_matrix, cmap='hot', aspect='auto')
            inh_stats_string = f"avg per row is {np.mean(np.sum(inh_wiring_matrix, axis=1)):.3f}, avg per col is {np.mean(np.sum(inh_wiring_matrix, axis=0)):.3f}"
            axs[1].set_title(f'Inh wiring matrix\n{inh_stats_string}')
            axs[1].xaxis.set_ticks_position('bottom')
            axs[1].set_xlabel("axon")
            axs[1].set_ylabel("segment")
            fig.savefig(f'{args.simulation_folder}/wiring.png')
            plt.close('all')

        if args.record_dendritic_voltages:
            plotter = NeuronPlotter(cell, list(syns['segments']))

            seg_id_to_average_seg_y = {}
            for seg_id, seg in enumerate(syns['segments']):
                seg_id_to_average_seg_y[seg_id] = plotter.get_seg_coord(seg_id)[1].mean()
            average_seg_ys = np.array([seg_id_to_average_seg_y[seg_id] for seg_id in range(len(syns['segments']))])

            fig = plt.figure(figsize=(32, 18))
            fig.subplots_adjust(left=0.01,right=0.99,top=0.99,bottom=0.01,hspace=0.01, wspace=0.2)
            ax1 = plt.subplot2grid((1, 4), (0, 0), colspan=1)
            ax2 = plt.subplot2grid((1, 4), (0, 1), colspan=3)

            ax2.set_axis_off()
            segmentd_index = np.array(list(range(output_data['len_dendritic_voltage_vecs'])))
            # dend_colors = segmentd_index*20
            # dend_colors = segmentd_index*20 + average_seg_ys
            dend_colors = average_seg_ys
            dend_colors = dend_colors / dend_colors.max()
            # colors = plt.cm.jet(dend_colors)
            colors = plt.cm.brg(dend_colors)
            sorted_according_to_colors = np.argsort(dend_colors) # segment indices by color
            delta_voltage = 1700.0 / sorted_according_to_colors.shape[0]
            add_from_bottom = 100
            for color_index, k in enumerate(sorted_according_to_colors):
                ax2.plot(recording_time_high_res, add_from_bottom+color_index*delta_voltage+recordings['dendritic_voltages_high_res'][k,:].T, c=colors[k], alpha=0.5)

            plotter.plot_shape(ax=ax1, allSegsColor=colors)
            ax1.set_axis_off()

            ax2.plot(recording_time_high_res, recordings['somatic_voltage_high_res'], c='black', lw=2.4)
            fig.savefig(f'{args.simulation_folder}/dendritic_voltage.png')
            plt.close('all')

            # plotting specific locations

            count_locations_to_plot = 20

            # plot count_locations_to_plot locations, distributed on the tree
            fig = plt.figure(figsize=(32, 15))
            fig.subplots_adjust(left=0.01,right=0.99,top=0.99,bottom=0.01,hspace=0.01, wspace=0.2)
            ax1 = plt.subplot2grid((1, 4), (0, 0), colspan=1)
            ax2 = plt.subplot2grid((1, 4), (0, 1), colspan=3)

            ax2.set_axis_off()
            segmentd_index = np.array(list(range(output_data['len_dendritic_voltage_vecs'])))
            # dend_colors = segmentd_index*20
            # dend_colors = segmentd_index*20 + average_seg_ys
            dend_colors = average_seg_ys
            dend_colors = dend_colors / dend_colors.max()
            # colors = plt.cm.jet(dend_colors)
            # colors = plt.cm.winter(dend_colors)
            # colors = plt.cm.cool(dend_colors)
            colors = plt.cm.brg(dend_colors)
            sorted_according_to_colors = np.argsort(dend_colors) # segment indices by color
            delta_voltage = 1700.0 / sorted_according_to_colors.shape[0]
            add_from_bottom = 100
            count_plotted = 0
            locations = []
            allSegsColor = ['grey' for _ in range(output_data['len_dendritic_voltage_vecs'])]
            for color_index, k in enumerate(sorted_according_to_colors):
                if color_index % (sorted_according_to_colors.shape[0] // count_locations_to_plot) == 0:
                    # ax2.plot(recording_time_high_res, 150+k*delta_voltage+recordings['dendritic_voltages_high_res'][k,:].T, c=colors[k], alpha=0.5)
                    ax2.plot(recording_time_high_res, add_from_bottom+color_index*delta_voltage+recordings['dendritic_voltages_high_res'][k,:].T, c=colors[k], alpha=0.5)

                    print(f'plotting {k} at {color_index} with color {colors[k]}')
                    min_voltage = recordings['dendritic_voltages_high_res'][k,:].min()
                    max_voltage = recordings['dendritic_voltages_high_res'][k,:].max()
                    print(f'min_voltage={min_voltage}, max_voltage={max_voltage}')
                    avg_voltage = recordings['dendritic_voltages_high_res'][k,:].mean()
                    std_voltage = recordings['dendritic_voltages_high_res'][k,:].std()
                    print(f'avg_voltage={avg_voltage}, std_voltage={std_voltage}')

                    # plot from avg-std to avg+std, vertically
                    ax2.plot([-100, -100], [add_from_bottom+color_index*delta_voltage+avg_voltage-std_voltage, add_from_bottom+color_index*delta_voltage+avg_voltage+std_voltage], c=colors[k], lw=2)
                    # scatter the avg
                    ax2.scatter(-100, add_from_bottom+color_index*delta_voltage+avg_voltage, color=colors[k], s=10)
                    # annotate the avg
                    ax2.text(-80, add_from_bottom+color_index*delta_voltage+avg_voltage, f'{avg_voltage:.3f}', fontsize=8, color=colors[k])
                    # # annotate the avg+std, avg-std
                    # ax2.text(-80, add_from_bottom+color_index*delta_voltage+avg_voltage+std_voltage, f'{avg_voltage+std_voltage:.3f}', fontsize=10, color=colors[k])
                    # ax2.text(-80, add_from_bottom+color_index*delta_voltage+avg_voltage-std_voltage, f'{avg_voltage-std_voltage:.3f}', fontsize=10, color=colors[k])

                    # plot from min to max, vertically
                    ax2.plot([-100, -100], [add_from_bottom+color_index*delta_voltage+min_voltage, add_from_bottom+color_index*delta_voltage+max_voltage], c=colors[k], lw=1)

                    # scatter the min and max
                    ax2.scatter(-100, add_from_bottom+color_index*delta_voltage+min_voltage, color=colors[k], s=10)
                    ax2.scatter(-100, add_from_bottom+color_index*delta_voltage+max_voltage, color=colors[k], s=10)
                    # annotate the min and max
                    ax2.text(-80, add_from_bottom+color_index*delta_voltage+min_voltage, f'{min_voltage:.3f}', fontsize=8, color=colors[k])
                    ax2.text(-80, add_from_bottom+color_index*delta_voltage+max_voltage, f'{max_voltage:.3f}', fontsize=8, color=colors[k])
                    
                    # # plot a voltage scale, from 0 to scale_top mV
                    # scale_top = 30
                    # ax2.plot([-200, -200], [add_from_bottom+color_index*delta_voltage, add_from_bottom+color_index*delta_voltage+scale_top], c='black', lw=2.4)
                    
                    # # annotate the scale (the 0 and the 50)
                    # ax2.text(-180, add_from_bottom+color_index*delta_voltage, f'0mV', fontsize=10, color=colors[k])
                    # ax2.text(-180, add_from_bottom+color_index*delta_voltage+scale_top, f'{scale_top}mV', fontsize=10, color=colors[k])

                    # annotate the segment index
                    ax2.text(-350, add_from_bottom//2+color_index*delta_voltage, f'{k}', fontsize=20, color=colors[k])
                    locations.append(k)
                    allSegsColor[k] = colors[k]

            ax2.plot(recording_time_high_res, recordings['somatic_voltage_high_res'], c='black', lw=2.4)

            plotter.plot_shape(ax=ax1, allSegsColor=allSegsColor)
            ax1.set_axis_off()
            for location in locations:
                seg_x, seg_y = plotter.get_seg_coord(location)
                cur_average_seg_y = seg_y.mean()
                cur_average_seg_x = seg_x.mean()
                ax1.text(cur_average_seg_x, cur_average_seg_y, f'{location}', fontsize=20, color=allSegsColor[location])
                ax1.scatter(cur_average_seg_x, cur_average_seg_y, color=allSegsColor[location], s=100)

            fig.savefig(f'{args.simulation_folder}/dendritic_voltage_{count_locations_to_plot}_locations.png')
            plt.close('all')

            # plot from 500ms to 1000ms
            high_res_start_index = int(500*args.count_samples_for_high_res)
            high_res_end_index = int(1000*args.count_samples_for_high_res)

            # plot count_locations_to_plot locations, distributed on the tree
            fig = plt.figure(figsize=(32, 15))
            fig.subplots_adjust(left=0.01,right=0.99,top=0.99,bottom=0.01,hspace=0.01, wspace=0.2)
            ax1 = plt.subplot2grid((1, 4), (0, 0), colspan=1)
            ax2 = plt.subplot2grid((1, 4), (0, 1), colspan=3)

            ax2.set_axis_off()
            segmentd_index = np.array(list(range(output_data['len_dendritic_voltage_vecs'])))
            # dend_colors = segmentd_index*20
            # dend_colors = segmentd_index*20 + average_seg_ys
            dend_colors = average_seg_ys
            dend_colors = dend_colors / dend_colors.max()
            # colors = plt.cm.jet(dend_colors)
            # colors = plt.cm.winter(dend_colors)
            # colors = plt.cm.cool(dend_colors)
            colors = plt.cm.brg(dend_colors)
            sorted_according_to_colors = np.argsort(dend_colors) # segment indices by color
            delta_voltage = 1700.0 / sorted_according_to_colors.shape[0]
            add_from_bottom = 100
            count_plotted = 0
            locations = []
            allSegsColor = ['grey' for _ in range(output_data['len_dendritic_voltage_vecs'])]
            for color_index, k in enumerate(sorted_according_to_colors):
                if color_index % (sorted_according_to_colors.shape[0] // count_locations_to_plot) == 0:
                    # ax2.plot(recording_time_high_res, 150+k*delta_voltage+recordings['dendritic_voltages_high_res'][k,:].T, c=colors[k], alpha=0.5)
                    ax2.plot(add_from_bottom+color_index*delta_voltage+recordings['dendritic_voltages_high_res'][k,:].T[high_res_start_index:high_res_end_index], c=colors[k], alpha=0.5)

                    print(f'plotting {k} at {color_index} with color {colors[k]}')
                    min_voltage = recordings['dendritic_voltages_high_res'][k,:].min()
                    max_voltage = recordings['dendritic_voltages_high_res'][k,:].max()
                    print(f'min_voltage={min_voltage}, max_voltage={max_voltage}')
                    avg_voltage = recordings['dendritic_voltages_high_res'][k,:].mean()
                    std_voltage = recordings['dendritic_voltages_high_res'][k,:].std()
                    print(f'avg_voltage={avg_voltage}, std_voltage={std_voltage}')

                    # plot from avg-std to avg+std, vertically
                    ax2.plot([-100, -100], [add_from_bottom+color_index*delta_voltage+avg_voltage-std_voltage, add_from_bottom+color_index*delta_voltage+avg_voltage+std_voltage], c=colors[k], lw=2)
                    # scatter the avg
                    ax2.scatter(-100, add_from_bottom+color_index*delta_voltage+avg_voltage, color=colors[k], s=10)
                    # annotate the avg
                    ax2.text(-80, add_from_bottom+color_index*delta_voltage+avg_voltage, f'{avg_voltage:.3f}', fontsize=8, color=colors[k])
                    # # annotate the avg+std, avg-std
                    # ax2.text(-80, add_from_bottom+color_index*delta_voltage+avg_voltage+std_voltage, f'{avg_voltage+std_voltage:.3f}', fontsize=10, color=colors[k])
                    # ax2.text(-80, add_from_bottom+color_index*delta_voltage+avg_voltage-std_voltage, f'{avg_voltage-std_voltage:.3f}', fontsize=10, color=colors[k])

                    # plot from min to max, vertically
                    ax2.plot([-100, -100], [add_from_bottom+color_index*delta_voltage+min_voltage, add_from_bottom+color_index*delta_voltage+max_voltage], c=colors[k], lw=1)

                    # scatter the min and max
                    ax2.scatter(-100, add_from_bottom+color_index*delta_voltage+min_voltage, color=colors[k], s=10)
                    ax2.scatter(-100, add_from_bottom+color_index*delta_voltage+max_voltage, color=colors[k], s=10)
                    # annotate the min and max
                    ax2.text(-80, add_from_bottom+color_index*delta_voltage+min_voltage, f'{min_voltage:.3f}', fontsize=8, color=colors[k])
                    ax2.text(-80, add_from_bottom+color_index*delta_voltage+max_voltage, f'{max_voltage:.3f}', fontsize=8, color=colors[k])
                    
                    # # plot a voltage scale, from 0 to scale_top mV
                    # scale_top = 30
                    # ax2.plot([-200, -200], [add_from_bottom+color_index*delta_voltage, add_from_bottom+color_index*delta_voltage+scale_top], c='black', lw=2.4)
                    
                    # # annotate the scale (the 0 and the 50)
                    # ax2.text(-180, add_from_bottom+color_index*delta_voltage, f'0mV', fontsize=10, color=colors[k])
                    # ax2.text(-180, add_from_bottom+color_index*delta_voltage+scale_top, f'{scale_top}mV', fontsize=10, color=colors[k])

                    # annotate the segment index
                    ax2.text(-350, add_from_bottom//2+color_index*delta_voltage, f'{k}', fontsize=20, color=colors[k])
                    locations.append(k)
                    allSegsColor[k] = colors[k]

            ax2.plot(recordings['somatic_voltage_high_res'][high_res_start_index:high_res_end_index], c='black', lw=2.4)

            plotter.plot_shape(ax=ax1, allSegsColor=allSegsColor)
            ax1.set_axis_off()
            for location in locations:
                seg_x, seg_y = plotter.get_seg_coord(location)
                cur_average_seg_y = seg_y.mean()
                cur_average_seg_x = seg_x.mean()
                ax1.text(cur_average_seg_x, cur_average_seg_y, f'{location}', fontsize=20, color=allSegsColor[location])
                ax1.scatter(cur_average_seg_x, cur_average_seg_y, color=allSegsColor[location], s=100)

            fig.savefig(f'{args.simulation_folder}/dendritic_voltage_{count_locations_to_plot}_locations_500ms_1000ms.png')
            plt.close('all')

            # plot from 500ms to 700ms
            high_res_start_index = int(500*args.count_samples_for_high_res)
            high_res_end_index = int(700*args.count_samples_for_high_res)

            # plot count_locations_to_plot locations, distributed on the tree
            fig = plt.figure(figsize=(32, 15))
            fig.subplots_adjust(left=0.01,right=0.99,top=0.99,bottom=0.01,hspace=0.01, wspace=0.2)
            ax1 = plt.subplot2grid((1, 4), (0, 0), colspan=1)
            ax2 = plt.subplot2grid((1, 4), (0, 1), colspan=3)

            ax2.set_axis_off()
            segmentd_index = np.array(list(range(output_data['len_dendritic_voltage_vecs'])))
            # dend_colors = segmentd_index*20
            # dend_colors = segmentd_index*20 + average_seg_ys
            dend_colors = average_seg_ys
            dend_colors = dend_colors / dend_colors.max()
            # colors = plt.cm.jet(dend_colors)
            # colors = plt.cm.winter(dend_colors)
            # colors = plt.cm.cool(dend_colors)
            colors = plt.cm.brg(dend_colors)
            sorted_according_to_colors = np.argsort(dend_colors) # segment indices by color
            delta_voltage = 1700.0 / sorted_according_to_colors.shape[0]
            add_from_bottom = 100
            count_plotted = 0
            locations = []
            allSegsColor = ['grey' for _ in range(output_data['len_dendritic_voltage_vecs'])]
            for color_index, k in enumerate(sorted_according_to_colors):
                if color_index % (sorted_according_to_colors.shape[0] // count_locations_to_plot) == 0:
                    # ax2.plot(recording_time_high_res, 150+k*delta_voltage+recordings['dendritic_voltages_high_res'][k,:].T, c=colors[k], alpha=0.5)
                    ax2.plot(add_from_bottom+color_index*delta_voltage+recordings['dendritic_voltages_high_res'][k,:].T[high_res_start_index:high_res_end_index], c=colors[k], alpha=0.5)

                    print(f'plotting {k} at {color_index} with color {colors[k]}')
                    min_voltage = recordings['dendritic_voltages_high_res'][k,:].min()
                    max_voltage = recordings['dendritic_voltages_high_res'][k,:].max()
                    print(f'min_voltage={min_voltage}, max_voltage={max_voltage}')
                    avg_voltage = recordings['dendritic_voltages_high_res'][k,:].mean()
                    std_voltage = recordings['dendritic_voltages_high_res'][k,:].std()
                    print(f'avg_voltage={avg_voltage}, std_voltage={std_voltage}')

                    # plot from avg-std to avg+std, vertically
                    ax2.plot([-100, -100], [add_from_bottom+color_index*delta_voltage+avg_voltage-std_voltage, add_from_bottom+color_index*delta_voltage+avg_voltage+std_voltage], c=colors[k], lw=2)
                    # scatter the avg
                    ax2.scatter(-100, add_from_bottom+color_index*delta_voltage+avg_voltage, color=colors[k], s=10)
                    # annotate the avg
                    ax2.text(-80, add_from_bottom+color_index*delta_voltage+avg_voltage, f'{avg_voltage:.3f}', fontsize=8, color=colors[k])
                    # # annotate the avg+std, avg-std
                    # ax2.text(-80, add_from_bottom+color_index*delta_voltage+avg_voltage+std_voltage, f'{avg_voltage+std_voltage:.3f}', fontsize=10, color=colors[k])
                    # ax2.text(-80, add_from_bottom+color_index*delta_voltage+avg_voltage-std_voltage, f'{avg_voltage-std_voltage:.3f}', fontsize=10, color=colors[k])

                    # plot from min to max, vertically
                    ax2.plot([-100, -100], [add_from_bottom+color_index*delta_voltage+min_voltage, add_from_bottom+color_index*delta_voltage+max_voltage], c=colors[k], lw=1)

                    # scatter the min and max
                    ax2.scatter(-100, add_from_bottom+color_index*delta_voltage+min_voltage, color=colors[k], s=10)
                    ax2.scatter(-100, add_from_bottom+color_index*delta_voltage+max_voltage, color=colors[k], s=10)
                    # annotate the min and max
                    ax2.text(-80, add_from_bottom+color_index*delta_voltage+min_voltage, f'{min_voltage:.3f}', fontsize=8, color=colors[k])
                    ax2.text(-80, add_from_bottom+color_index*delta_voltage+max_voltage, f'{max_voltage:.3f}', fontsize=8, color=colors[k])
                    
                    # # plot a voltage scale, from 0 to scale_top mV
                    # scale_top = 30
                    # ax2.plot([-200, -200], [add_from_bottom+color_index*delta_voltage, add_from_bottom+color_index*delta_voltage+scale_top], c='black', lw=2.4)
                    
                    # # annotate the scale (the 0 and the 50)
                    # ax2.text(-180, add_from_bottom+color_index*delta_voltage, f'0mV', fontsize=10, color=colors[k])
                    # ax2.text(-180, add_from_bottom+color_index*delta_voltage+scale_top, f'{scale_top}mV', fontsize=10, color=colors[k])

                    # annotate the segment index
                    ax2.text(-350, add_from_bottom//2+color_index*delta_voltage, f'{k}', fontsize=20, color=colors[k])
                    locations.append(k)
                    allSegsColor[k] = colors[k]

            ax2.plot(recordings['somatic_voltage_high_res'][high_res_start_index:high_res_end_index], c='black', lw=2.4)

            plotter.plot_shape(ax=ax1, allSegsColor=allSegsColor)
            ax1.set_axis_off()
            for location in locations:
                seg_x, seg_y = plotter.get_seg_coord(location)
                cur_average_seg_y = seg_y.mean()
                cur_average_seg_x = seg_x.mean()
                ax1.text(cur_average_seg_x, cur_average_seg_y, f'{location}', fontsize=20, color=allSegsColor[location])
                ax1.scatter(cur_average_seg_x, cur_average_seg_y, color=allSegsColor[location], s=100)

            fig.savefig(f'{args.simulation_folder}/dendritic_voltage_{count_locations_to_plot}_locations_500ms_700ms.png')
            plt.close('all')

            count_random_location_plots = 10

            # plot count_locations_to_plot random locations, count_random_location_plots times
            for random_plot_index in range(count_random_location_plots):
                # plot count_locations_to_plot random locations
                fig = plt.figure(figsize=(32, 15))
                fig.subplots_adjust(left=0.01,right=0.99,top=0.99,bottom=0.01,hspace=0.01, wspace=0.2)
                ax1 = plt.subplot2grid((1, 4), (0, 0), colspan=1)
                ax2 = plt.subplot2grid((1, 4), (0, 1), colspan=3)

                ax2.set_axis_off()
                segmentd_index = np.array(list(range(output_data['len_dendritic_voltage_vecs'])))
                # dend_colors = segmentd_index*20
                # dend_colors = segmentd_index*20 + average_seg_ys
                dend_colors = average_seg_ys
                dend_colors = dend_colors / dend_colors.max()
                colors = plt.cm.brg(dend_colors)
                sorted_according_to_colors = np.argsort(dend_colors) # segment indices by color
                delta_voltage = 1700.0 / sorted_according_to_colors.shape[0]
                add_from_bottom = 100
                count_plotted = 0
                locations = np.random.choice(sorted_according_to_colors, count_locations_to_plot, replace=False)
                allSegsColor = ['grey' for _ in range(output_data['len_dendritic_voltage_vecs'])]
                count_plotted = 0
                for color_index, k in enumerate(sorted_according_to_colors):
                    if k in locations:
                        plot_index = count_plotted * (sorted_according_to_colors.shape[0] // count_locations_to_plot)
                        # ax2.plot(recording_time_high_res, 150+k*delta_voltage+recordings['dendritic_voltages_high_res'][k,:].T, c=colors[k], alpha=0.5)
                        ax2.plot(recording_time_high_res, add_from_bottom+plot_index*delta_voltage+recordings['dendritic_voltages_high_res'][k,:].T, c=colors[k], alpha=0.5)

                        print(f'plotting {k} at {plot_index} with color {colors[k]}')
                        min_voltage = recordings['dendritic_voltages_high_res'][k,:].min()
                        max_voltage = recordings['dendritic_voltages_high_res'][k,:].max()
                        print(f'min_voltage={min_voltage}, max_voltage={max_voltage}')
                        avg_voltage = recordings['dendritic_voltages_high_res'][k,:].mean()
                        std_voltage = recordings['dendritic_voltages_high_res'][k,:].std()
                        print(f'avg_voltage={avg_voltage}, std_voltage={std_voltage}')

                        # plot from avg-std to avg+std, vertically
                        ax2.plot([-100, -100], [add_from_bottom+plot_index*delta_voltage+avg_voltage-std_voltage, add_from_bottom+plot_index*delta_voltage+avg_voltage+std_voltage], c=colors[k], lw=2)
                        # scatter the avg
                        ax2.scatter(-100, add_from_bottom+plot_index*delta_voltage+avg_voltage, color=colors[k], s=10)
                        # annotate the avg
                        ax2.text(-80, add_from_bottom+plot_index*delta_voltage+avg_voltage, f'{avg_voltage:.3f}', fontsize=8, color=colors[k])
                        # # annotate the avg+std, avg-std
                        # ax2.text(-80, add_from_bottom+plot_index*delta_voltage+avg_voltage+std_voltage, f'{avg_voltage+std_voltage:.3f}', fontsize=10, color=colors[k])
                        # ax2.text(-80, add_from_bottom+plot_index*delta_voltage+avg_voltage-std_voltage, f'{avg_voltage-std_voltage:.3f}', fontsize=10, color=colors[k])

                        # plot from min to max, vertically
                        ax2.plot([-100, -100], [add_from_bottom+plot_index*delta_voltage+min_voltage, add_from_bottom+plot_index*delta_voltage+max_voltage], c=colors[k], lw=1)

                        # scatter the min and max
                        ax2.scatter(-100, add_from_bottom+plot_index*delta_voltage+min_voltage, color=colors[k], s=10)
                        ax2.scatter(-100, add_from_bottom+plot_index*delta_voltage+max_voltage, color=colors[k], s=10)
                        # annotate the min and max
                        ax2.text(-80, add_from_bottom+plot_index*delta_voltage+min_voltage, f'{min_voltage:.3f}', fontsize=8, color=colors[k])
                        ax2.text(-80, add_from_bottom+plot_index*delta_voltage+max_voltage, f'{max_voltage:.3f}', fontsize=8, color=colors[k])
                        
                        # # plot a voltage scale, from 0 to scale_top mV
                        # scale_top = 30
                        # ax2.plot([-200, -200], [add_from_bottom+plot_index*delta_voltage, add_from_bottom+plot_index*delta_voltage+scale_top], c='black', lw=2.4)
                        
                        # # annotate the scale (the 0 and the 50)
                        # ax2.text(-180, add_from_bottom+plot_index*delta_voltage, f'0mV', fontsize=10, color=colors[k])
                        # ax2.text(-180, add_from_bottom+plot_index*delta_voltage+scale_top, f'{scale_top}mV', fontsize=10, color=colors[k])

                        # annotate the segment index
                        ax2.text(-350, add_from_bottom//2+plot_index*delta_voltage, f'{k}', fontsize=20, color=colors[k])
                        allSegsColor[k] = colors[k]
                        count_plotted += 1

                ax2.plot(recording_time_high_res, recordings['somatic_voltage_high_res'], c='black', lw=2.4)

                plotter.plot_shape(ax=ax1, allSegsColor=allSegsColor)
                ax1.set_axis_off()
                for location in locations:
                    seg_x, seg_y = plotter.get_seg_coord(location)
                    cur_average_seg_y = seg_y.mean()
                    cur_average_seg_x = seg_x.mean()
                    ax1.text(cur_average_seg_x, cur_average_seg_y, f'{location}', fontsize=20, color=allSegsColor[location])
                    ax1.scatter(cur_average_seg_x, cur_average_seg_y, color=allSegsColor[location], s=100)

                fig.savefig(f'{args.simulation_folder}/dendritic_voltage_{count_locations_to_plot}_random_locations_{random_plot_index}.png')
                plt.close('all')
            

    run_simulation_duration_in_seconds = time.time() - run_simulation_start_time
    logger.info(f"run simulation finished!, it took {run_simulation_duration_in_seconds/60.0:.3f} minutes")

    if args.finish_file:
        with open(args.finish_file, 'w') as f:
            f.write('finished')
    
    return run_simulation_duration_in_seconds

def get_simulation_args():
    saver = ArgumentSaver()
    saver.add_argument('--simulation_duration_in_seconds', default=10, type=int)
    saver.add_argument('--random_seed', default=None, type=int)

    saver.add_argument('--max_segment_length', default=None, type=float)
    saver.add_argument('--count_segments_to_stimulate', default=None, type=int)
    saver.add_argument('--force_number_of_segments', default=None, type=int)

    saver.add_argument('--use_cvode', type=str2bool, nargs='?', const=True, default=False)
    saver.add_argument('--simulation_initialization_duration_in_ms', default=500, type=int)
    saver.add_argument('--zero_padding_for_initialization', type=str2bool, nargs='?', const=True, default=False)
    saver.add_argument('--add_explicit_padding_for_initialization', type=str2bool, nargs='?', const=True, default=True)
    saver.add_argument('--count_samples_for_high_res', default=8, type=int)
    saver.add_argument('--record_dendritic_voltages', type=str2bool, nargs='?', const=True, default=False)
    saver.add_argument('--record_synaptic_traces', type=str2bool, nargs='?', const=True, default=False)
    saver.add_argument('--save_auxiliary_information', type=str2bool, nargs='?', const=True, default=False)
    saver.add_argument('--dt', default=0.025, type=float)
    saver.add_argument('--v_init', default=-76.0, type=float)
    saver.add_argument('--spike_threshold_for_computation', default=-20, type=float)
    saver.add_argument('--spike_threshold', default=-55, type=float)

    saver.add_argument('--use_rounded_weight', type=str2bool, nargs='?', const=True, default=True)
    saver.add_argument('--weight_rounding_precision', default=5, type=int)

    # number of spike ranges for the simulation
    saver.add_argument('--count_exc_spikes_per_synapse_per_100ms_range', nargs='+', type=float, default=[0, 0.1]) # up to average 1Hz
    saver.add_argument('--count_inh_spikes_per_synapse_per_100ms_range', nargs='+', type=float, default=[0, 0.1]) # up to average 1Hz
    saver.add_argument('--adaptive_inh', type=str2bool, nargs='?', const=True, default=False)
    saver.add_argument('--adaptive_inh_additive_factor_per_synapse_per_100ms_range', nargs='+', type=float, default=[-0.07, 0.03])
    
    saver.add_argument('--count_trials_for_nonzero_output_firing_rate', default=1, type=int)
    saver.add_argument('--force_multiply_count_spikes_per_synapse_per_100ms_range_by_average_segment_length', type=str2bool, nargs='?', const=True, default=False)
    
    # define inst rate between change interval and smoothing sigma options (two rules of thumb:)
    # (A) increasing sampling time interval increases firing rate (more cumulative spikes at "lucky high rate" periods)
    # (B) increasing smoothing sigma reduces output firing rate (reduce effect of "lucky high rate" periods due to averaging)
    saver.add_argument('--inst_rate_sampling_time_interval_options_ms', nargs='+', type=int, default=[25,30,35,40,45,50,55,60,65,70,75,80,85,90,100,150,200,300,450])
    saver.add_argument('--temporal_inst_rate_smoothing_sigma_options_ms', nargs='+', type=int, default=[25,30,35,40,45,50,55,60,65,80,100,150,200,250,300,400,500,600])
    saver.add_argument('--inst_rate_sampling_time_interval_jitter_range', default=20, type=int)
    saver.add_argument('--temporal_inst_rate_smoothing_sigma_jitter_range', default=20, type=int)
    saver.add_argument('--temporal_inst_rate_smoothing_sigma_mult', default=7.0, type=float)

    saver.add_argument('--spatial_multiplicative_randomness', type=str2bool, nargs='?', const=True, default=True)
    saver.add_argument('--simple_stimulation_spatial_multiplicative_randomness', type=str2bool, nargs='?', const=True, default=False)    
    saver.add_argument('--exc_spatial_multiplicative_randomness_delta_prob', default=0.85, type=float)
    saver.add_argument('--inh_spatial_multiplicative_randomness_delta_prob', default=0.85, type=float)
    saver.add_argument('--exc_spatial_multiplicative_randomness_delta_range', nargs='+', type=float, default=[0.4, 0.8])
    saver.add_argument('--inh_spatial_multiplicative_randomness_delta_range', nargs='+', type=float, default=[0.4, 0.8])
    saver.add_argument('--same_exc_inh_spatial_multiplicative_randomness_delta_prob', default=0.7, type=float)

    # synchronization
    saver.add_argument('--synchronization_prob', default=0.20, type=float)
    saver.add_argument('--exc_synchronization_profile_mult_range', nargs='+', type=float, default=[0.4, 0.8])
    saver.add_argument('--inh_synchronization_profile_mult_range', nargs='+', type=float, default=[0.4, 0.8])
    saver.add_argument('--same_exc_inh_synchronization_profile_mult_prob', default=0.6, type=float)
    saver.add_argument('--same_exc_inh_synchronization_prob', default=0.80, type=float)
    saver.add_argument('--no_exc_synchronization_prob', default=0.3, type=float)
    saver.add_argument('--no_inh_synchronization_prob', default=0.3, type=float)
    saver.add_argument('--exc_synchronization_period_range', nargs='+', type=int, default=[30, 200])
    saver.add_argument('--inh_synchronization_period_range', nargs='+', type=int, default=[30, 200])

    # remove inhibition fraction
    saver.add_argument('--remove_inhibition_prob', default=0.15, type=float)
    saver.add_argument('--remove_inhibition_exc_mult_range', nargs='+', type=float, default=[0.05, 0.3])
    saver.add_argument('--remove_inhibition_exc_mult_jitter_range', nargs='+', type=float, default=[0.3, 0.7])

    # deactivation parameters
    saver.add_argument('--deactivate_synapses_prob', default=0.1, type=float)
    saver.add_argument('--exc_deactivate_synapses_ratio_range', nargs='+', type=float, default=[0.01, 0.3])
    saver.add_argument('--inh_deactivate_synapses_ratio_range', nargs='+', type=float, default=[0.01, 0.3])
    saver.add_argument('--same_exc_inh_deactivation_count', default=0.4, type=float)
    saver.add_argument('--same_exc_inh_deactivations', default=0.3, type=float)
    saver.add_argument('--no_inh_deactivation_prob', default=0.2, type=float)
    saver.add_argument('--no_exc_deactivation_prob', default=0.2, type=float)

    # spatial clustering params
    saver.add_argument('--spatial_clustering_prob', default=0.25, type=float)
    saver.add_argument('--no_exc_spatial_clustering_prob', default=0.3, type=float)
    saver.add_argument('--no_inh_spatial_clustering_prob', default=0.3, type=float)
    saver.add_argument('--same_exc_inh_spatial_clustering_prob', default=0.7, type=float)
    saver.add_argument('--exc_spatial_cluster_size_ratio_range', nargs='+', type=float, default=[0.01, 0.1])
    saver.add_argument('--inh_spatial_cluster_size_ratio_range', nargs='+', type=float, default=[0.01, 0.1])
    saver.add_argument('--active_exc_spatial_cluster_ratio_range', nargs='+', type=float, default=[0.3, 1.0])
    saver.add_argument('--active_inh_spatial_cluster_ratio_range', nargs='+', type=float, default=[0.3, 1.0])
    saver.add_argument('--random_exc_spatial_clusters_prob', default=0.4, type=float)
    saver.add_argument('--random_inh_spatial_clusters_prob', default=0.4, type=float)

    saver.add_argument('--same_exc_inh_inst_rate_prob', default=0.02, type=float)
    saver.add_argument('--same_exc_inh_spikes_bin_prob', default=0.01, type=float)
    saver.add_argument('--same_exc_inh_spikes_bin_prob_weighted_multiply', default=5, type=float)
    saver.add_argument('--same_exc_inh_all_kernels_prob', default=0.01, type=float)
    saver.add_argument('--same_exc_inh_kernels_prob', default=0.02, type=float)

    saver.add_argument('--special_interval_transition_dur_ms', default=25, type=int)
    saver.add_argument('--special_interval_transition_dur_ms_gaussian_mult', default=7.0, type=float)
    saver.add_argument('--special_interval_transition_threshold', default=0.2, type=float)
    saver.add_argument('--count_special_intervals', default=7, type=int)
    saver.add_argument('--special_interval_high_dur_ms', default=1500, type=int)
    saver.add_argument('--special_interval_offset_ms', default=10, type=int)
    saver.add_argument('--special_interval_low_dur_ms', default=500, type=int)

    # weight generation parameters
    saver.add_argument('--exc_weights_ratio_range', nargs='+', type=float, default=[1.0, 1.0])
    saver.add_argument('--inh_weights_ratio_range', nargs='+', type=float, default=[1.0, 1.0])
    saver.add_argument('--generate_weights_using_constrained_linear', type=str2bool, nargs='?', const=True, default=True)
    
    # multiple connections parameters
    saver.add_argument('--exc_multiple_connections_upperbound', type=int, default=50)
    saver.add_argument('--inh_multiple_connections_upperbound', type=int, default=50)
    saver.add_argument('--average_exc_multiple_connections_avg_std_min', nargs='+', type=float, default=[15, 10, 1]) # old [3, 10, 1]
    saver.add_argument('--average_inh_multiple_connections_avg_std_min', nargs='+', type=float, default=[15, 10, 1]) # old [3, 10, 1]
    saver.add_argument('--multiple_connections_prob', default=0.9, type=float)
    saver.add_argument('--simple_stimulation_multiple_connections_prob', default=0.0, type=float)
    saver.add_argument('--same_exc_inh_average_multiple_connections_prob', default=0.7, type=float)

    # count of initial synapses per super synapse parameters
    saver.add_argument('--multiply_count_initial_synapses_per_super_synapse_prob', default=0.2, type=float)
    saver.add_argument('--count_exc_initial_synapses_per_super_synapse_mult_factor_range', nargs='+', type=float, default=[1, 5])
    saver.add_argument('--count_inh_initial_synapses_per_super_synapse_mult_factor_range', nargs='+', type=float, default=[1, 5])
    saver.add_argument('--same_exc_inh_count_initial_synapses_per_super_synapse_prob', default=0.7, type=float)
    saver.add_argument('--force_count_initial_synapses_per_super_synapse', default=None, type=int)
    saver.add_argument('--force_count_initial_synapses_per_tree', default=None, type=int)

    saver.add_argument('--simple_stimulation', type=str2bool, nargs='?', const=True, default=False) # a shortcut
    saver.add_argument('--default_weighted', type=str2bool, nargs='?', const=True, default=False) # a shortcut
    saver.add_argument('--wide_weighted', default=None, type=float) # a shortcut
    saver.add_argument('--wide_weighted_inh_multiplicative_factor', default=None, type=float) # a shortcut
    saver.add_argument('--wide_fr', default=None, type=float) # a shortcut
    saver.add_argument('--wide_fr_base', default=0.0, type=float) # related to wide_fr
    saver.add_argument('--wide_fr_inh_additive_factor', default=0.0, type=float) # related to wide_fr
    saver.add_argument('--wide_fr_inh_adaptive_additive_factor', nargs='+', type=float, default=None) # related to wide_fr

    return saver

def get_args():
    parser = argparse.ArgumentParser(description='Simulate a neuron')
    parser.add_argument('--neuron_model_folder')
    parser.add_argument('--simulation_folder', action=AddOutFileAction)
    parser.add_argument('--input_file', default=None)

    job_saver = get_job_args()
    job_saver.add_to_parser(parser)
    
    saver = get_simulation_args()
    saver.add_to_parser(parser)
    
    parser.add_argument('--save_plots', type=str2bool, nargs='?', const=True, default=True)
    return parser.parse_args()

def main():
    args = get_args()
    TeeAll(args.outfile)
    setup_logger(logging.getLogger())

    job_id = os.environ["SLURM_JOB_ID"] if "SLURM_JOB_ID" in os.environ else -1
    logger.info(f"Welcome to neuron simulator! running on {os.uname()} (job_id={job_id}, pid={os.getpid()}, ppid={os.getppid()})")
    run_simulation(args)
    logger.info(f"Goodbye from neuron simulator! running on {os.uname()} (job_id={job_id}, pid={os.getpid()}, ppid={os.getppid()})")

if __name__ == "__main__":
    main()
