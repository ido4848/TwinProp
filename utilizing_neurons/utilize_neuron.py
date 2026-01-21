from __future__ import print_function
import argparse
import os
import io
import pathlib
import sys
import numpy as np
from scipy.stats import norm
import pickle
import time
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import FashionMNIST, MNIST, CIFAR10, EMNIST, ImageFolder
from torchvision import transforms
from sklearn import linear_model
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn import metrics
from sklearn.metrics import confusion_matrix, explained_variance_score
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
from sklearn import linear_model
from sklearn import svm
from sklearn import neural_network

sys.path.append(str(pathlib.Path(__file__).parent.absolute()))
sys.path.append(str(pathlib.Path(__file__).parent.parent.absolute()))

from utils.slurm_job import get_job_submitter_args
from utils.utils import setup_logger, str2bool, ArgumentSaver, AddDefaultInformationAction, AddOutFileAction, TeeAll, MAXIMAL_RANDOM_SEED
from generating_data.to_spikes import GaborMethod, BinarizationMethod
from generating_data.spiking_vision_data import SpikingCatAndDog, NonSpikingSpikingCatAndDog, SpikingAfhq, NonSpikingSpikingAfhq
from generating_data.spiking_audition_data import Shd, NonSpikingShd, Ssc, NonSpikingSsc
from generating_data.abstract_data import NonSpikingAbstract, SpikingAbstract
from training_nets import fully_connected
from training_nets.neuron_nn_wrapper import NeuronNnWrapper
from training_nets.expressive_leaky_memory_neuron import ELM
from simulating_neurons.neuron_simulation_wrapper import NeuronSimulationWrapper
from utilizing_neurons.neuron_utilizer import NeuronUtilizer, DecodingType, UtilizerVerbosity, OptimizerType
from utilizing_neurons.utilize_neuron_args import get_utilize_neuron_args

import logging
logger = logging.getLogger(__name__)

def run_sklearn_model(model, ds, args, model_name="logistic_regression", ds_extra_label_information=False):
    run_sklearn_model_start_time = time.time()

    # get train data
    X_train, y_train = next(iter(ds.get_train_loader(batch_size=ds.train_size())))

    if ds_extra_label_information:
        print(y_train)
        print(y_train[0])
        print(y_train.shape)
        y_train = y_train[:, 0]

    X_train = X_train.reshape(X_train.shape[0], -1)
    y_train = np.argmax(y_train, axis=1)
    
    print("X_train", X_train.shape)
    print("y_train", y_train.shape)

    # fit on the train data using sklearn interface
    model.fit(X_train, y_train)

    # get valid data
    X_valid, y_valid = next(iter(ds.get_valid_loader(batch_size=ds.valid_size())))

    if ds_extra_label_information:
        y_valid = y_valid[:, 0]

    X_valid = X_valid.reshape(X_valid.shape[0], -1)
    y_valid = np.argmax(y_valid, axis=1)

    print("X_valid", X_valid.shape)
    print("y_valid", y_valid.shape)

    # calculate accuracy on valid_dataset, using sklearn interface
    valid_accuracy = model.score(X_valid, y_valid)

    # TODO: calculate that too
    valid_auc = -1
    valid_mae = -1

    logger.info(f"sklearn {model_name} valid accuracy: {valid_accuracy}")

    results = {}

    results["args"] = args
    results["valid_accuracy"] = valid_accuracy
    results["last_valid_accuracy"] = valid_accuracy
    results["valid_auc"] = valid_auc
    results["last_valid_auc"] = valid_auc
    results["valid_mae"] = valid_mae
    results["last_valid_mae"] = valid_mae
    results["first_n_epochs_best_k_models"] = {}
    results["every_n_epochs_best_k_models"] = {}
    results["valid_acc_best_k_models"] = {}
    results["valid_auc_best_k_models"] = {}
    results["valid_mae_best_k_models"] = {}
    
    pickle.dump(results, open(f'{args.utilize_neuron_folder}/final_results.pkl', 'wb'), protocol=-1)            

    run_sklearn_model_duration_in_seconds = time.time() - run_sklearn_model_start_time
    logger.info(f"utilize model finished!, it took {run_sklearn_model_duration_in_seconds/60.0:.3f} minutes")

    if args.finish_file:
        with open(args.finish_file, 'w') as f:
            f.write('finished')

    return run_sklearn_model_duration_in_seconds

class PlaceHolderNn(nn.Module):
    def __init__(self):
        super(PlaceHolderNn, self).__init__()
    def forward(self, x):
        return x

def utilize_neuron_for_task(args):
    logger.info("Going to run utilize model for task with args:")
    logger.info("{}".format(args))
    logger.info("...")

    os.makedirs(args.utilize_neuron_folder, exist_ok=True)
    logs_folder = f'{args.utilize_neuron_folder}/logs'
    os.makedirs(logs_folder, exist_ok=True)

    if args.save_plots:
        plots_folder = f'{args.utilize_neuron_folder}/plots_folder'
        os.makedirs(plots_folder, exist_ok=True)

    pickle.dump(args, open(f'{args.utilize_neuron_folder}/args.pkl','wb'), protocol=-1)

    random_seed = args.random_seed
    if random_seed is None:
        random_seed = int(time.time())
    logger.info(f"seeding with random_seed={random_seed}")
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    # TODO: make it work with python 3.11
    # pl.utilities.seed.seed_everything(random_seed)

    utilize_neuron_name = args.utilize_neuron_name
    if not utilize_neuron_name:
        utilize_neuron_name = os.path.basename(args.utilize_neuron_folder)

    utilize_neuron_start_time = time.time()

    loss_weights = None

    transform_labels = None
    if args.label_x_vs_all is not None:
        transform_labels = lambda x: 1 if x == args.label_x_vs_all else 0
        loss_weights = torch.tensor([1.0, args.label_x_vs_all_label_x_weight])

    if args.odd_labels_vs_even_labels:
        transform_labels = lambda x: 1 if x % 2 == 0 else 0
        # TODO: implement that?
        # loss_weights = torch.tensor([1.0, args.odd_labels_vs_even_labels_even_labels_weight])

    if args.ssc_odd_number_labels_vs_all_other_labels:
        ssc_odd_number_labels = [21, 27, 7, 23, 17]
        transform_labels = lambda x: 1 if x in ssc_odd_number_labels else 0
        # TODO: implement that?
        # loss_weights = torch.tensor([1.0, args.ssc_odd_number_labels_vs_all_other_labels_weight])
    
    keep_labels = None
    if args.keep_labels:
        keep_labels = [int(x) for x in args.keep_labels.split(",")]

    split_seed = args.split_seed
    if split_seed is None:
        split_seed = np.random.randint(0, MAXIMAL_RANDOM_SEED + 1)
    logger.info(f"split_seed={split_seed}")

    args.effective_background_firing_rate_duration_from_start = args.background_firing_rate_duration_from_start
    if args.effective_background_firing_rate_duration_from_start < 0:
        args.effective_background_firing_rate_duration_from_start = args.stimulus_duration_in_ms
    ds_extra_label_information = args.extra_label_information

    if args.average_stimulus_burst_firing_rate_per_axon is None:
        args.average_stimulus_burst_firing_rate_per_axon = args.average_stimulus_firing_rate_per_axon

    
    gabor_method = GaborMethod.NONE
    if args.gabor_method is not None:
        if args.gabor_method == 0:
            gabor_method = GaborMethod.NONE
        elif args.gabor_method == 1:
            gabor_method = GaborMethod.GABOR_S1
        elif args.gabor_method == 2:
            gabor_method = GaborMethod.GABOR_C1
        elif args.gabor_method == 3:
            gabor_method = GaborMethod.GABOR_S2
        elif args.gabor_method == 4:
            gabor_method = GaborMethod.GABOR_C2
        else:
            raise Exception(f"Invalid gabor method {args.gabor_method}")

    binarization_method = BinarizationMethod.THRESHOLD
    if args.binarization_method is not None:
        if args.binarization_method == 0:
            binarization_method = BinarizationMethod.NONE
        elif args.binarization_method == 1:
            binarization_method = BinarizationMethod.THRESHOLD
        elif args.binarization_method == 2:
            binarization_method = BinarizationMethod.THRESHOLD_MEAN
        elif args.binarization_method == 3:
            binarization_method = BinarizationMethod.BUCKETIZE_POISSON
        elif args.binarization_method == 4:
            binarization_method = BinarizationMethod.BUCKETIZE_TEMPORAL     
        elif args.binarization_method == 5:
            binarization_method = BinarizationMethod.BUCKETIZE_RAW
        elif args.binarization_method == 6:
            binarization_method = BinarizationMethod.THRESHOLD_POISSON
        elif args.binarization_method == 7:
            binarization_method = BinarizationMethod.THRESHOLD_TEMPORAL
        elif args.binarization_method == 8:
            binarization_method = BinarizationMethod.THRESHOLD_MEAN_POISSON
        elif args.binarization_method == 9:
            binarization_method = BinarizationMethod.THRESHOLD_MEAN_TEMPORAL
        else:
            raise Exception(f"Invalid binarization method {args.binarization_method}") 

    abstract_valid_jitters = None
    if args.abstract_valid_jitters_start is not None and args.abstract_valid_jitters_end is not None and args.abstract_valid_jitters_step is not None:
        abstract_valid_jitters = np.arange(args.abstract_valid_jitters_start, args.abstract_valid_jitters_end, args.abstract_valid_jitters_step)

    # ds selection
    if args.abstract:
        ds = NonSpikingAbstract(valid_percentage=args.valid_percentage, 
        use_raw_patterns=args.abstract_use_raw_patterns, jitter=args.abstract_jitter, valid_jitter=args.abstract_valid_jitter, valid_jitters=abstract_valid_jitters, extra_label_information=ds_extra_label_information,
         input_encoding=args.abstract_input_encoding,
         count_values_per_dimension=args.abstract_count_values_per_dimension,
        count_dimensions=args.abstract_count_dimensions, samples_per_pattern=args.abstract_samples_per_pattern,
         N_e=args.count_exc_axons, N_i=args.count_inh_axons, stim_dur=args.stimulus_duration_in_ms,
          average_firing_rate_per_axon=args.average_stimulus_firing_rate_per_axon,
            r_0=args.average_background_firing_rate_per_axon, t_off = args.effective_background_firing_rate_duration_from_start,
            N_e_bias=args.count_exc_bias_axons, N_i_bias=args.count_inh_bias_axons,
          shuffle_inds=args.abstract_shuffle_inds,
            mutually_exclusive_synapses_for_each_pattern=args.abstract_mutually_exclusive_synapses_for_each_pattern,
            arange_spikes=args.arange_spikes, random_labels=args.abstract_random_labels,
            random_permutation_labels=args.abstract_random_permutation_labels, num_t=args.abstract_num_t, trial=args.abstract_trial)
        logger.info(f"Using non-spiking {ds.dim_values_name} abstract dataset with samples_per_pattern={args.abstract_samples_per_pattern} and background_firing_rate={args.average_background_firing_rate_per_axon}")
    elif args.spiking_abstract:
        ds = SpikingAbstract(valid_percentage=args.valid_percentage, 
        use_raw_patterns=args.abstract_use_raw_patterns, jitter=args.abstract_jitter, valid_jitter=args.abstract_valid_jitter, valid_jitters=abstract_valid_jitters, extra_label_information=ds_extra_label_information,
         input_encoding=args.abstract_input_encoding,
         count_values_per_dimension=args.abstract_count_values_per_dimension,
        count_dimensions=args.abstract_count_dimensions, samples_per_pattern=args.abstract_samples_per_pattern,
         N_e=args.count_exc_axons, N_i=args.count_inh_axons, stim_dur=args.stimulus_duration_in_ms, 
         average_firing_rate_per_axon=args.average_stimulus_firing_rate_per_axon, 
         r_0=args.average_background_firing_rate_per_axon, t_off = args.effective_background_firing_rate_duration_from_start,
         N_e_bias=args.count_exc_bias_axons, N_i_bias=args.count_inh_bias_axons,
         shuffle_inds=args.abstract_shuffle_inds,
           mutually_exclusive_synapses_for_each_pattern=args.abstract_mutually_exclusive_synapses_for_each_pattern,
           arange_spikes=args.arange_spikes, random_labels=args.abstract_random_labels,
           random_permutation_labels=args.abstract_random_permutation_labels, num_t=args.abstract_num_t, trial=args.abstract_trial)
        logger.info(f"Using spiking {ds.dim_values_name} abstract dataset with samples_per_pattern={args.abstract_samples_per_pattern} and background_firing_rate={args.average_background_firing_rate_per_axon}")
    elif args.spiking_cat_and_dog:
        # TODO: support count_exc_axons and count_inh_axons
        if args.count_exc_axons != args.count_inh_axons:
            raise Exception("count_exc_axons must be equal to count_inh_axons for spiking cat and dog")
        count_axons = args.count_exc_axons + args.count_inh_axons

        # TODO: support count_exc_bias_axons and count_inh_bias_axons
        if args.count_exc_bias_axons != args.count_inh_bias_axons:
            raise Exception("count_const_firing_axons must be equal to count_exc_bias_axons + count_inh_bias_axons for non spiking spiking cat and dog")
        count_const_firing_axons = args.count_exc_bias_axons + args.count_inh_bias_axons

        ds = SpikingCatAndDog(extra_label_information=ds_extra_label_information, keep_labels=keep_labels, valid_percentage=args.valid_percentage, 
        transform_labels=transform_labels, time_in_ms=args.stimulus_duration_in_ms,
          count_axons=count_axons, count_const_firing_axons=count_const_firing_axons, average_firing_rate_per_axon=args.average_stimulus_firing_rate_per_axon,
          bg_activity_hz=args.average_background_firing_rate_per_axon,
         split_seed=split_seed, gabor_method=gabor_method, binarization_method=binarization_method,
          max_count_samples=args.max_count_samples, to_presampled=args.presampled,
           average_burst_firing_rate_per_axon=args.average_stimulus_burst_firing_rate_per_axon, jitter=args.spike_jitter,
           arange_spikes=args.arange_spikes, temporal_delay_from_start=args.temporal_delay_from_start,
             gabor_c1_only_unit_0=args.gabor_c1_only_unit_0, subsample_modulo=args.subsample_modulo, initial_image_size=args.initial_image_size)
    elif args.non_spiking_spiking_cat_and_dog:
        # TODO: support count_exc_axons and count_inh_axons
        if args.count_exc_axons != args.count_inh_axons:
            raise Exception("count_exc_axons must be equal to count_inh_axons for non spiking spiking cat and dog")
        count_axons = args.count_exc_axons + args.count_inh_axons

        # TODO: support count_exc_bias_axons and count_inh_bias_axons
        if args.count_exc_bias_axons != args.count_inh_bias_axons:
            raise Exception("count_const_firing_axons must be equal to count_exc_bias_axons + count_inh_bias_axons for non spiking spiking cat and dog")
        count_const_firing_axons = args.count_exc_bias_axons + args.count_inh_bias_axons

        ds = NonSpikingSpikingCatAndDog(extra_label_information=ds_extra_label_information, keep_labels=keep_labels, valid_percentage=args.valid_percentage, 
        transform_labels=transform_labels, time_in_ms=args.stimulus_duration_in_ms,
          count_axons=count_axons, count_const_firing_axons=count_const_firing_axons, average_firing_rate_per_axon=args.average_stimulus_firing_rate_per_axon,
          bg_activity_hz=args.average_background_firing_rate_per_axon,
         split_seed=split_seed, gabor_method=gabor_method, binarization_method=binarization_method,
          max_count_samples=args.max_count_samples, to_presampled=args.presampled,
           average_burst_firing_rate_per_axon=args.average_stimulus_burst_firing_rate_per_axon, jitter=args.spike_jitter,
            arange_spikes=args.arange_spikes, temporal_delay_from_start=args.temporal_delay_from_start,
              gabor_c1_only_unit_0=args.gabor_c1_only_unit_0, subsample_modulo=args.subsample_modulo, initial_image_size=args.initial_image_size)
    elif args.spiking_afhq:
        # TODO: support count_exc_axons and count_inh_axons
        if args.count_exc_axons != args.count_inh_axons:
            raise Exception("count_exc_axons must be equal to count_inh_axons for spiking afhq")
        count_axons = args.count_exc_axons + args.count_inh_axons

        # TODO: support count_exc_bias_axons and count_inh_bias_axons
        if args.count_exc_bias_axons != args.count_inh_bias_axons:
            raise Exception("count_const_firing_axons must be equal to count_exc_bias_axons + count_inh_bias_axons for spiking afhq")
        count_const_firing_axons = args.count_exc_bias_axons + args.count_inh_bias_axons

        ds = SpikingAfhq(extra_label_information=ds_extra_label_information, keep_labels=keep_labels, valid_percentage=args.valid_percentage, 
        transform_labels=transform_labels, time_in_ms=args.stimulus_duration_in_ms,
          count_axons=count_axons, count_const_firing_axons=count_const_firing_axons, average_firing_rate_per_axon=args.average_stimulus_firing_rate_per_axon,
          bg_activity_hz=args.average_background_firing_rate_per_axon,
         split_seed=split_seed, gabor_method=gabor_method, binarization_method=binarization_method,
          max_count_samples=args.max_count_samples, to_presampled=args.presampled,
           average_burst_firing_rate_per_axon=args.average_stimulus_burst_firing_rate_per_axon, jitter=args.spike_jitter,
           arange_spikes=args.arange_spikes, temporal_delay_from_start=args.temporal_delay_from_start,
             gabor_c1_only_unit_0=args.gabor_c1_only_unit_0, subsample_modulo=args.subsample_modulo, initial_image_size=args.initial_image_size)
    elif args.non_spiking_spiking_afhq:
        # TODO: support count_exc_axons and count_inh_axons
        if args.count_exc_axons != args.count_inh_axons:
            raise Exception("count_exc_axons must be equal to count_inh_axons for non spiking spiking afhq")
        count_axons = args.count_exc_axons + args.count_inh_axons

        # TODO: support count_exc_bias_axons and count_inh_bias_axons
        if args.count_exc_bias_axons != args.count_inh_bias_axons:
            raise Exception("count_const_firing_axons must be equal to count_exc_bias_axons + count_inh_bias_axons for non spiking spiking afhq")
        count_const_firing_axons = args.count_exc_bias_axons + args.count_inh_bias_axons     

        ds = NonSpikingSpikingAfhq(extra_label_information=ds_extra_label_information, keep_labels=keep_labels, valid_percentage=args.valid_percentage, 
        transform_labels=transform_labels, time_in_ms=args.stimulus_duration_in_ms,
          count_axons=count_axons, count_const_firing_axons=count_const_firing_axons, average_firing_rate_per_axon=args.average_stimulus_firing_rate_per_axon,
          bg_activity_hz=args.average_background_firing_rate_per_axon,
         split_seed=split_seed, gabor_method=gabor_method, binarization_method=binarization_method,
          max_count_samples=args.max_count_samples, to_presampled=args.presampled,
           average_burst_firing_rate_per_axon=args.average_stimulus_burst_firing_rate_per_axon, jitter=args.spike_jitter,
           arange_spikes=args.arange_spikes, temporal_delay_from_start=args.temporal_delay_from_start,
             gabor_c1_only_unit_0=args.gabor_c1_only_unit_0, subsample_modulo=args.subsample_modulo, initial_image_size=args.initial_image_size)
    elif args.shd:
        # TODO: support count_exc_axons and count_inh_axons
        if args.count_exc_axons != args.count_inh_axons:
            raise Exception("count_exc_axons must be equal to count_inh_axons for shd")
        count_axons = args.count_exc_axons + args.count_inh_axons

        # TODO: support count_exc_bias_axons and count_inh_bias_axons
        if args.count_exc_bias_axons != args.count_inh_bias_axons:
            raise Exception("count_const_firing_axons must be equal to count_exc_bias_axons + count_inh_bias_axons for shd")
        count_const_firing_axons = args.count_exc_bias_axons + args.count_inh_bias_axons

        ds = Shd(extra_label_information=ds_extra_label_information, keep_labels=keep_labels, valid_percentage=args.valid_percentage, 
        transform_labels=transform_labels, time_in_ms=args.stimulus_duration_in_ms,
        count_axons=count_axons, count_const_firing_axons=count_const_firing_axons, average_firing_rate_per_axon=args.average_stimulus_firing_rate_per_axon,
          bg_activity_hz=args.average_background_firing_rate_per_axon,
         split_seed=split_seed, 
          binarization_method=binarization_method, max_count_samples=args.max_count_samples, to_presampled=args.presampled,
            average_burst_firing_rate_per_axon=args.average_stimulus_burst_firing_rate_per_axon, jitter=args.spike_jitter,
           arange_spikes=args.arange_spikes, temporal_delay_from_start=args.temporal_delay_from_start,
           reduce_fr=args.reduce_fr, temporal_adaptation=args.temporal_adaptation, detect_onsets_offsets=args.detect_onsets_offsets,
           detect_onsets_offsets_window_size=args.detect_onsets_offsets_window_size, detect_onsets_offsets_threshold=args.detect_onsets_offsets_threshold,
              detect_onsets_offsets_sustained=args.detect_onsets_offsets_sustained, detect_onsets_offsets_sustained_window_size=args.detect_onsets_offsets_sustained_window_size,
                detect_onsets_offsets_sustained_overlap=args.detect_onsets_offsets_sustained_overlap,
                detect_onsets_offsets_sustained_onset_threshold=args.detect_onsets_offsets_sustained_onset_threshold,
                detect_onsets_offsets_sustained_offset_threshold=args.detect_onsets_offsets_sustained_offset_threshold,
                envelope_extraction=args.envelope_extraction, envelope_extraction_kernel_size=args.envelope_extraction_kernel_size,
                envelope_extraction_threshold=args.envelope_extraction_threshold,
                subsample_envelope=args.subsample_envelope, subsample_envelope_time_window=args.subsample_envelope_time_window,
                subsample_envelope_axon_group=args.subsample_envelope_axon_group, subsample_envelope_statistic=args.subsample_envelope_statistic,
                binarise_subsample_envelope=args.binarise_subsample_envelope, 
                binarised_subsample_envelope_and_detect_onsets_offsets_sustained_not_binary=args.binarised_subsample_envelope_and_detect_onsets_offsets_sustained_not_binary,
                hierarchical_audio_processing=args.hierarchical_audio_processing,
                hierarchical_audio_processing2=args.hierarchical_audio_processing2,
                wav_to_binary_features=args.wav_to_binary_features,
                ds_shorter_name=args.ds_shorter_name,
           subsample_modulo=args.subsample_modulo)
    elif args.non_spiking_shd:
        # TODO: support count_exc_axons and count_inh_axons
        if args.count_exc_axons != args.count_inh_axons:
            raise Exception("count_exc_axons must be equal to count_inh_axons for non spiking shd")
        count_axons = args.count_exc_axons + args.count_inh_axons

        # TODO: support count_exc_bias_axons and count_inh_bias_axons
        if args.count_exc_bias_axons != args.count_inh_bias_axons:
            raise Exception("count_const_firing_axons must be equal to count_exc_bias_axons + count_inh_bias_axons for non spiking shd")
        count_const_firing_axons = args.count_exc_bias_axons + args.count_inh_bias_axons

        ds = NonSpikingShd(extra_label_information=ds_extra_label_information, keep_labels=keep_labels, valid_percentage=args.valid_percentage, 
        transform_labels=transform_labels, time_in_ms=args.stimulus_duration_in_ms,
        count_axons=count_axons, count_const_firing_axons=count_const_firing_axons, average_firing_rate_per_axon=args.average_stimulus_firing_rate_per_axon,
          bg_activity_hz=args.average_background_firing_rate_per_axon,
         split_seed=split_seed, 
          binarization_method=binarization_method, max_count_samples=args.max_count_samples, to_presampled=args.presampled,
          average_burst_firing_rate_per_axon=args.average_stimulus_burst_firing_rate_per_axon, jitter=args.spike_jitter,
           arange_spikes=args.arange_spikes, temporal_delay_from_start=args.temporal_delay_from_start,
           reduce_fr=args.reduce_fr, temporal_adaptation=args.temporal_adaptation, detect_onsets_offsets=args.detect_onsets_offsets,
           detect_onsets_offsets_window_size=args.detect_onsets_offsets_window_size, detect_onsets_offsets_threshold=args.detect_onsets_offsets_threshold,
           detect_onsets_offsets_sustained=args.detect_onsets_offsets_sustained, detect_onsets_offsets_sustained_window_size=args.detect_onsets_offsets_sustained_window_size,
                detect_onsets_offsets_sustained_overlap=args.detect_onsets_offsets_sustained_overlap,
                detect_onsets_offsets_sustained_onset_threshold=args.detect_onsets_offsets_sustained_onset_threshold,
                detect_onsets_offsets_sustained_offset_threshold=args.detect_onsets_offsets_sustained_offset_threshold,
                envelope_extraction=args.envelope_extraction, envelope_extraction_kernel_size=args.envelope_extraction_kernel_size,
                envelope_extraction_threshold=args.envelope_extraction_threshold,
                subsample_envelope=args.subsample_envelope, subsample_envelope_time_window=args.subsample_envelope_time_window,
                subsample_envelope_axon_group=args.subsample_envelope_axon_group, subsample_envelope_statistic=args.subsample_envelope_statistic,
                binarise_subsample_envelope=args.binarise_subsample_envelope,
                  binarised_subsample_envelope_and_detect_onsets_offsets_sustained_not_binary=args.binarised_subsample_envelope_and_detect_onsets_offsets_sustained_not_binary,
                  hierarchical_audio_processing=args.hierarchical_audio_processing,
                  hierarchical_audio_processing2=args.hierarchical_audio_processing2,
                  wav_to_binary_features=args.wav_to_binary_features,
                  ds_shorter_name=args.ds_shorter_name,
           subsample_modulo=args.subsample_modulo)
    elif args.ssc:
        # TODO: support count_exc_axons and count_inh_axons
        if args.count_exc_axons != args.count_inh_axons:
            raise Exception("count_exc_axons must be equal to count_inh_axons for ssc")
        count_axons = args.count_exc_axons + args.count_inh_axons

        # TODO: support count_exc_bias_axons and count_inh_bias_axons
        if args.count_exc_bias_axons != args.count_inh_bias_axons:
            raise Exception("count_const_firing_axons must be equal to count_exc_bias_axons + count_inh_bias_axons for ssc")
        count_const_firing_axons = args.count_exc_bias_axons + args.count_inh_bias_axons

        ds = Ssc(extra_label_information=ds_extra_label_information, keep_labels=keep_labels, valid_percentage=args.valid_percentage, 
        transform_labels=transform_labels, time_in_ms=args.stimulus_duration_in_ms,
        count_axons=count_axons, count_const_firing_axons=count_const_firing_axons, average_firing_rate_per_axon=args.average_stimulus_firing_rate_per_axon,
          bg_activity_hz=args.average_background_firing_rate_per_axon,
         split_seed=split_seed, 
          binarization_method=binarization_method, max_count_samples=args.max_count_samples, to_presampled=args.presampled,
          average_burst_firing_rate_per_axon=args.average_stimulus_burst_firing_rate_per_axon, jitter=args.spike_jitter,
           arange_spikes=args.arange_spikes, temporal_delay_from_start=args.temporal_delay_from_start,
           reduce_fr=args.reduce_fr, temporal_adaptation=args.temporal_adaptation, detect_onsets_offsets=args.detect_onsets_offsets,
           detect_onsets_offsets_window_size=args.detect_onsets_offsets_window_size, detect_onsets_offsets_threshold=args.detect_onsets_offsets_threshold,
           detect_onsets_offsets_sustained=args.detect_onsets_offsets_sustained, detect_onsets_offsets_sustained_window_size=args.detect_onsets_offsets_sustained_window_size,
                detect_onsets_offsets_sustained_overlap=args.detect_onsets_offsets_sustained_overlap,
                detect_onsets_offsets_sustained_onset_threshold=args.detect_onsets_offsets_sustained_onset_threshold,
                detect_onsets_offsets_sustained_offset_threshold=args.detect_onsets_offsets_sustained_offset_threshold,
                envelope_extraction=args.envelope_extraction, envelope_extraction_kernel_size=args.envelope_extraction_kernel_size,
                envelope_extraction_threshold=args.envelope_extraction_threshold,
                subsample_envelope=args.subsample_envelope, subsample_envelope_time_window=args.subsample_envelope_time_window,
                subsample_envelope_axon_group=args.subsample_envelope_axon_group, subsample_envelope_statistic=args.subsample_envelope_statistic,
                binarise_subsample_envelope=args.binarise_subsample_envelope, 
                binarised_subsample_envelope_and_detect_onsets_offsets_sustained_not_binary=args.binarised_subsample_envelope_and_detect_onsets_offsets_sustained_not_binary,
                hierarchical_audio_processing=args.hierarchical_audio_processing,
                hierarchical_audio_processing2=args.hierarchical_audio_processing2,
                wav_to_binary_features=args.wav_to_binary_features,
                ds_shorter_name=args.ds_shorter_name,
           subsample_modulo=args.subsample_modulo)
    elif args.non_spiking_ssc:
        # TODO: support count_exc_axons and count_inh_axons
        if args.count_exc_axons != args.count_inh_axons:
            raise Exception("count_exc_axons must be equal to count_inh_axons for non spiking ssc")
        count_axons = args.count_exc_axons + args.count_inh_axons

        # TODO: support count_exc_bias_axons and count_inh_bias_axons
        if args.count_exc_bias_axons != args.count_inh_bias_axons:
            raise Exception("count_const_firing_axons must be equal to count_exc_bias_axons + count_inh_bias_axons for non spiking ssc")
        count_const_firing_axons = args.count_exc_bias_axons + args.count_inh_bias_axons

        ds = NonSpikingSsc(extra_label_information=ds_extra_label_information, keep_labels=keep_labels, valid_percentage=args.valid_percentage, 
        transform_labels=transform_labels, time_in_ms=args.stimulus_duration_in_ms,
        count_axons=count_axons, count_const_firing_axons=count_const_firing_axons, average_firing_rate_per_axon=args.average_stimulus_firing_rate_per_axon,
          bg_activity_hz=args.average_background_firing_rate_per_axon,
         split_seed=split_seed, 
          binarization_method=binarization_method, max_count_samples=args.max_count_samples, to_presampled=args.presampled,
          average_burst_firing_rate_per_axon=args.average_stimulus_burst_firing_rate_per_axon, jitter=args.spike_jitter,
           arange_spikes=args.arange_spikes, temporal_delay_from_start=args.temporal_delay_from_start,
           reduce_fr=args.reduce_fr, temporal_adaptation=args.temporal_adaptation, detect_onsets_offsets=args.detect_onsets_offsets,
           detect_onsets_offsets_window_size=args.detect_onsets_offsets_window_size, detect_onsets_offsets_threshold=args.detect_onsets_offsets_threshold,
           detect_onsets_offsets_sustained=args.detect_onsets_offsets_sustained, detect_onsets_offsets_sustained_window_size=args.detect_onsets_offsets_sustained_window_size,
                detect_onsets_offsets_sustained_overlap=args.detect_onsets_offsets_sustained_overlap,
                detect_onsets_offsets_sustained_onset_threshold=args.detect_onsets_offsets_sustained_onset_threshold,
                detect_onsets_offsets_sustained_offset_threshold=args.detect_onsets_offsets_sustained_offset_threshold,
                envelope_extraction=args.envelope_extraction, envelope_extraction_kernel_size=args.envelope_extraction_kernel_size,
                envelope_extraction_threshold=args.envelope_extraction_threshold,
                subsample_envelope=args.subsample_envelope, subsample_envelope_time_window=args.subsample_envelope_time_window,
                subsample_envelope_axon_group=args.subsample_envelope_axon_group, subsample_envelope_statistic=args.subsample_envelope_statistic,
                binarise_subsample_envelope=args.binarise_subsample_envelope,
                  binarised_subsample_envelope_and_detect_onsets_offsets_sustained_not_binary=args.binarised_subsample_envelope_and_detect_onsets_offsets_sustained_not_binary,
                  hierarchical_audio_processing=args.hierarchical_audio_processing,
                  hierarchical_audio_processing2=args.hierarchical_audio_processing2,
                  wav_to_binary_features=args.wav_to_binary_features,
                  ds_shorter_name=args.ds_shorter_name,
           subsample_modulo=args.subsample_modulo)

    else:
        raise Exception("No dataset selected")

    ds_shape = ds.get_ds_shape()

    # model selection

    # first check if we have an nn model for a neuron / model
    if args.neuron_nn_file is not None:
        model = NeuronNnWrapper(args.neuron_nn_file, threshold=args.neuron_nn_threshold)
        logger.info(f"Using neuron nn model from file {args.neuron_nn_file} with threshold {args.neuron_nn_threshold}")


    elif args.sklearn_logistic_regression:
        model = linear_model.LogisticRegression(max_iter=args.count_epochs)
        logger.info(f"Using sklearn logistic regression model")
        # this is a special case, bypass to a different function
        return run_sklearn_model(model, ds, args, model_name="logistic_regression", ds_extra_label_information=ds_extra_label_information)
    elif args.sklearn_svm:
        model = svm.SVC(max_iter=args.count_epochs, kernel='rbf', gamma='scale')
        logger.info(f"Using sklearn svm model")
        # this is a special case, bypass to a different function
        return run_sklearn_model(model, ds, args, model_name="svm", ds_extra_label_information=ds_extra_label_information)
    elif args.sklearn_mlp:
        model = neural_network.MLPClassifier(max_iter=args.count_epochs, hidden_layer_sizes=(args.sklearn_mlp_hidden_layer_size,), activation='relu', solver='adam')
        logger.info(f"Using sklearn mlp model")
        # this is a special case, bypass to a different function
        return run_sklearn_model(model, ds, args, model_name="mlp", ds_extra_label_information=ds_extra_label_information)
    elif args.fully_connected:
        model = fully_connected.FullyConnected(ds_shape[0], ds_shape[2], bias=args.fully_connected_bias,
         count_hidden_layers=args.fully_connected_count_hidden_layers, hidden_layer_size=args.fully_connected_hidden_layer_size,
         batch_norm=args.fully_connected_batch_norm)
        logger.info(f"Using fully connected model")
    
    # then check if we have a model that can be run from a simulation
    elif args.neuron_model_folder is not None:
        neuron_simulation_wrapper_folder = f'{args.utilize_neuron_folder}/neuron_simulation_wrapper_folder'
        os.makedirs(neuron_simulation_wrapper_folder, exist_ok=True)
        model = NeuronSimulationWrapper(args.neuron_model_folder, neuron_simulation_wrapper_folder, args=args, nseg=args.neuron_model_nseg, max_segment_length=args.neuron_model_max_segment_length)
        logger.info(f"Using neuron simulation wrapper model from folder {args.neuron_model_folder}")

    # then check if we have a model from a checkpoint
    elif args.model_from_checkpoint is not None:
        utilizer = NeuronUtilizer.load_from_checkpoint(args.model_from_checkpoint)
        model = utilizer.model
        logger.info(f"Using model from checkpoint {args.model_from_checkpoint}")
    else:
        raise Exception("No model selected")

    model = model.float()
    model_shape = model.get_model_shape()

    # decoding selection
    if args.decoding_type == "max_pooling":
        decoding_type = DecodingType.MAX_POOLING
        logger.info(f"Using max pooling decoding")
    elif args.decoding_type == "sum_pooling":
        decoding_type = DecodingType.SUM_POOLING
        logger.info(f"Using sum pooling decoding")
    elif args.decoding_type == "linear_softmax":
        decoding_type = DecodingType.LINEAR_SOFTMAX
        logger.info(f"Using linear softmax decoding")
    elif args.decoding_type == "binary_linear_softmax":
        decoding_type = DecodingType.BINARY_LINEAR_SOFTMAX
        logger.info(f"Using binary linear softmax decoding")
    elif args.decoding_type == "none":
        decoding_type = DecodingType.NONE
        logger.info(f"Using no decoding")
    else:
        raise Exception("No decoding selected")

    # optimizer selection
    if args.optimizer == "adam":
        optimizer_type = OptimizerType.ADAM
        logger.info(f"Using adam optimizer with lr={args.lr}")
    elif args.optimizer == "sgd":
        optimizer_type = OptimizerType.SGD
        logger.info(f"Using sgd optimizer with lr={args.lr}, momentum={args.momentum} and weight_decay={args.weight_decay}")
    else:
        raise Exception("No optimizer selected")

    # utilizer definition
    if args.utilizer_from_checkpoint is not None:
        try:
            utilizer = NeuronUtilizer.load_from_checkpoint(args.utilizer_from_checkpoint)
        except TypeError as e:
            logger.info(f"Failed to load from checkpoint with error {e}")
            logger.info("It probably means the model wasn't saved correctly, so editing the checkpoint to hold a placeholder model, and retrying")
            
            # loading the ckpt
            edited_ckpt = torch.load(args.utilizer_from_checkpoint, map_location=torch.device('cpu'))

            # creating a placeholder model
            phn = PlaceHolderNn()
            phn_state_dict = phn.state_dict()

            # setting the model to be the placeholder model
            edited_ckpt['hyper_parameters']['model'] = phn

            # removing original model state dict entries
            delete_list = []
            for key in edited_ckpt['state_dict'].keys():
                if key.startswith('model'):
                    delete_list.append(key)
                    
            for key in delete_list:
                del edited_ckpt['state_dict'][key]

            # adding the placeholder model state dict entries
            for key in phn_state_dict.keys():
                edited_ckpt['state_dict'][f'model.{key}'] = phn_state_dict[key]

            # write edited_ckpt into a virtual bytesIO buffer
            edited_ckpt_buffer = io.BytesIO()
            torch.save(edited_ckpt, edited_ckpt_buffer)
            edited_ckpt_buffer.seek(0)

            # loading the utilizer from the buffer
            utilizer = NeuronUtilizer.load_from_checkpoint(edited_ckpt_buffer)

        utilizer.set_model(model, model_shape, disable_model_last_layer=args.disable_model_last_layer, freeze_model=args.freeze_model,
          extra_time_left_padding_if_needed=args.time_left_padding_extra_in_ms,
           time_left_padding_firing_rate=args.time_left_padding_firing_rate, time_left_padding_before_wiring=args.time_left_padding_before_wiring,)

        utilizer.enable_progress_bar = args.enable_progress_bar
        utilizer.plots_folder = plots_folder
        utilizer.enable_plotting = args.enable_plotting
        utilizer.plot_train_every_in_epochs = args.plot_train_every_in_epochs
        utilizer.plot_valid_every_in_epochs = args.plot_valid_every_in_epochs
        utilizer.plot_train_every_in_batches = args.plot_train_every_in_batches
        utilizer.plot_valid_every_in_batches = args.plot_valid_every_in_batches

    else:
        utilizer = NeuronUtilizer(model, model_shape, ds_shape, ds_extra_label_information=ds_extra_label_information,
         freeze_model=args.freeze_model, disable_model_last_layer=args.disable_model_last_layer,
         time_left_padding_firing_rate=args.time_left_padding_firing_rate, time_left_padding_before_wiring=args.time_left_padding_before_wiring,
          extra_time_left_padding_if_needed=args.time_left_padding_extra_in_ms,
          use_wiring_layer=args.use_wiring_layer, positive_wiring=args.positive_wiring, wiring_zero_smaller_than=args.wiring_zero_smaller_than,
           functional_only_wiring=args.functional_only_wiring, wiring_bias=args.wiring_bias,
            wiring_weight_init_mean=args.wiring_weight_init_mean, wiring_weight_init_bound=args.wiring_weight_init_bound,
             wiring_weight_init_sparsity=args.wiring_weight_init_sparsity, wiring_keep_max_k_from_input=args.wiring_keep_max_k_from_input, 
             wiring_keep_max_k_to_output=args.wiring_keep_max_k_to_output, wiring_keep_weight_mean=args.wiring_keep_weight_mean, 
             wiring_keep_weight_std=args.wiring_keep_weight_std, wiring_keep_weight_max=args.wiring_keep_weight_max,
             wiring_weight_l1_reg=args.wiring_weight_l1_reg, wiring_weight_l2_reg=args.wiring_weight_l2_reg,
              wiring_enforce_every_in_train_epochs=args.wiring_enforce_every_in_train_epochs,
               wiring_enforce_every_in_train_batches=args.wiring_enforce_every_in_train_batches, wiring_dales_law=args.wiring_dales_law,
               population_k=args.population_k, use_population_masking_layer=args.use_population_masking_layer,
               population_masking_bias=args.population_masking_bias, population_masking_weight_init_mean=args.population_masking_weight_init_mean,
                population_masking_weight_init_bound=args.population_masking_weight_init_bound, population_masking_weight_init_sparsity=args.population_masking_weight_init_sparsity,
                 functional_only_population_masking=args.functional_only_population_masking,
                   decoding_type=decoding_type, decoding_time_from_end=args.decoding_time_from_end,
                   require_no_spikes_before_decoding_time=args.require_no_spikes_before_decoding_time,
                  grad_abs=args.grad_abs, positive_by_sigmoid=args.positive_by_sigmoid, positive_by_softplus=args.positive_by_softplus, optimizer_type=optimizer_type, loss_weights=loss_weights, lr=args.lr, momentum=args.momentum,
                   weight_decay=args.weight_decay, step_lr=args.step_lr,
                   differentiable_binarization_threshold_surrogate_spike=args.differentiable_binarization_threshold_surrogate_spike,
                   differentiable_binarization_threshold_surrogate_spike_beta=args.differentiable_binarization_threshold_surrogate_spike_beta,
                   differentiable_binarization_threshold_straight_through=args.differentiable_binarization_threshold_straight_through,
                   enable_progress_bar=args.enable_progress_bar,
                    plots_folder=plots_folder, enable_plotting=args.enable_plotting,
                     plot_train_every_in_epochs=args.plot_train_every_in_epochs, plot_valid_every_in_epochs=args.plot_valid_every_in_epochs,
                      plot_train_every_in_batches=args.plot_train_every_in_batches,
                       plot_valid_every_in_batches=args.plot_valid_every_in_batches)

    # get data loaders
    train_loader = ds.get_train_loader(args.batch_size)
    valid_loader = ds.get_valid_loader(args.batch_size)

    if args.only_calculate_metrics:
        logger.info(f"Calculating metrics:")
        utilizer.eval()
        utilizer.verbosity = UtilizerVerbosity.NONE

        save_to_folder = None
        if args.calculate_metrics_save_to_folder:
            save_to_folder = f'{args.utilize_neuron_folder}/calculate_metrics'
            os.makedirs(save_to_folder, exist_ok=True)

        valid_metrics = utilizer.calculate_metrics(valid_loader, save_to_folder=save_to_folder)
        valid_accuracy = valid_metrics["accuracy"]
        valid_auc = valid_metrics["auc"]
        valid_mae = valid_metrics["mae"]
        logger.info(f"Valid accuracy is {valid_accuracy}")
        logger.info(f"Valid auc is {valid_auc}")
        logger.info(f"Valid mae is {valid_mae}")

        results = {}

        results["args"] = args
        results["valid_accuracy"] = valid_accuracy.detach().cpu().numpy()
        results["valid_auc"] = valid_auc.detach().cpu().numpy()
        results["valid_mae"] = valid_mae.detach().cpu().numpy()

        if abstract_valid_jitters is not None:
            results["valid_jittered_accuracies"] = {}
            results["valid_jittered_aucs"] = {}
            results["valid_jittered_maes"] = {}
            for cur_valid_jitter, cur_valid_loader in zip(abstract_valid_jitters, ds.get_valid_loaders(args.batch_size)):
                logger.info(f"Calculating metrics for jittered loader with jitter {cur_valid_jitter}")
                utilizer.eval()
                utilizer.verbosity = UtilizerVerbosity.NONE

                save_to_folder = None
                if args.calculate_metrics_save_to_folder:
                    save_to_folder = f'{args.utilize_neuron_folder}/calculate_metrics_jittered_{cur_valid_jitter}'
                    os.makedirs(save_to_folder, exist_ok=True)

                cur_valid_metrics = utilizer.calculate_metrics(cur_valid_loader, save_to_folder=save_to_folder)
                cur_valid_accuracy = cur_valid_metrics["accuracy"]
                cur_valid_auc = cur_valid_metrics["auc"]
                cur_valid_mae = cur_valid_metrics["mae"]
                logger.info(f"Valid accuracy for jitter {cur_valid_jitter} is {cur_valid_accuracy}")
                logger.info(f"Valid auc for jitter {cur_valid_jitter} is {cur_valid_auc}")
                logger.info(f"Valid mae for jitter {cur_valid_jitter} is {cur_valid_mae}")

                results["valid_jittered_accuracies"][cur_valid_jitter] = cur_valid_accuracy.detach().cpu().numpy()
                results["valid_jittered_aucs"][cur_valid_jitter] = cur_valid_auc.detach().cpu().numpy()
                results["valid_jittered_maes"][cur_valid_jitter] = cur_valid_mae.detach().cpu().numpy()
        
        pickle.dump(results, open(f'{args.utilize_neuron_folder}/final_results.pkl', 'wb'), protocol=-1)

    else:
        # dry run
        logger.info(f"Dry run:")
        utilizer.verbosity = UtilizerVerbosity.HIGH
        x, y = next(iter(train_loader))
        y_extra = None
        if ds_extra_label_information:
            y, y_extra = y
        logger.info(f"x.shape = {x.shape}, y.shape = {y.shape}")
        out = utilizer(x)[1]
        logger.info(f"out.shape = {out.shape}")

        callbacks = []
        
        if args.enable_checkpointing:
            first_n_epochs_checkpoint_callback = ModelCheckpoint(monitor='epoch', save_top_k=args.checkpoint_first_k_epochs, mode='min',
            filename='checkpoint-{epoch}-{valid_accuracy_epoch:.5f}-{valid_auc_epoch:.5f}-{valid_mae_epoch:.5f}')
            callbacks.append(first_n_epochs_checkpoint_callback)

            every_n_epochs_checkpoint_callback = ModelCheckpoint(monitor='epoch', save_top_k=args.count_epochs, mode='max',
            every_n_epochs=args.checkpoint_every_in_epochs, filename='checkpoint-{epoch}-{valid_accuracy_epoch:.5f}-{valid_auc_epoch:.5f}-{valid_mae_epoch:.5f}', save_last=True)
            callbacks.append(every_n_epochs_checkpoint_callback)

            valid_acc_checkpoint_callback = ModelCheckpoint(monitor='valid_accuracy_epoch', save_top_k=args.checkpoint_top_k_valid_accuracies,
            mode='max', filename='checkpoint-{epoch}-{valid_accuracy_epoch:.5f}-{valid_auc_epoch:.5f}-{valid_mae_epoch:.5f}')
            callbacks.append(valid_acc_checkpoint_callback)

            valid_auc_checkpoint_callback = ModelCheckpoint(monitor='valid_auc_epoch', save_top_k=args.checkpoint_top_k_valid_aucs,
            mode='max', filename='checkpoint-{epoch}-{valid_accuracy_epoch:.5f}-{valid_auc_epoch:.5f}-{valid_mae_epoch:.5f}')
            callbacks.append(valid_auc_checkpoint_callback)

            valid_mae_checkpoint_callback = ModelCheckpoint(monitor='valid_mae_epoch', save_top_k=args.checkpoint_bottom_k_valid_maes,
            mode='min', filename='checkpoint-{epoch}-{valid_accuracy_epoch:.5f}-{valid_auc_epoch:.5f}-{valid_mae_epoch:.5f}')
            callbacks.append(valid_mae_checkpoint_callback)

        # training
        logger.info(f"Training:")
        utilizer.verbosity = UtilizerVerbosity.NONE

        if args.ut_on_gpu:
            accelerator="gpu"
        else:
            accelerator="cpu"
            
        trainer = pl.Trainer(accelerator=accelerator, max_epochs=args.count_epochs,
        default_root_dir=logs_folder, enable_progress_bar=args.enable_progress_bar,
            callbacks=callbacks)
        
        trainer.fit(utilizer, train_loader, valid_loader)

        results = {}

        results["args"] = args
        results["last_valid_accuracy"] = utilizer.last_valid_accuracy.detach().cpu().numpy()
        results["last_valid_auc"] = utilizer.last_valid_auc.detach().cpu().numpy()
        results["last_valid_mae"] = utilizer.last_valid_mae.detach().cpu().numpy()

        if args.enable_checkpointing:
            results["first_n_epochs_best_k_models"] = {k: v.detach().cpu().numpy().item()  for k,v in first_n_epochs_checkpoint_callback.best_k_models.items()}
            results["every_n_epochs_best_k_models"] = {k: v.detach().cpu().numpy().item()  for k,v in every_n_epochs_checkpoint_callback.best_k_models.items()}
            results["valid_acc_best_k_models"] = {k: v.detach().cpu().numpy() for k,v in valid_acc_checkpoint_callback.best_k_models.items()}
            results["valid_auc_best_k_models"] = {k: v.detach().cpu().numpy() for k,v in valid_auc_checkpoint_callback.best_k_models.items()}
            results["valid_mae_best_k_models"] = {k: v.detach().cpu().numpy() for k,v in valid_mae_checkpoint_callback.best_k_models.items()}

            logger.info(f"first_n_epochs_best_k_models are {first_n_epochs_checkpoint_callback.best_k_models}")
            logger.info(f"every_n_epochs_best_k_models are {every_n_epochs_checkpoint_callback.best_k_models}")
            logger.info(f"valid_acc_best_k_models are {valid_acc_checkpoint_callback.best_k_models}")
            logger.info(f"valid_auc_best_k_models are {valid_auc_checkpoint_callback.best_k_models}")
            logger.info(f"valid_mae_best_k_models are {valid_mae_checkpoint_callback.best_k_models}")
        
        pickle.dump(results, open(f'{args.utilize_neuron_folder}/final_results.pkl', 'wb'), protocol=-1)

        logger.info(f"Last valid accuracy is {utilizer.last_valid_accuracy}")
        logger.info(f"Last valid auc is {utilizer.last_valid_auc}")
        logger.info(f"Last valid mae is {utilizer.last_valid_mae}")

    utilize_neuron_duration_in_seconds = time.time() - utilize_neuron_start_time
    logger.info(f"utilize model finished!, it took {utilize_neuron_duration_in_seconds/60.0:.3f} minutes")

    if args.finish_file:
        with open(args.finish_file, 'w') as f:
            f.write('finished')

    return utilize_neuron_duration_in_seconds

def get_args():
    parser = argparse.ArgumentParser(description='Utilize model for a task')
    parser.add_argument('--utilize_neuron_folder', action=AddOutFileAction)
    parser.add_argument('--utilize_neuron_name', default=None)

    from utils.slurm_job import get_job_args
    job_saver = get_job_args()
    job_saver.add_to_parser(parser)

    get_job_submitter_args_saver = get_job_submitter_args()
    get_job_submitter_args_saver.add_to_parser(parser)

    saver = get_utilize_neuron_args()
    saver.add_to_parser(parser)
    
    parser.add_argument('--ut_on_gpu', type=str2bool, nargs='?', const=True, default=True)

    parser.add_argument('--save_plots', type=str2bool, nargs='?', const=True, default=True)
    return parser.parse_args()

def main():
    args = get_args()
    TeeAll(args.outfile)
    setup_logger(logging.getLogger())

    job_id = os.environ["SLURM_JOB_ID"] if "SLURM_JOB_ID" in os.environ else -1
    logger.info(f"Welcome to utilize model! running on {os.uname()} (job_id={job_id}, pid={os.getpid()}, ppid={os.getppid()})")
    utilize_neuron_for_task(args)
    logger.info(f"Goodbye from utilize model! running on {os.uname()} (job_id={job_id}, pid={os.getpid()}, ppid={os.getppid()})")

if __name__ == "__main__":
    main()
