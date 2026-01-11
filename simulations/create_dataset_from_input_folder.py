from __future__ import print_function
import os
import sys
import copy
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import argparse
import pathlib
import heapq

sys.path.append(str(pathlib.Path(__file__).parent.absolute()))
sys.path.append(str(pathlib.Path(__file__).parent.parent.absolute()))

from utils.utils import setup_logger, str2bool, ArgumentSaver, AddOutFileAction, TeeAll
from simulations.submit_simulate_neuron_and_create_dataset import get_generate_dataset_args, generate_dataset

import logging
logger = logging.getLogger(__name__)

def get_create_dataset_from_input_folder_args():
    saver = ArgumentSaver()

    saver.add_argument('--input_folder')
    
    dataset_saver = get_generate_dataset_args()
    dataset_saver.add_to_parser(saver)

    return saver

def get_create_dataset_from_input_folder_parser():
    parser = argparse.ArgumentParser(description='Create dataset from input folder for a neuron')
    parser.add_argument('--neuron_model_folder')
    parser.add_argument('--dataset_folder', action=AddOutFileAction)
    parser.add_argument('--dataset_name', default=None)
    
    parser.add_argument('--return_output_spike_times', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--save_plots', type=str2bool, nargs='?', const=True, default=True)

    saver = get_create_dataset_from_input_folder_args()
    saver.add_to_parser(parser)

    return parser
    
def get_args():
    parser = get_create_dataset_from_input_folder_parser()
    return parser.parse_args()

def create_dataset_from_input_folder(args):
    logger.info("Going to create dataset from input folder with args:")
    logger.info("{}".format(args))
    logger.info("...")

    random_seed = args.random_seed
    if random_seed is None:
        random_seed = int(time.time())
    logger.info(f"seeding with random_seed={random_seed}")
    np.random.seed(random_seed)

    dataset_basename = args.dataset_name
    if not dataset_basename:
        dataset_basename = os.path.basename(args.dataset_folder)

    os.makedirs(args.dataset_folder, exist_ok=True)
    generate_dataset_args = copy.deepcopy(args)
    generate_dataset_args.simulation_dataset_folder = os.path.join(args.dataset_folder, 'simulation_dataset')
    generate_dataset_args.simulation_dataset_name = f"{dataset_basename}"

    train_input_folder = f"{args.input_folder}/train"
    create_from_train = False
    if os.path.exists(train_input_folder):
        input_dir_numbers = np.sort([int(dir) for dir in os.listdir(train_input_folder) if os.path.isdir(os.path.join(train_input_folder, dir))])
        if len(input_dir_numbers) > 0:
            create_from_train = True
            generate_dataset_args.train_input_folder = train_input_folder
            generate_dataset_args.count_simulations_for_train = input_dir_numbers[-1] + 1
    
    create_from_valid = False
    valid_input_folder = f"{args.input_folder}/valid"
    if os.path.exists(valid_input_folder):
        input_dir_numbers = np.sort([int(dir) for dir in os.listdir(valid_input_folder) if os.path.isdir(os.path.join(valid_input_folder, dir))])
        if len(input_dir_numbers) > 0:
            create_from_valid = True
            generate_dataset_args.valid_input_folder = valid_input_folder
            generate_dataset_args.count_simulations_for_valid = input_dir_numbers[-1] + 1

    create_from_test = False
    test_input_folder = f"{args.input_folder}/test"
    if os.path.exists(test_input_folder):
        input_dir_numbers = np.sort([int(dir) for dir in os.listdir(test_input_folder) if os.path.isdir(os.path.join(test_input_folder, dir))])
        if len(input_dir_numbers) > 0:
            create_from_test = True
            generate_dataset_args.test_input_folder = test_input_folder
            generate_dataset_args.count_simulations_for_test = input_dir_numbers[-1] + 1
            
    if args.return_output_spike_times:
        generate_dataset_args.summarize_output_spike_times = True
    create_dataset_total_duration_in_seconds = generate_dataset(generate_dataset_args)

    # TODO?
    # generate_dataset_args.simulation_job_timelimit = 60 * 15

    output_spike_times = {}
    if args.return_output_spike_times:
        if create_from_train:
            train_summary = pickle.load(open(f'{generate_dataset_args.simulation_dataset_folder}/train/summary.pkl','rb'))
            output_spike_times_information = train_summary['output_spike_times_information']
            output_spike_times['train'] = output_spike_times_information

        if create_from_valid:
            valid_summary = pickle.load(open(f'{generate_dataset_args.simulation_dataset_folder}/valid/summary.pkl','rb'))
            output_spike_times_information = valid_summary['output_spike_times_information']
            output_spike_times['valid'] = output_spike_times_information

        if create_from_test:
            test_summary = pickle.load(open(f'{generate_dataset_args.simulation_dataset_folder}/test/summary.pkl','rb'))
            output_spike_times_information = test_summary['output_spike_times_information']
            output_spike_times['test'] = output_spike_times_information

    pickle.dump({'args':args, 'output_spike_times': output_spike_times}, open(f'{args.dataset_folder}/summary.pkl', 'wb'), protocol=-1)

    logger.info(f"create dataset from input folder finished!, it took {create_dataset_total_duration_in_seconds/60.0:.3f} minutes")

    return output_spike_times, create_dataset_total_duration_in_seconds

def main():
    args = get_args()
    TeeAll(args.outfile)
    setup_logger(logging.getLogger())
    
    job_id = os.environ["SLURM_JOB_ID"] if "SLURM_JOB_ID" in os.environ else -1
    logger.info(f"Welcome to create dataset from input folder for neuron! running on {os.uname()} (job_id={job_id}, pid={os.getpid()}, ppid={os.getppid()})")
    create_dataset_from_input_folder(args)
    logger.info(f"Goodbye from create dataset from input folder for neuron! running on {os.uname()} (job_id={job_id}, pid={os.getpid()}, ppid={os.getppid()})")

if __name__ == "__main__":
    main()
