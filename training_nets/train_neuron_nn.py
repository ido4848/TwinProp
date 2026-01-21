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
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import wandb

sys.path.append(str(pathlib.Path(__file__).parent.parent.absolute()))
sys.path.append(str(pathlib.Path(__file__).parent.absolute()))

from utils.roc_utils import window_roc_curve
from utils.utils import setup_logger, str2bool, ArgumentSaver, AddDefaultInformationAction, AddOutFileAction, TeeAll
from training_nets.expressive_leaky_memory_neuron import ELM


logger = logging.getLogger(__name__)

INFINITE_VOLTAGE_CLIP = 9e9

class FocalLossWithLogitsLoss(nn.Module):    
    def __init__(self, gamma=1.0, alpha=0.25, pos_weight=None):
        super(FocalLossWithLogitsLoss, self).__init__()
        self.register_buffer('gamma', torch.tensor(gamma))
        self.register_buffer('alpha', torch.tensor(alpha))
        self.register_buffer('pos_weight', pos_weight)

    def forward(self, input, target):
        bce_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none', pos_weight=self.pos_weight)
        pt = torch.exp(-bce_loss) # prevents nans when probability 0
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()

def calc_AUC_at_desired_FP(desired_FPs, roc_data):
    fpr, tpr, thresholds = roc_data
    auc_for_FP = {}
    for desired_false_positive_rate in desired_FPs:
        linear_spaced_FPR = np.linspace(0,1,num=20000)
        linear_spaced_TPR = np.interp(linear_spaced_FPR, fpr, tpr)

        desired_fp_ind = min(max(1,np.argmin(abs(linear_spaced_FPR-desired_false_positive_rate))),linear_spaced_TPR.shape[0]-1)
        
        auc_for_FP[desired_false_positive_rate] = linear_spaced_TPR[:desired_fp_ind].mean()
    return auc_for_FP

def calc_TP_at_desired_FP(desired_FPs, roc_data):
    fpr, tpr, thresholds = roc_data
    TP_at_desired_FP = {}
    for desired_false_positive_rate in desired_FPs:
        desired_fp_ind = np.argmin(abs(fpr-desired_false_positive_rate))
        if desired_fp_ind == 0:
            desired_fp_ind = 1
        TP_at_desired_FP[desired_false_positive_rate] = tpr[desired_fp_ind]
    return TP_at_desired_FP

def calc_precision_and_recall(y_true, y_pred):
    ret = confusion_matrix(y_true, y_pred).ravel()

    if len(ret) < 4:
        return 0.0, 0.0
    tn, fp, fn, tp = ret
    if tp+fp == 0:
        ret1 = 0.0
    else:
        ret1 = tp/(tp+fp)
    if tp+fn == 0:
        ret2 = 0.0
    else:
        ret2 = tp/(tp+fn)

    return ret1, ret2

def calc_PR_at_input_fr(y_true, y_pred, roc_data):
    num_of_spikes = sum(y_true>0)
    num_of_no_spikes = sum(y_true<1)
    fpr, tpr, thresholds = roc_data # Not sure this one is working correclty
    ind = np.argmin(abs(np.array([tpr[i]*num_of_spikes + fpr[i]*num_of_no_spikes for i in range(len(fpr))])-num_of_spikes))
    return calc_precision_and_recall(y_true, y_pred>thresholds[ind])

def calc_PR_at_plot_FP(y_true, y_pred, roc_data, args):
    fpr, tpr, thresholds = roc_data # Not sure this one is working correclty
    fp_ind_for_plot = np.argmin(abs(fpr-args.false_positive_rate_for_plot))
    if fp_ind_for_plot == 0:
        fp_ind_for_plot = 1
    false_positive_rate_for_plot = fpr[fp_ind_for_plot]
    true_positive_rate_for_plot = tpr[fp_ind_for_plot]

    spike_threshold_for_plot = thresholds[fp_ind_for_plot]

    return calc_precision_and_recall(y_true, y_pred>spike_threshold_for_plot)

def write_more_metrics(y_true, y_pred, v_true, v_pred, fpr, tpr, thresholds, desired_FPs, writer, steps, case, default_threshold, wandb_init, args):
    '''
    case = 'train or test'
    '''
    TP_at_desired_FP  = calc_TP_at_desired_FP(desired_FPs, roc_data = [fpr, tpr, thresholds])
    auc_for_FP = calc_AUC_at_desired_FP(desired_FPs, roc_data = [fpr, tpr, thresholds])
    precision_at_input_fr, recall_at_input_fr = calc_PR_at_input_fr(y_true, y_pred, roc_data = [fpr, tpr, thresholds])
    precision_at_default_threshold, recall_at_default_threshold = calc_precision_and_recall(y_true, y_pred > default_threshold)
    precision_at_plot_FP, recall_at_plot_FP = calc_PR_at_plot_FP(y_true, y_pred, roc_data = [fpr, tpr, thresholds], args = args)
    
    wandb_log = {}

    for FP_rate in TP_at_desired_FP:
        writer.add_scalar(f'TP_at/{FP_rate}FP/{case}', TP_at_desired_FP[FP_rate], steps)
        wandb_log[f'TP_at/{FP_rate}FP/{case}'] = TP_at_desired_FP[FP_rate]
        writer.add_scalar(f'auc_for/{FP_rate}FP/{case}', auc_for_FP[FP_rate], steps)
        wandb_log[f'auc_for/{FP_rate}FP/{case}'] = auc_for_FP[FP_rate]


    writer.add_scalar(f'precision_at_input_fr/{case}', precision_at_input_fr, steps)
    wandb_log[f'precision_at_input_fr/{case}'] = precision_at_input_fr
    writer.add_scalar(f'recall_at_input_fr/{case}', recall_at_input_fr, steps)
    wandb_log[f'recall_at_input_fr/{case}'] = recall_at_input_fr
    writer.add_scalar(f'precision_at_default_threshold/{case}', precision_at_default_threshold, steps)
    wandb_log[f'precision_at_default_threshold/{case}'] = precision_at_default_threshold
    writer.add_scalar(f'recall_at_default_threshold/{case}', recall_at_default_threshold, steps)
    wandb_log[f'recall_at_default_threshold/{case}'] = recall_at_default_threshold
    writer.add_scalar(f'precision_at_plot_FP/{case}', precision_at_plot_FP, steps)
    wandb_log[f'precision_at_plot_FP/{case}'] = precision_at_plot_FP
    writer.add_scalar(f'recall_at_plot_FP/{case}', recall_at_plot_FP, steps)
    wandb_log[f'recall_at_plot_FP/{case}'] = recall_at_plot_FP

    if wandb_init:
        wandb.log(wandb_log, step=steps)

    return auc_for_FP, TP_at_desired_FP

def get_device(args):
    if torch.cuda.is_available() and args.run_on_gpu:
        device = torch.device('cuda')
        number_of_devices = torch.cuda.device_count()
        logger.info(f"Using {number_of_devices} GPU devices:")
        for i in range(number_of_devices):
            gpu_name = torch.cuda.get_device_name(i)
            logger.info(f"GPU device {i} is {gpu_name}")
        return device
    else:
        device = torch.device('cpu')
        cpu_name = platform.processor()
        logger.info(f"Using CPU device {cpu_name}")
        return device
    
class SimulationData(Dataset):
    def __init__(self, base_directory, window_size=700, v_clip=INFINITE_VOLTAGE_CLIP, v_offset=0,\
         start_t=500, overlap_size=None, normalized_test=False, test_name='test', remove_zero_simulations=True):
        super(SimulationData).__init__()

        self.v_clip = v_clip
        self.window_size = window_size
        self.base_directory = base_directory
        self.v_offset = v_offset
        self.start_t = start_t

        self.dataset_summary = pickle.load(open(f'{self.base_directory}/summary.pkl','rb'))

        bigger_than = 0 if remove_zero_simulations else -1
        self.simulation_indices = [k for k, v in self.dataset_summary['firing_rate_information']['by_index'].items() if v > bigger_than]
        self.avg_firing_rate = np.mean([self.dataset_summary['firing_rate_information']['by_index'][k] for k in self.simulation_indices])

        self.normalized_test = False
        if normalized_test:
            if 'firing_rate_information' in self.dataset_summary and f'normalized_{test_name}_set_by_index' in self.dataset_summary['firing_rate_information']:
                self.normalized_test = True
                self.simulation_indices = list(self.dataset_summary['firing_rate_information'][f'normalized_{test_name}_set_by_index'].keys())
                if f'normalized_{test_name}_average_firing_rate' in self.dataset_summary['firing_rate_information']:
                    self.avg_firing_rate = self.dataset_summary['firing_rate_information'][f'normalized_{test_name}_average_firing_rate']
            else:
                raise ValueError(f"Can't use normalized test set, no 'normalized_{test_name}_set_by_index' data in 'firing_rate_information'")      
                # logger.info(f"No 'normalized_{test_name}_set_by_index' data in 'firing_rate_information', normalized {test_name} will be regular {test_name}")

        self.count_simulations = len(self.simulation_indices)

        if isinstance(self.v_offset, tuple):
            if 'average_somatic_voltage_information' in self.dataset_summary:
                if self.v_clip < INFINITE_VOLTAGE_CLIP and 'average_clipped_somatic_voltage' in self.dataset_summary['average_somatic_voltage_information']:
                    self.v_offset = self.dataset_summary['average_somatic_voltage_information']['average_clipped_somatic_voltage']
                elif self.v_clip >= INFINITE_VOLTAGE_CLIP and 'average_somatic_voltage' in self.dataset_summary['average_somatic_voltage_information']:
                    self.v_offset = self.dataset_summary['average_somatic_voltage_information']['average_somatic_voltage']
                else:
                    self.v_offset = self.v_offset[0]    
            else:
                self.v_offset = self.v_offset[0]

        # test one file
        example_simulation = f'{self.base_directory}/simulation_0'
        voltage = h5py.File(f'{example_simulation}/voltage.h5', 'r')
        summary = pickle.load(open(f'{example_simulation}/summary.pkl','rb'))
        simulation_duration_in_ms = summary['simulation_duration_in_ms']
        example_simulation_spike_count = len(np.nonzero(summary['output_spike_times'])[0])
        logger.info(f'{example_simulation} has {example_simulation_spike_count} spikes')

        self.simulation_duration_in_seconds = self.dataset_summary['args'].simulation_duration_in_seconds
        self.simulation_duration_in_ms = self.dataset_summary['args'].simulation_initialization_duration_in_ms + self.simulation_duration_in_seconds * 1000

        self.overlap_size = int(self.window_size / 2) if overlap_size is None else overlap_size

        # remove the first window from the length and check how many overlays fit
        self.num_per_sim = int((self.simulation_duration_in_ms-start_t-window_size)/(window_size-self.overlap_size))+1

        logger.info(f"SimulationData({base_directory}) [normalized_test={self.normalized_test}] has {self.count_simulations*self.num_per_sim} samples and {(self.count_simulations*(self.simulation_duration_in_ms-start_t))/1000/60/60:.3f} hours of data with average firing rate {self.avg_firing_rate}")
        
    def __len__(self):
        return self.count_simulations * self.num_per_sim
        
    def __getitem__(self, idx, debug=False):
        simulation_n_orig = int(idx/self.num_per_sim)
        simulation_n = self.simulation_indices[simulation_n_orig]

        st_pos = self.start_t + (self.window_size-self.overlap_size)*(idx%self.num_per_sim) 
        en_pos = st_pos + self.window_size

        sim_folder = f"{self.base_directory}/simulation_{simulation_n}"

        if os.path.exists(f'{sim_folder}/exc_weighted_spikes.npz'):
            exc_weighted_spikes = sparse.load_npz(f'{self.base_directory}/simulation_{simulation_n}/exc_weighted_spikes.npz').A
            inh_weighted_spikes = sparse.load_npz(f'{self.base_directory}/simulation_{simulation_n}/inh_weighted_spikes.npz').A

            exc_weighted_spikes_for_window = exc_weighted_spikes[:,st_pos:en_pos]
            inh_weighted_spikes_for_window = inh_weighted_spikes[:,st_pos:en_pos]
                
            all_weighted_spikes_for_window = np.vstack((exc_weighted_spikes_for_window, inh_weighted_spikes_for_window))

            exc_weighted_spikes = None
            del exc_weighted_spikes
            inh_weighted_spikes = None
            del inh_weighted_spikes

        else:
            all_weighted_spikes = sparse.load_npz(f'{self.base_directory}/simulation_{simulation_n}/all_weighted_spikes.npz').A
            all_weighted_spikes_for_window = all_weighted_spikes[:,st_pos:en_pos]

            all_weighted_spikes = None
            del all_weighted_spikes

        # get voltage
        somatic_voltage = h5py.File(f'{self.base_directory}/simulation_{simulation_n}/voltage.h5','r')['somatic_voltage']

        somatic_voltage_for_window = somatic_voltage[st_pos:en_pos]
        somatic_voltage_for_window[somatic_voltage_for_window>self.v_clip] = self.v_clip
        somatic_voltage_for_window = somatic_voltage_for_window - self.v_offset

        somatic_voltage = None
        del somatic_voltage

        # get output spike times
        summary = pickle.load(open(f'{self.base_directory}/simulation_{simulation_n}/summary.pkl','rb'))

        output_spikes_for_window = np.zeros(self.window_size)
        spike_times = summary['output_spike_times'][(summary['output_spike_times']>=st_pos) & (summary['output_spike_times']<en_pos)]
        output_spikes_for_window[spike_times.astype(int)-st_pos] = 1

        r_item = {'sps_in':all_weighted_spikes_for_window,
                  'somatic_voltage_out':somatic_voltage_for_window,
                  'sps_out':output_spikes_for_window}
        return r_item

class CausalConv1d(torch.nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        self.__padding = (kernel_size - 1) * dilation
        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            groups=groups,
            bias=bias)

    def forward(self, input):
        result = super(CausalConv1d, self).forward(input)
        
        
        if self.__padding != 0:
            return result[:, :, :-self.__padding]
        return result

def load_saved_models(model, optimizer, scheduler, output_path, map_device, continue_training_from=None):
    loading_path = output_path
    model_steps = np.sort([int(i.split('_')[2]) for i in os.listdir(output_path) if 'model' in i and 'pkl' not in i])
    continue_training = False

    if len(model_steps) == 0 and continue_training_from and os.path.isdir(continue_training_from):
        logger.info(f"Going to continue from directory {continue_training_from}")
        model_steps = np.sort([int(i.split('_')[2]) for i in os.listdir(continue_training_from) if 'model' in i])
        loading_path = continue_training_from
        continue_training = True

    if len(model_steps) == 0:
        if continue_training_from and not os.path.isdir(continue_training_from):
            logger.info(f"Going to continue from file {continue_training_from}")
            model_file = continue_training_from
            continue_training = True
        else:
            return model, optimizer, scheduler, 0, 0, 0, wandb.util.generate_id(), {"normalized_auc": -1, "auc": -1},\
                 {"train": {}, "test": {}, "normalized_test": {}, "valid": {}, "normalized_valid": {}}
    else:
        try:
            model_file = glob.glob(f'{loading_path}/model_*_{model_steps[-1]}')[0]
        except Exception as e:
            logger.error(f"model_steps: {model_steps}")
            raise e

    logger.info(f"Going to load model {model_file}:")

    checkpoint = torch.load(model_file, map_location=map_device)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    step = checkpoint['step']
    total_training_time_in_seconds = checkpoint['total_training_time_in_seconds']
    wandb_id = checkpoint['wandb_id']
    best_save = checkpoint['best_save']
    eval_history = checkpoint['eval_history']

    logger.info(f"model of epoch {epoch} step {step} loaded successfully!")

    if continue_training:
        logger.info("on continue training, resetting wandb_id")
        return model, optimizer, scheduler, epoch + 1, step, total_training_time_in_seconds, wandb.util.generate_id(), best_save, eval_history
    else:
        return model, optimizer, scheduler, epoch + 1, step, total_training_time_in_seconds, wandb_id, best_save, eval_history

def compute_taylor_consistency_loss(model_fwd, x, delta_scale=0.1, sparsity_fraction=0.5,
                            spikes_loss_weight_tc=1.0, soma_loss_weight_tc=0.3, 
                            near_spike_loss_weight_tc=0.2, inst_rate_loss_weight_tc=0.1, dend_v_loss_weight_tc=0.1,
                            create_graph=True, x_is_dict=True):
    
    if x_is_dict:
        x_dict = x
        x = x_dict['sps_in']
                            
    # create the perturbation
    dx = delta_scale * torch.randn_like(x)
    dx = dx * (torch.rand_like(x) < sparsity_fraction)
    x_dx = x + dx

    # forward pass through the model with the perturbed input

    dx_dict = copy.deepcopy(x_dict)
    dx_dict['sps_in'] = dx

    x_dx_dict = copy.deepcopy(x_dict)
    x_dx_dict['sps_in'] = x_dx

    # y_dy_soma_model, y_dy_spikes_model = model_fwd(x_dx_dict)
    y_dy_soma_model, y_dy_spikes_model = model_fwd(x_dx)


    # compute and unpack the jvp (Jacobian-vector product) J=dy/dx, <J,v> = dy/dx * dx = dy
    # NOTE: when create_graph is True, to minimize the loss, model weight changes will also be made in order to 
    # dircetly change the f'(x) of the model to match the models finite difference approximation (f(x+dx) - f(x)) / dx
    # meaning, there is a double backward step here. the derivative of the derivative of the model with resppect to loss
    # when create_graph is False, the weights will only try to change f(x) so that it matches the empirical "detached" f'(x)
    y_jvp, dy_jvp = torch.autograd.functional.jvp(model_fwd, x, dx, create_graph=create_graph)
    # y_jvp, dy_jvp = torch.autograd.functional.jvp(model_fwd, x_dict, dx_dict, create_graph=create_graph)
    y_jvp_soma, y_jvp_spikes = y_jvp
    dy_jvp_soma, dy_jvp_spikes = dy_jvp

    # compute the taylor approximation: f(x+dx) ≈ f(x) + ∇f(x)·dx
    y_dy_spikes_jvp = y_jvp_spikes + dy_jvp_spikes
    y_dy_soma_jvp = y_jvp_soma + dy_jvp_soma

    # compute the loss (minimize the difference between the model output and the taylor approximation)
    loss = spikes_loss_weight_tc * F.mse_loss(y_dy_spikes_model, y_dy_spikes_jvp)
    loss += soma_loss_weight_tc * F.mse_loss(y_dy_soma_model, y_dy_soma_jvp)

    return loss

def train_neuron_nn_on_data(args):
    logger.info("Going to run train neuron nn with args:")
    logger.info("{}".format(args))
    logger.info("...")

    os.makedirs(args.neuron_nn_folder, exist_ok=True)
    logs_folder = f'{args.neuron_nn_folder}/logs'
    models_folder = f'{args.neuron_nn_folder}/models'
    os.makedirs(logs_folder, exist_ok=True)
    os.makedirs(models_folder, exist_ok=True)

    if args.save_plots:
        plots_folder = f'{args.neuron_nn_folder}/plots_folder'
        os.makedirs(plots_folder, exist_ok=True)

    pickle.dump(args, open(f'{args.neuron_nn_folder}/args.pkl','wb'), protocol=-1)

    random_seed = args.random_seed
    if random_seed is None:
        random_seed = int(time.time())
    logger.info(f"seeding with random_seed={random_seed}")
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    train_neuron_nn_start_time = time.time()

    device = get_device(args)

    writer = SummaryWriter(logs_folder)

    if args.use_elm:
        network_temporal_extent = 150
    else:
        raise NotImplementedError("Only ELM model is supported in train_neuron_nn.py")

    test_name = 'test'
    if args.use_valid_as_test:
        test_name = 'valid'

    v_clip = args.spike_threshold if args.clip_somatic_voltage_to_spike_threshold else INFINITE_VOLTAGE_CLIP
    train_data = SimulationData(f'{args.simulation_dataset_folder}/train', window_size=args.window_size, overlap_size=network_temporal_extent, v_offset=(args.default_v_offset,), v_clip=v_clip, start_t=args.simulation_initialization_duration_in_ms)
    used_v_offset = train_data.v_offset

    test_data = SimulationData(f'{args.simulation_dataset_folder}/{test_name}', window_size=args.window_size, overlap_size=network_temporal_extent, v_offset=used_v_offset, v_clip=v_clip, start_t=args.simulation_initialization_duration_in_ms, test_name=test_name)
    normalized_test_data = SimulationData(f'{args.simulation_dataset_folder}/{test_name}', window_size=args.window_size, overlap_size=network_temporal_extent, v_offset=used_v_offset, v_clip=v_clip, start_t=args.simulation_initialization_duration_in_ms, normalized_test=True, test_name=test_name)

    train_data_loader = DataLoader(train_data, batch_size=args.batch_size, num_workers=args.count_train_workers, shuffle=True)
    test_data_loader = DataLoader(test_data, batch_size=args.batch_size, num_workers=args.count_test_workers, shuffle=False)
    normalized_test_data_loader = DataLoader(normalized_test_data, batch_size=args.batch_size, num_workers=args.count_normalized_test_workers, shuffle=False)

    in_chans = train_data[0]['sps_in'].shape[0]

    if args.use_elm:
        model_config = dict()
        model_config["input_to_synapse_routing"] = "neuronio_routing"
        model_config["learn_memory_tau"] = args.elm_learn_memory_tau
        model_config["memory_tau_max"] = args.elm_memory_tau_max
        model_config["memory_tau_min"] = args.elm_memory_tau_min
        model_config["mlp_num_layers"] = args.elm_mlp_num_layers
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
        raise NotImplementedError("Only ELM model is supported in train_neuron_nn.py")

    logger.info(f"model created: {repr(model)}")

    # Loss
    v_loss = torch.nn.MSELoss()

    if args.use_pos_weight:
        # we have train_data.avg_firing_rate spikes for every 1000 - train_data.avg_firing_rate non spikes (train_data.avg_firing_rate Hz firing rate)
        pos_weight = 1000.0 - train_data.avg_firing_rate / (train_data.avg_firing_rate) if train_data.avg_firing_rate > 0 else 1000.0
        pos_weight = torch.Tensor([pos_weight])
    else:
        pos_weight = None
    
    if args.use_focal_loss:
        sp_loss = FocalLossWithLogitsLoss(alpha=args.fl_alpha, gamma=args.fl_gamma, pos_weight=pos_weight)
    else:
        sp_loss = FocalLossWithLogitsLoss(alpha=1.0, gamma=0.0, pos_weight=pos_weight)

    # data parallelism
    if args.use_data_parallel:
        model = nn.DataParallel(model)

    # send model to device
    model.to(device)
    v_loss.to(device)
    sp_loss.to(device)

    if args.use_nadam:
        optimizer = optim.NAdam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.use_adamw:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    using_cosine_annealing_lr_scheduler = False
    using_step_lr_scheduler = False
    using_reduce_lr_on_plateau_scheduler = False
    if args.use_cosine_annealing_lr:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.count_epochs * len(train_data_loader))
        using_cosine_annealing_lr_scheduler = True
    elif args.use_step_lr:
        scheduler = StepLR(optimizer, step_size=args.step_lr_step_size, gamma=args.step_lr_gamma)
        using_step_lr_scheduler = True
    else:
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=args.reduce_lr_on_plateau_factor, patience=args.reduce_lr_on_plateau_patience)
        using_reduce_lr_on_plateau_scheduler = True

    # load saved model
    model, optimizer, scheduler, epoch, steps, total_training_time_in_seconds, wandb_id, best_save, eval_history = load_saved_models(model, optimizer, scheduler, models_folder, map_device=device, continue_training_from=args.continue_training_from)

    model.double()

    wandb_init = False
    if args.reset_lr: optimizer.param_groups[0]['lr'] = args.lr
    
    last_train_step = steps
    current_train_start_time = time.time()
    count_tests = 0
    
    if args.count_epochs < 0:
        max_epoch = epoch + abs(args.count_epochs)
    else:
        max_epoch = args.count_epochs

    neuron_nn_name = args.neuron_nn_name
    if not neuron_nn_name:
        neuron_nn_name = os.path.basename(args.neuron_nn_folder)
        if neuron_nn_name.startswith("neuron_nn"):
            dir_name = os.path.basename(os.path.dirname(args.neuron_nn_folder))
            neuron_nn_name = f'{dir_name}_{neuron_nn_name}'

    simulation_dataset_name = os.path.basename(args.simulation_dataset_folder)

    def compute_losses_and_collect_results(out_v, out_sp, data, y_preds, y_trues, v_preds, v_trues, loss_lists, state="train", data_index=None, model_fwd=None):
        # computing loss only when meaningfull, i.e. after network_temporal_extent ms
        sp_pred = out_sp[:,network_temporal_extent:,:].squeeze()
        sp_gt = data['sps_out'][:,network_temporal_extent:]
        v_pred = out_v[:,network_temporal_extent:,:].squeeze()
        v_gt = data['somatic_voltage_out'][:,network_temporal_extent:]

        y_preds.extend(list(torch.sigmoid(sp_pred).flatten().detach().cpu().numpy()))
        y_trues.extend(list(sp_gt.flatten().numpy()))
        v_preds.extend(list(v_pred.flatten().detach().cpu().numpy()))
        v_trues.extend(list(v_gt.flatten().numpy()))

        v_pred = v_pred.double()
        v_gt = v_gt.double()
        sp_pred = sp_pred.double()
        sp_gt = sp_gt.double()

        loss_v = v_loss(v_pred, v_gt.to(device))
        loss_sps = sp_loss(sp_pred, sp_gt.to(device))
        both_loss = loss_sps + loss_v*args.v_loss_weight

        full_loss = both_loss

        if args.use_taylor_consistency_loss:
            if data_index is None or data_index % args.taylor_consistency_loss_every_in_samples == 0:
                create_graph = state == "train"
                taylor_consistency_loss = compute_taylor_consistency_loss(model_fwd, data, create_graph=create_graph, x_is_dict=True)

                full_loss = full_loss + args.taylor_consistency_loss_weight * taylor_consistency_loss
        
        loss_lists['full'].append(float(full_loss))
        loss_lists['both'].append(float(both_loss))
        loss_lists['sps'].append(float(loss_sps))
        loss_lists['rmse'].append(float(loss_v))

        return full_loss, y_preds, y_trues, v_preds, v_trues, loss_lists

    def log_and_write_and_plot(ep, y_preds, y_trues, v_preds, v_trues, loss_lists, steps, total_training_time_in_seconds, case):
        y_preds, y_trues, v_preds, v_trues = np.array(y_preds), np.array(y_trues), np.array(v_preds), np.array(v_trues)

        fpr, tpr, thresholds = metrics.roc_curve(y_trues, y_preds)
        cur_auc = metrics.auc(fpr, tpr)

        _, cur_auc_95ci = confidenceinterval.roc_auc_score(y_trues,
                                y_preds,
                                confidence_level=0.95)
        cur_auc_95ci = np.array(cur_auc_95ci)

        triplet_fpr, triplet_tpr, triplet_thresholds = window_roc_curve(y_trues, y_preds, window_size=3, prediction_round=3)
        cur_triplet_auc = metrics.auc(triplet_fpr, triplet_tpr)

        prc_precision, prc_recall, prc_thresholds = metrics.precision_recall_curve(y_trues, y_preds)
        cur_prc_auc = metrics.auc(prc_recall, prc_precision)

        cur_aps = metrics.average_precision_score(y_trues, y_preds)
        cur_loss = np.mean(loss_lists["both"])

        soma_explained_variance_percent = 100.0*explained_variance_score(v_trues,v_preds)
        soma_RMSE = np.sqrt(MSE(v_trues,v_preds))
        soma_MAE  = MAE(v_trues, v_preds)     

        logger.info(f"{case} results on epoch {ep} and after {steps} steps:")
        logger.info(f'loss = {cur_loss:.5f}')
        logger.info(f'auc = {cur_auc:.5f}')
        logger.info(f'auc 95% CI = {cur_auc_95ci}')
        logger.info(f'triplet auc = {cur_triplet_auc:.5f}')
        logger.info(f'prc_auc = {cur_prc_auc:.5f}')
        logger.info(f'aps = {cur_aps:.5f}')
        logger.info(f'somatic voltage explained variance = {soma_explained_variance_percent:.5f}%')

        wandb_log = {}

        for k in loss_lists:
            writer.add_scalar(f'{k}/{case}', np.mean(loss_lists[k]), steps)
            wandb_log[f'{k}/{case}'] = np.mean(loss_lists[k])
        writer.add_scalar(f'auc/{case}', cur_auc, steps)
        wandb_log[f'auc/{case}'] = cur_auc
        writer.add_scalar(f'auc_95ci_low/{case}', cur_auc_95ci[0], steps)
        writer.add_scalar(f'auc_95ci_high/{case}', cur_auc_95ci[1], steps)
        wandb_log[f'auc_95ci/{case}'] = cur_auc_95ci
        writer.add_scalar(f'triplet_auc/{case}', cur_triplet_auc, steps)
        wandb_log[f'triplet_auc/{case}'] = cur_triplet_auc
        writer.add_scalar(f'prc_auc/{case}', cur_prc_auc, steps)
        wandb_log[f'prc_auc/{case}'] = cur_prc_auc
        writer.add_scalar(f'aps/{case}', cur_aps, steps)
        wandb_log[f'aps/{case}'] = cur_aps
        writer.add_scalar(f'soma_explained_var/{case}', soma_explained_variance_percent, steps)
        wandb_log[f'soma_explained_var/{case}'] = soma_explained_variance_percent
        writer.add_scalar(f'soma_RMSE/{case}', soma_RMSE, steps)
        wandb_log[f'soma_RMSE/{case}'] = soma_RMSE
        writer.add_scalar(f'soma_MAE/{case}', soma_MAE, steps)
        wandb_log[f'soma_MAE/{case}'] = soma_MAE

        cur_lr = optimizer.param_groups[0]["lr"]

        writer.add_scalar('lr', cur_lr , steps)
        wandb_log['lr'] = cur_lr
        writer.add_scalar('epoch', ep, steps)
        wandb_log['epoch'] = ep

        if case not in eval_history:
            eval_history[case] = {}

        if len(eval_history[case]) > 0:
            last_eval = eval_history[case][np.sort(list(eval_history[case].keys()))[-1]]
            last_lr = last_eval['lr']
            last_lr_changes = last_eval['lr_changes']
            if np.abs(cur_lr - last_lr) >= args.lr_change_threshold:
                lr_changes = last_lr_changes + 1
            else:
                lr_changes = last_lr_changes
        else: 
            lr_changes = 0

        eval_history[case][steps] = {'ep': ep, 'total_training_time_in_seconds':total_training_time_in_seconds,\
             'auc': cur_auc, 'auc_95ci': cur_auc_95ci, 'triplet_auc': cur_triplet_auc, 'prc_auc': cur_prc_auc, 'aps': cur_aps, 'loss': cur_loss, 'lr': cur_lr,\
             'lr_changes': lr_changes, 'soma_explained_variance_percent': soma_explained_variance_percent, 'soma_RMSE': soma_RMSE, 'soma_MAE': soma_MAE}

        if wandb_init:
            wandb.log(wandb_log, step=steps)

        auc_for_FP, TP_at_desired_FP = write_more_metrics(y_trues, y_preds, v_trues, v_preds, fpr, tpr, thresholds, args.desired_FP_list, writer, steps, case=case, default_threshold=args.default_threshold, wandb_init=wandb_init, args=args)

        if args.save_plots:
            threshold_jumps = int(len(thresholds) / args.count_annotations) + 1
            plt.plot(fpr, tpr, label=f'ROC curve (area = {cur_auc:.5f})')
            for x, y, txt in zip(fpr[::threshold_jumps], tpr[::threshold_jumps], thresholds[::threshold_jumps]):
                plt.annotate(np.round(txt,2), (x, y-0.04))
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC curve for {case} epoch {ep} after {steps} steps')
            plt.legend()
            plt.savefig(f'{plots_folder}/roc_{case}_{ep}_{steps}.png')
            plt.close('all')
    
            fp_ind_for_plot = np.argmin(abs(fpr-args.false_positive_rate_for_plot))
            if fp_ind_for_plot == 0:
                fp_ind_for_plot = 1
            false_positive_rate_for_plot = fpr[fp_ind_for_plot]
            true_positive_rate_for_plot = tpr[fp_ind_for_plot]

            spike_threshold_for_plot = thresholds[fp_ind_for_plot]

            fig = plt.figure(figsize=(40, 20))
            axs = fig.subplots(4, 5)
            plotted_intervals = []
            for i in range(4):
                for j in range(5):
                    t_begin = None
                    overlapping = True
                    while overlapping:
                        overlapping = False
                        t_begin = np.random.randint(len(v_trues))
                        for interval in plotted_intervals:
                            if interval[0] <= t_begin and t_begin <= interval[1]:
                                overlapping = True
                                break
                        if not overlapping:
                            # require some spikes in the window
                            overlapping = len(np.nonzero(y_trues[t_begin:t_begin+args.plot_window_size])[0]) == 0 and len(np.nonzero(y_preds[t_begin:t_begin+args.plot_window_size] > spike_threshold_for_plot)[0]) == 0

                    gt_somatic_voltage_to_plot = v_trues[t_begin:t_begin+args.plot_window_size] + used_v_offset
                    pred_somatic_voltage_to_plot = v_preds[t_begin:t_begin+args.plot_window_size] + used_v_offset

                    gt_spikes = y_trues[t_begin:t_begin+args.plot_window_size]
                    pred_spikes = y_preds[t_begin:t_begin+args.plot_window_size] > spike_threshold_for_plot

                    output_spike_times_in_ms_gt   = np.nonzero(gt_spikes)[0]
                    output_spike_times_in_ms_pred = np.nonzero(pred_spikes)[0]

                    if args.clip_somatic_voltage_to_spike_threshold:
                        gt_somatic_voltage_to_plot[output_spike_times_in_ms_gt] = args.spike_voltage_value_for_plot
                        pred_somatic_voltage_to_plot[output_spike_times_in_ms_pred] = args.spike_voltage_value_for_plot

                    axs[i][j].plot(gt_somatic_voltage_to_plot, '-b', label='Somatic voltage')
                    axs[i][j].plot(pred_somatic_voltage_to_plot, ':r', label='Predicted somatic voltage')
                    axs[i][j].set_xlabel('Time (ms)')
                    axs[i][j].set_ylabel('Somatic voltage (mV)')
                    axs[i][j].set_title(f'[{t_begin},{t_begin+args.plot_window_size}]')
                    axs[i][j].legend()
            fig.suptitle(f'Predicted somatic voltage ({soma_explained_variance_percent:.5f}% explained) for {case} epoch {ep} after {steps} steps (threshold={spike_threshold_for_plot:.3f}, tpr={true_positive_rate_for_plot:.3f}, fpr={false_positive_rate_for_plot:.3f})')
            plt.savefig(f'{plots_folder}/predicted_somatic_voltage_{case}_{ep}_{steps}.png')
            plt.close('all')

        results = {} 
        results["case"] = case
        results["args"] = args
        results["neuron_nn"] = neuron_nn_name
        results["neuron_nn_folder"] = args.neuron_nn_folder
        results["simulation_dataset"] = simulation_dataset_name
        results["simulation_dataset_folder"] = args.simulation_dataset_folder

        results["depth"] = args.depth
        results["width"] = args.width
        results["first_kernel_size"] = args.first_kernel_size
        results["later_kernels_size"] = args.later_kernels_size
        results["leaky_relu_alpha"] = args.leaky_relu_alpha
        results["dropout"] = args.dropout
        results["refractory_period_threshold"] = args.refractory_period_threshold
        results["count_connections_to_skip"] = args.count_connections_to_skip
        results["is_fcn"] = args.later_kernels_size == 1
        results["network_temporal_extent"] = network_temporal_extent
        results["count_train_samples"] = len(train_data_loader)

        results["ep"] = ep
        results["steps"] = steps
        results["loss"] = cur_loss
        results["auc"] = cur_auc
        results["triplet_auc"] = cur_triplet_auc
        results["prc_auc"] = cur_prc_auc
        results["aps"] = cur_aps
        results["lr"] = cur_lr
        results["spikes_d_prime"] = np.sqrt(2) * norm.ppf(results['auc'])
        results["somatic_voltage_explained_variance"] = soma_explained_variance_percent
        results["somatic_voltage_RMSE"] = soma_RMSE
        results["soma_RMSE"] = soma_RMSE
        results["somatic_voltage_MAE"] = soma_MAE
        results["soma_MAE"] = soma_MAE
        results["auc_for_FP"] = auc_for_FP
        results["TP_at_desired_FP"] = TP_at_desired_FP

        return results

    kwargs = vars(args)
    kwargs["dataset"] = os.path.basename(simulation_dataset_name)

    try:
        wandb.init(project="neuron_as_deep_net_2", config=kwargs, dir=args.neuron_nn_folder, id=wandb_id, name=neuron_nn_name, resume="allow")
        wandb_init = True
    except Exception as e:
        logger.info(f"wandb init failed with [{e}], not using wandb.")

    logger.info('Starting to train model!')

    writer.add_scalar('train_files', train_data_loader.dataset.count_simulations, steps)
    if wandb_init:
        wandb.log({'train_files': train_data_loader.dataset.count_simulations}, step=steps)

    finish = False

    ep = epoch

    def model_forward(cur_data, self_reliant=False, data_is_dict=True):
        if data_is_dict:
            cur_input = cur_data['sps_in']
        else:
            cur_input = cur_data
        if args.use_elm:
            cur_input = cur_input.transpose(1,2)

        cur_input = cur_input.double()

        outputs = model(cur_input.to(device))

        out_v = outputs[..., 1].unsqueeze(-1)
        out_sp = outputs[..., 0].unsqueeze(-1)

        return out_v, out_sp

    using_self_reliant_train = not args.train_not_self_reliant

    for ep in range(epoch, max_epoch):
        loss_lists = {'full':[], 'both':[], 'sps':[], 'rmse':[]}
        y_preds,y_trues,v_preds,v_trues = [], [], [], []
        count_steps_since_last_show = 0
        writer.add_scalar('epoch', ep, steps)
        if wandb_init:
            wandb.log({"epoch": ep}, step=steps)

        logger.info("In train mode now.")
        model.train()
        optimizer.zero_grad()

        ################## Train ################
        for data_index, data in enumerate(train_data_loader):
            avg_sps_in_sample = (torch.sum(data['sps_out'])/data['sps_out'].shape[0]).detach().numpy()
            if args.do_not_skip_data_with_higher_than_average_firing_rate and avg_sps_in_sample >= train_data_loader.dataset.avg_firing_rate:
                pass
            elif np.random.random() < args.skip_data_prob:
                continue 

            out_v, out_sp = model_forward(data, self_reliant=using_self_reliant_train)

            # compute loss and collect results
            full_loss, y_preds, y_trues, v_preds, v_trues, loss_lists = compute_losses_and_collect_results(out_v, out_sp, data, y_preds, y_trues, v_preds, v_trues, loss_lists, state="train", data_index=data_index, model_fwd=lambda x: model_forward(x, self_reliant=using_self_reliant_train, data_is_dict=False))

            full_loss = full_loss.double()
            
            # backward pass
            full_loss.backward()

            if args.use_clip_grad_norm:
                # clip gradients (replace `model` with your model variable)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad_norm_max_norm)  # norm_type=2 by default
        
            # update weights
            optimizer.step()
            if using_cosine_annealing_lr_scheduler:
                scheduler.step()
            optimizer.zero_grad()

            count_steps_since_last_show += len(data['sps_in'])
            steps += len(data['sps_in'])

            # show results after args.show_train_results_every_in_steps steps
            if count_steps_since_last_show > args.show_train_results_every_in_steps:
                log_and_write_and_plot(ep, y_preds, y_trues, v_preds, v_trues, loss_lists, steps, total_training_time_in_seconds, 'train')

                loss_lists = {'full': [], 'both':[], 'sps':[], 'rmse':[]}    
                y_preds,y_trues,v_preds, v_trues = [], [], [], []       
                count_steps_since_last_show=0

            ################## Finish training ################
            if (total_training_time_in_seconds + time.time() - current_train_start_time)/60/60 > args.maximum_train_time_in_hours or steps > args.maximum_train_steps:
                finish = True
                break

            if len(eval_history['train']) > 0 and eval_history['train'][np.sort(list(eval_history['train'].keys()))[-1]]['lr_changes'] >= args.maximum_lr_changes:
                finish = True
                break

            ################## Test every x [hours] time or every y [fraction] of train ################
            if (time.time() - current_train_start_time)/60/60 > args.test_every_in_hours or last_train_step + len(train_data_loader.dataset) * args.test_every_in_fraction < steps\
                or last_train_step + args.test_every_in_steps < steps:
                current_test_start_time = time.time()

                logger.info(f"In {test_name} mode now.")
                model.eval()

                loss_lists = {'full': [], 'both':[], 'sps':[], 'rmse':[]}
                y_preds, y_trues, v_preds, v_trues = [], [], [], []
                for data_test_index, data_test in enumerate(test_data_loader):
                    with torch.no_grad():
                        out_v, out_sp = model_forward(data_test, self_reliant=True)

                    # compute loss and collect results
                    full_loss, y_preds, y_trues, v_preds, v_trues, loss_lists = compute_losses_and_collect_results(out_v, out_sp, data_test, y_preds, y_trues, v_preds, v_trues, loss_lists, state="test", data_index=data_test_index, model_fwd=lambda x: model_forward(x, self_reliant=True, data_is_dict=False))

                    if (time.time() - current_test_start_time)/60/60 > args.maximum_test_time_in_hours_per_test:
                        break
                
                if using_reduce_lr_on_plateau_scheduler and using_self_reliant_train:
                    scheduler.step(np.mean(loss_lists['full']))

                results = log_and_write_and_plot(ep, y_preds, y_trues, v_preds, v_trues, loss_lists, steps, total_training_time_in_seconds, test_name)
                test_results = results

                # save at the end of test
                neuron_nn_filename = f'{models_folder}/model_{ep}_{steps}'

                # save results
                results['neuron_nn_filename'] = neuron_nn_filename
                pickle.dump(results, open(f'{models_folder}/model_{ep}_{steps}_{test_name}_results.pkl', 'wb'), protocol=-1)

                ################## Test Normalized every x [count] tests or every y [fraction] of train ################
                if (count_tests - 1) % args.test_normalized_every_in_count_tests == 0 or\
                     last_train_step + len(train_data_loader.dataset) * args.test_normalized_every_in_fraction < steps\
                        or last_train_step + args.test_normalized_every_in_steps < steps:
                    current_normalized_test_time = time.time()
                    last_train_step = steps

                    logger.info(f"In {test_name} normalized mode now.")
                    model.eval()

                    loss_lists = {'full':[], 'both':[], 'sps':[], 'rmse':[]}
                    y_preds, y_trues, v_preds, v_trues = [], [], [], []
                    for data_normalized_test_index, data_normalized_test in enumerate(normalized_test_data_loader):
                        with torch.no_grad():
                            out_v, out_sp = model_forward(data_normalized_test, self_reliant=True)

                        # compute loss and collect results
                        full_loss, y_preds, y_trues, v_preds, v_trues, loss_lists = compute_losses_and_collect_results(out_v, out_sp, data_normalized_test, y_preds, y_trues, v_preds, v_trues, loss_lists, state="test", data_index=data_normalized_test_index, model_fwd=lambda x: model_forward(x, self_reliant=True, data_is_dict=False))

                        if (time.time() - current_normalized_test_time)/60/60 > args.maximum_normalized_test_time_in_hours_per_normalized_test:
                            break

                    results = log_and_write_and_plot(ep, y_preds, y_trues, v_preds, v_trues, loss_lists, steps, total_training_time_in_seconds, f'normalized_{test_name}')
                    
                    # save results
                    results['neuron_nn_filename'] = neuron_nn_filename
                    pickle.dump(results, open(f'{models_folder}/model_{ep}_{steps}_normalized_{test_name}_results.pkl', 'wb'), protocol=-1)

                    auc_for_best_name = 'normalized_auc' if args.best_by_normalized else 'auc'
                    auc_for_best = results['auc'] if args.best_by_normalized else test_results['auc']
                    best_auc_for_best = best_save['normalized_auc'] if args.best_by_normalized else best_save['auc']

                    if auc_for_best > best_auc_for_best:
                        logger.info(f"New best model on epoch {ep} and after {steps} steps, with {auc_for_best_name} of {auc_for_best}")

                        best_save["ep"] = ep
                        best_save["total_training_time_in_seconds"] = total_training_time_in_seconds
                        best_save["auc"] = test_results["auc"]
                        best_save["normalized_auc"] = results["auc"]
                        best_save["triplet_auc"] = test_results["triplet_auc"]
                        best_save["normalized_triplet_auc"] = results["triplet_auc"]
                        best_save["prc_auc"] = test_results["prc_auc"]
                        best_save["normalized_prc_auc"] = results["prc_auc"]
                        best_save["aps"] = test_results["aps"]
                        best_save["normalized_aps"] = results["aps"]
                        best_save["somatic_voltage_explained_variance"] = test_results["somatic_voltage_explained_variance"]
                        best_save["normalized_somatic_voltage_explained_variance"] = results["somatic_voltage_explained_variance"]
                        best_save["soma_RMSE"] = test_results["soma_RMSE"]
                        best_save["normalized_soma_RMSE"] = results["soma_RMSE"]
                        best_save["soma_MAE"] = test_results["soma_MAE"]
                        best_save["steps"] = steps
                        best_save["loss"] = test_results["loss"]
                        best_save["normalized_loss"] = results["loss"]
                        best_save["lr"] = test_results["lr"]
                        best_save["normalized_lr"] = results["lr"]
                        best_save["neuron_nn_filename"] = neuron_nn_filename

                # actual saving (after updating best_save)
                light_save = {'epoch':ep,
                    'step':steps,
                    'lr': optimizer.param_groups[0]['lr'],
                    'args': args,
                    'in_chans': in_chans,
                    'total_training_time_in_seconds': total_training_time_in_seconds + time.time() - current_train_start_time,
                    'wandb_id': wandb_id,
                    'best_save': best_save,
                    'eval_history': eval_history,
                    'neuron_nn_filename': neuron_nn_filename}

                heavy_save = copy.deepcopy(light_save)
                heavy_save['model_state_dict'] = model.state_dict()
                heavy_save['optimizer_state_dict'] = optimizer.state_dict()
                heavy_save['scheduler_state_dict'] = scheduler.state_dict()

                torch.save(heavy_save, neuron_nn_filename)
                pickle.dump(light_save, open(f'{neuron_nn_filename}_light.pkl', 'wb'), protocol=-1)

                loss_lists = {'full':[], 'both':[], 'sps':[], 'rmse':[]}
                y_preds,y_trues,v_preds, v_trues = [], [], [], []
                last_train_step = steps

                writer.add_scalar('train_files', train_data_loader.dataset.count_simulations, steps)
                if wandb_init:
                    wandb.log({'train_files': train_data_loader.dataset.count_simulations}, step=steps)

                logger.info("In train mode now.")
                model.train()
                optimizer.zero_grad()
                total_training_time_in_seconds += time.time() - current_train_start_time
                current_train_start_time = time.time()

        if finish:
            break

        # save at the end of epoch
        cur_neuron_nn_filename = f'{models_folder}/model_{ep}_{steps}'
        light_save = {'epoch':ep,
            'step':steps,
            'lr': optimizer.param_groups[0]['lr'],
            'args': args,
            'in_chans': in_chans,
            'total_training_time_in_seconds': total_training_time_in_seconds + time.time() - current_train_start_time,
            'wandb_id': wandb_id,
            'best_save': best_save,
            'eval_history': eval_history,
            'neuron_nn_filename': cur_neuron_nn_filename}

        heavy_save = copy.deepcopy(light_save)
        heavy_save['model_state_dict'] = model.state_dict()
        heavy_save['optimizer_state_dict'] = optimizer.state_dict()
        heavy_save['scheduler_state_dict'] = scheduler.state_dict()

        torch.save(heavy_save, cur_neuron_nn_filename)
        pickle.dump(light_save, open(f'{cur_neuron_nn_filename}_light.pkl', 'wb'), protocol=-1)
    
    logger.info("Finished training, doing last tests")

    ep += 1

    # last test, a full test
    current_test_start_time = time.time()

    logger.info(f"In {test_name} mode now.")
    model.eval()

    loss_lists = {'full':[], 'both':[], 'sps':[], 'rmse':[]}
    y_preds, y_trues, v_preds, v_trues = [], [], [], []
    for data_test_index, data_test in enumerate(test_data_loader):
        with torch.no_grad():
            out_v, out_sp = model_forward(data_test, self_reliant=True)

        # compute loss and collect results
        full_loss, y_preds, y_trues, v_preds, v_trues, loss_lists = compute_losses_and_collect_results(out_v, out_sp, data_test, y_preds, y_trues, v_preds, v_trues, loss_lists, state="test", data_index=data_test_index, model_fwd=lambda x: model_forward(x, self_reliant=True, data_is_dict=False))
    
    if using_reduce_lr_on_plateau_scheduler and using_self_reliant_train:
        scheduler.step(np.mean(loss_lists['full']))

    results = log_and_write_and_plot(ep, y_preds, y_trues, v_preds, v_trues, loss_lists, steps, total_training_time_in_seconds, test_name)

    # save at the end of test
    neuron_nn_filename = f'{models_folder}/model_{ep}_{steps}'

    # save results
    results['neuron_nn_filename'] = neuron_nn_filename
    test_results = results
    pickle.dump(results, open(f'{models_folder}/model_{ep}_{steps}_test_results.pkl', 'wb'), protocol=-1)
    last_test_results = results

    count_tests += 1

    logger.info(f"Last {test_name} finished, it took {(time.time() - current_test_start_time)/60/60:.2f} hours")

    # last normalized test
    current_normalized_test_time = time.time()
    last_train_step = steps

    logger.info(f"In {test_name} normalized mode now.")
    model.eval()

    loss_lists = {'full':[], 'both':[], 'sps':[], 'rmse':[]}
    y_preds, y_trues, v_preds, v_trues = [], [], [], []
    for data_normalized_test_index, data_normalized_test in enumerate(normalized_test_data_loader):
        with torch.no_grad():
            out_v, out_sp = model_forward(data_normalized_test, self_reliant=True)
            
        # compute loss and collect results
        full_loss, y_preds, y_trues, v_preds, v_trues, loss_lists = compute_losses_and_collect_results(out_v, out_sp, data_normalized_test, y_preds, y_trues, v_preds, v_trues, loss_lists, state="test", data_index=data_normalized_test_index, model_fwd=lambda x: model_forward(x, self_reliant=True, data_is_dict=False))

    results = log_and_write_and_plot(ep, y_preds, y_trues, v_preds, v_trues, loss_lists, steps, total_training_time_in_seconds, f'normalized_{test_name}')
    pickle.dump(results, open(f'{models_folder}/model_{ep}_{steps}_normalized_{test_name}_results.pkl', 'wb'), protocol=-1)
    last_normalized_test_results = results

    auc_for_best_name = 'normalized_auc' if args.best_by_normalized else 'auc'
    auc_for_best = results['auc'] if args.best_by_normalized else test_results['auc']
    best_auc_for_best = best_save['normalized_auc'] if args.best_by_normalized else best_save['auc']

    if auc_for_best > best_auc_for_best:
        logger.info(f"New best model on epoch {ep} and after {steps} steps, with {auc_for_best_name} of {auc_for_best}")
        
        best_save["ep"] = ep
        best_save["total_training_time_in_seconds"] = total_training_time_in_seconds
        best_save["auc"] = test_results["auc"]
        best_save["normalized_auc"] = results["auc"]
        best_save["triplet_auc"] = test_results["triplet_auc"]
        best_save["normalized_triplet_auc"] = results["triplet_auc"]
        best_save["prc_auc"] = test_results["prc_auc"]
        best_save["normalized_prc_auc"] = results["prc_auc"]
        best_save["aps"] = test_results["aps"]
        best_save["normalized_aps"] = results["aps"]
        best_save["somatic_voltage_explained_variance"] = test_results["somatic_voltage_explained_variance"]
        best_save["normalized_somatic_voltage_explained_variance"] = results["somatic_voltage_explained_variance"]
        best_save["soma_RMSE"] = test_results["soma_RMSE"]
        best_save["normalized_soma_RMSE"] = results["soma_RMSE"]
        best_save["soma_MAE"] = test_results["soma_MAE"]
        best_save["steps"] = steps
        best_save["loss"] = test_results["loss"]
        best_save["normalized_loss"] = results["loss"]
        best_save["lr"] = test_results["lr"]
        best_save["normalized_lr"] = results["lr"]
        best_save["neuron_nn_filename"] = neuron_nn_filename

    # actual saving (after updating best_save)
    light_save = {'epoch':ep,
        'step':steps,
        'lr': optimizer.param_groups[0]['lr'],
        'args': args,
        'in_chans': in_chans,
        'total_training_time_in_seconds': total_training_time_in_seconds + time.time() - current_train_start_time,
        'wandb_id': wandb_id,
        'best_save': best_save,
        'eval_history': eval_history,
        'neuron_nn_filename': neuron_nn_filename}

    heavy_save = copy.deepcopy(light_save)
    heavy_save['model_state_dict'] = model.state_dict()
    heavy_save['optimizer_state_dict'] = optimizer.state_dict()
    heavy_save['scheduler_state_dict'] = scheduler.state_dict()

    torch.save(heavy_save, neuron_nn_filename)
    pickle.dump(light_save, open(f'{neuron_nn_filename}_light.pkl', 'wb'), protocol=-1)

    # save results
    results['neuron_nn_filename'] = neuron_nn_filename
    pickle.dump(results, open(f'{models_folder}/model_{ep}_{steps}_normalized_{test_name}_results.pkl', 'wb'), protocol=-1)
    logger.info(f"Last normalized {test_name} finished, it took {(time.time() - current_normalized_test_time)/60/60:.2f} hours")

    best_ep = best_save['ep']
    best_steps = best_save['steps']

    try:
        best_results = pickle.load(open(f'{models_folder}/model_{best_ep}_{best_steps}_{test_name}_results.pkl', "rb"))
        best_normalized_results = pickle.load(open(f'{models_folder}/model_{best_ep}_{best_steps}_normalized_{test_name}_results.pkl', "rb"))
    except:
        continue_training_from_final_results = pickle.load(open(f'{args.continue_training_from[:-7]}/final_results.pkl', "rb"))
        best_results = continue_training_from_final_results['best_results']
        best_normalized_results = continue_training_from_final_results['best_normalized_results']

    # save final results
    final_results = {}
    final_results['last_neuron_nn_filename'] = neuron_nn_filename
    final_results['last_results'] = last_test_results
    final_results['last_normalized_results'] = last_normalized_test_results
    
    final_results['best_neuron_nn_filename'] = best_save["neuron_nn_filename"]
    final_results['best_results'] = best_results
    final_results['best_normalized_results'] = best_normalized_results
    pickle.dump(final_results, open(f'{args.neuron_nn_folder}/final_results.pkl', 'wb'), protocol=-1)

    if wandb_init:
        wandb.finish()


    train_neuron_nn_duration_in_seconds = time.time() - train_neuron_nn_start_time
    logger.info(f"train neuron nn finished!, it took {train_neuron_nn_duration_in_seconds/60.0:.3f} minutes")

    if args.finish_file:
        with open(args.finish_file, 'w') as f:
            f.write('finished')

    return train_neuron_nn_duration_in_seconds

def get_nn_training_args():
    saver = ArgumentSaver()
    saver.add_argument('--random_seed', default=None, type=int)

    saver.add_argument('--run_on_gpu', type=str2bool, nargs='?', const=True, default=True, action=AddDefaultInformationAction)

    # data loading parameters
    saver.add_argument('--count_train_workers', default=2, type=int)
    saver.add_argument('--count_test_workers', default=1, type=int)
    saver.add_argument('--count_normalized_test_workers', default=1, type=int)
    saver.add_argument('--use_valid_as_test', type=str2bool, nargs='?', const=True, default=True)
    saver.add_argument('--best_by_normalized', type=str2bool, nargs='?', const=True, default=True)

    saver.add_argument('--use_elm', type=str2bool, nargs='?', const=True, default=True, action=AddDefaultInformationAction)

    # elm parameters
    saver.add_argument('--elm_branch_mode', type=str2bool, nargs='?', const=True, default=False, action=AddDefaultInformationAction)
    saver.add_argument('--elm_num_memory', default=100, type=int)
    saver.add_argument('--elm_mlp_num_layers', default=1, type=int)
    saver.add_argument('--elm_memory_tau_max', default=150.0, type=float)
    saver.add_argument('--elm_memory_tau_min', default=1.0, type=float)
    saver.add_argument('--elm_learn_memory_tau', type=str2bool, nargs='?', const=True, default=False, action=AddDefaultInformationAction)

    # general parameters (loss, lr, batch, etc)
    saver.add_argument('--use_data_parallel',  type=str2bool, nargs='?', const=True, default=True)
    saver.add_argument('--skip_data_prob', type=float, default=0.0)
    saver.add_argument('--do_not_skip_data_with_higher_than_average_firing_rate', type=str2bool, nargs='?', const=True, default=True)
    saver.add_argument('--batch_size', type=int, default=128)
    saver.add_argument('--weight_decay', type=float, default=1e-8)
    saver.add_argument('--leaky_relu_alpha', type=float, default=0.3)
    saver.add_argument('--fl_alpha', type=float, default=0.25)
    saver.add_argument('--fl_gamma', type=float, default=1.0)
    saver.add_argument('--use_pos_weight',  type=str2bool, nargs='?', const=True, default=False)
    saver.add_argument('--use_focal_loss',  type=str2bool, nargs='?', const=True, default=True)
    saver.add_argument('--use_nadam',  type=str2bool, nargs='?', const=True, default=True)
    saver.add_argument('--use_adamw',  type=str2bool, nargs='?', const=True, default=True)
    saver.add_argument('--lr', type=float, default=0.007)
    saver.add_argument('--lr_change_threshold', type=float, default=1e-10)
    saver.add_argument('--reduce_lr_on_plateau_factor', type=float, default=0.8)
    saver.add_argument('--reduce_lr_on_plateau_patience', type=int, default=12)
    saver.add_argument('--use_cosine_annealing_lr',  type=str2bool, nargs='?', const=True, default=False)
    saver.add_argument('--use_step_lr',  type=str2bool, nargs='?', const=True, default=False)
    saver.add_argument('--step_lr_step_size', type=int, default=1)
    saver.add_argument('--step_lr_gamma', type=float, default=0.7)
    saver.add_argument('--reset_lr',  type=str2bool, nargs='?', const=True, default=False)
    saver.add_argument('--v_loss_weight', default=0.02, type=float)
    saver.add_argument('--use_taylor_consistency_loss',  type=str2bool, nargs='?', const=True, default=False)
    saver.add_argument('--taylor_consistency_loss_weight', default=0.25, type=float)
    saver.add_argument('--taylor_consistency_loss_every_in_samples', type=int, default=20)
    saver.add_argument('--use_clip_grad_norm',  type=str2bool, nargs='?', const=True, default=False)
    saver.add_argument('--clip_grad_norm_max_norm', type=float, default=5.0)

    # train parameters
    saver.add_argument('--count_epochs', type=int, default=int(1e20), action=AddDefaultInformationAction) # default infinity
    saver.add_argument('--show_train_results_every_in_steps', default=5000, type=int, action=AddDefaultInformationAction) # default 5k
    saver.add_argument('--test_every_in_hours', default=int(1e20), type=float, action=AddDefaultInformationAction) # default infinity 
    saver.add_argument('--test_every_in_steps', default=50000, type=int, action=AddDefaultInformationAction) # default 50k
    saver.add_argument('--test_every_in_fraction', default=float(1e20), type=float)  # default infinity
    saver.add_argument('--maximum_train_time_in_hours', default=int(1e20), type=float, action=AddDefaultInformationAction) # default infinity 
    saver.add_argument('--maximum_train_steps', default=10000000, type=int, action=AddDefaultInformationAction) # default 10m
    saver.add_argument('--maximum_lr_changes', default=int(1e20), type=int, action=AddDefaultInformationAction) # default infinity
    saver.add_argument('--maximum_test_time_in_hours_per_test', default=float(1e20), type=float) # default infinity
    saver.add_argument('--test_normalized_every_in_count_tests', default=3, type=int) # default 3
    saver.add_argument('--test_normalized_every_in_fraction', default=float(1e20), type=float) # default infinity
    saver.add_argument('--test_normalized_every_in_steps', default=50000, type=int) # default 50k
    saver.add_argument('--maximum_normalized_test_time_in_hours_per_normalized_test', default=float(1e20), type=float) # default infinity
    saver.add_argument('--default_threshold', type=float, default=0.5)
    saver.add_argument('--desired_FP_list', nargs='+', type=float, default=[0.0001, 0.0010, 0.0020, 0.0050, 0.0200, 0.0400, 0.0500, 0.1000])
    saver.add_argument('--count_annotations', default=15, type=int)
    saver.add_argument('--plot_window_size', default=1000, type=int)
    saver.add_argument('--false_positive_rate_for_plot', default=0.002, type=float)
    saver.add_argument('--spike_voltage_value_for_plot', default=-40, type=float)
    saver.add_argument('--continue_training_from', default=None)
    saver.add_argument('--train_not_self_reliant', type=str2bool, nargs='?', const=True, default=True)


    # data parameters
    saver.add_argument('--simulation_initialization_duration_in_ms', default=500, type=int)
    saver.add_argument('--window_size', type=int, default=700)
    saver.add_argument('--default_v_offset', type=float, default=-67.7)
    saver.add_argument('--spike_threshold', default=-55, type=int)
    saver.add_argument('--clip_somatic_voltage_to_spike_threshold', type=str2bool, nargs='?', const=True, default=True)

    return saver

def get_args():
    parser = argparse.ArgumentParser(description='Train NN for a neuron')
    parser.add_argument('--simulation_dataset_folder')
    parser.add_argument('--neuron_nn_folder', action=AddOutFileAction)
    parser.add_argument('--neuron_nn_name', default=None)

    from utils.slurm_job import get_job_args
    job_saver = get_job_args()
    job_saver.add_to_parser(parser)

    saver = get_nn_training_args()
    saver.add_to_parser(parser)
    
    parser.add_argument('--save_plots', type=str2bool, nargs='?', const=True, default=True)
    return parser.parse_args()

def main():
    args = get_args()
    TeeAll(args.outfile)
    setup_logger(logging.getLogger())

    job_id = os.environ["SLURM_JOB_ID"] if "SLURM_JOB_ID" in os.environ else -1
    logger.info(f"Welcome to neuron nn trainer! running on {os.uname()} (job_id={job_id}, pid={os.getpid()}, ppid={os.getppid()})")
    train_neuron_nn_on_data(args)
    logger.info(f"Goodbye from neuron nn trainer! running on {os.uname()} (job_id={job_id}, pid={os.getpid()}, ppid={os.getppid()})")

if __name__ == "__main__":
    main()
