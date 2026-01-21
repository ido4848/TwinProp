import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities import grad_norm
from scipy import sparse
import torchmetrics
import logging
import sys
import types
import pathlib
import os
from enum import IntEnum

sys.path.append(str(pathlib.Path(__file__).parent.absolute()))
sys.path.append(str(pathlib.Path(__file__).parent.parent.absolute()))

from training_nets.constrained_linear import ConstrainedLinear
from utils.utils import setup_logger
from utils.surrogate_spike_gradient import get_surrogate_spike_gradient

logger = logging.getLogger(__name__)

class DecodingType(IntEnum):
    NONE = 0
    MAX_POOLING = 1
    LINEAR = 2
    LINEAR_BIAS = 3
    BINARY_LINEAR = 4
    BINARY_LINEAR_BIAS = 5
    LINEAR_SOFTMAX = 6
    LINEAR_BIAS_SOFTMAX = 7
    BINARY_LINEAR_SOFTMAX = 8
    BINARY_LINEAR_BIAS_SOFTMAX = 9
    SUM_POOLING = 10

class UtilizerVerbosity(IntEnum):
    NONE = 0
    LOW = 1
    HIGH = 2

class OptimizerType(IntEnum):
    SGD = 0
    ADAM = 1    


def binary_to_hot_cmap(x):
    max_weight = np.max(x)

    if max_weight > 1:
        first = int(128 / (np.ceil(max_weight)-1))
        first = max(first, 1)
        colors1 = plt.cm.binary(np.linspace(0., 1, first))
        # colors2 = plt.cm.gist_heat(np.linspace(0, 0.8, 256-first))
        colors2 = plt.cm.hot(np.linspace(0, 0.8, 256-first))
        colors = np.vstack((colors1, colors2))
        mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
        chosen_cmap = mymap
    else:
        chosen_cmap = plt.cm.binary

    return chosen_cmap

def plot_function_neuron_model(utilizer, x, y, y_extra, y_for_accuracy, z, pred, loss, state, current_epoch, batch_idx, current_plot_folder):
    count_examples_to_plot = 5
    count_examples_to_plot = min(count_examples_to_plot, x.shape[0])
    indices = np.random.choice(x.shape[0], count_examples_to_plot, replace=False)

    if utilizer.use_wiring_layer:
        fig = plt.figure(figsize=(15, 2 + count_examples_to_plot * 4))
        count_cols = 5
        if utilizer.wiring_dales_law:
            count_cols = 6
        axs = fig.subplots(count_examples_to_plot, count_cols)
        if count_examples_to_plot == 1:
            axs = [axs]
        for i, idx in enumerate(indices):
            with torch.no_grad():
                cur_input = x[idx].squeeze().detach().cpu().numpy()
                weights = utilizer.wiring_layer.get_weights().detach().cpu().numpy()
                cur_input_before_model = utilizer.last_x_before_model[idx].detach().cpu().numpy()
                cur_output_after_model = utilizer.first_x_after_model[idx].detach().cpu().numpy()
                cur_gt = y_for_accuracy[idx].detach().cpu().numpy()
                cur_y_extra = None
                if y_extra is not None:
                    cur_y_extra = y_extra[idx].detach().cpu().numpy()
                spike = z[0][idx].squeeze().detach().cpu().numpy()
                spike_pred = pred[idx].detach().cpu().numpy()

            is_gt_array = False
            # if cur_gt is np array
            if len(y_for_accuracy.shape) > 1 and y_for_accuracy.shape[1] > 1:
                is_gt_array = True

            input_text = f"input (idx = {idx}"
            if not is_gt_array:
                input_text += f", gt = {cur_gt}"
                if cur_y_extra is not None:
                    input_text += f", extra = {cur_y_extra}"
            input_text += ")"
            axs[i][0].matshow(cur_input, aspect='auto', cmap=binary_to_hot_cmap(cur_input))
            axs[i][0].xaxis.set_ticks_position('bottom')
            axs[i][0].set_xlabel("time [ms]")
            axs[i][0].set_ylabel("axon")
            axs[i][0].set_title(input_text)

            if utilizer.functional_only_wiring:
                # matshow only diagonal
                weights = np.diag(weights).reshape(-1, 1)

            if utilizer.wiring_dales_law:
                weights_exc = weights_first_quarter = weights[:weights.shape[0]//2, :weights.shape[1]//2]
                weights_inh = weights_fourth_quarter = weights[weights.shape[0]//2:, weights.shape[1]//2:]

                wiring_exc_text = f"wiring excitatory weights\nmin={weights_exc.min():.3f}, max={weights_exc.max():.3f}\nmean={np.mean(weights_exc):.3f}, std={np.std(weights_exc):.3f}\nsum={np.sum(weights_exc):.3f}"

                axs[i][1].imshow(weights_exc, aspect='auto', cmap='hot')
                axs[i][1].xaxis.set_ticks_position('bottom')
                axs[i][1].set_xlabel("axon")
                axs[i][1].set_ylabel("segment")
                axs[i][1].set_title(wiring_exc_text)

                wiring_inh_text = f"wiring inhibitory weights\nmin={weights_inh.min():.3f}, max={weights_inh.max():.3f}\nmean={np.mean(weights_inh):.3f}, std={np.std(weights_inh):.3f}\nsum={np.sum(weights_inh):.3f}"

                axs[i][2].imshow(weights_inh, aspect='auto', cmap='hot')
                axs[i][2].xaxis.set_ticks_position('bottom')
                axs[i][2].set_xlabel("axon")
                axs[i][2].set_ylabel("segment")
                axs[i][2].set_title(wiring_inh_text)
            
                last_weights_col = 2
            else:
                wiring_text = f"wiring weights\nmin={weights.min():.3f}, max={weights.max():.3f}\nmean={np.mean(weights):.3f}, std={np.std(weights):.3f}\nsum={np.sum(weights):.3f}"

                axs[i][1].matshow(weights, aspect='auto', cmap='hot')
                axs[i][1].xaxis.set_ticks_position('bottom')
                axs[i][1].set_xlabel("axon")
                axs[i][1].set_ylabel("segment")
                axs[i][1].set_title(wiring_text)

                last_weights_col = 1

            axs[i][last_weights_col+1].matshow(cur_input_before_model, aspect='auto', cmap=binary_to_hot_cmap(cur_input_before_model))
            axs[i][last_weights_col+1].xaxis.set_ticks_position('bottom')
            axs[i][last_weights_col+1].set_xlabel("time [ms]")
            axs[i][last_weights_col+1].set_ylabel("segment")
            axs[i][last_weights_col+1].set_title("input before model")

            axs[i][last_weights_col+2].plot(cur_output_after_model)
            axs[i][last_weights_col+2].xaxis.set_ticks_position('bottom')
            axs[i][last_weights_col+2].set_xlabel("time [ms]")
            axs[i][last_weights_col+2].set_ylabel("spike prediction")
            axs[i][last_weights_col+2].set_title("output after model")

            axs[i][last_weights_col+3].plot(spike)
            axs[i][last_weights_col+3].set_xlabel("time [ms]")
            axs[i][last_weights_col+3].set_ylabel("spike")
            if is_gt_array:
                axs[i][last_weights_col+3].set_title(f"spike")
            else:
                axs[i][last_weights_col+3].set_title(f"spike (pred = {spike_pred})")

            if is_gt_array:
                axs[i][last_weights_col+3].plot(cur_gt, color='gray', linestyle='--')    

            # signify utilizer.effective_decoding_time_from_end using shaded area
            axs[i][last_weights_col+3].axvspan(xmin=spike.shape[0] - utilizer.effective_decoding_time_from_end, xmax=spike.shape[0], color='green', alpha=0.5)
            
            # signify utilizer.binarized_decoding_threshold using horizontal line, if not None
            if utilizer.binarized_decoding_threshold is not None:
                axs[i][last_weights_col+3].axhline(y=utilizer.binarized_decoding_threshold, color='red', linestyle='--')

        plt.subplots_adjust(wspace=0.5, hspace=0.5)


    else:
        fig = plt.figure(figsize=(15, 2 + count_examples_to_plot * 4))
        axs = fig.subplots(count_examples_to_plot, 2)
        if count_examples_to_plot == 1:
            axs = [axs]
        for i, idx in enumerate(indices):
            with torch.no_grad():
                cur_input = x[idx].squeeze().detach().cpu().numpy()
                cur_gt = y_for_accuracy[idx].detach().cpu().numpy()
                cur_y_extra = None
                if y_extra is not None:
                    cur_y_extra = y_extra[idx].detach().cpu().numpy()
                spike = z[0][idx].squeeze().detach().cpu().numpy()
                spike_pred = pred[idx].detach().cpu().numpy()

            input_text = f"input (idx = {idx}, gt = {cur_gt}"
            if cur_y_extra is not None:
                input_text += f", extra = {cur_y_extra}"
            input_text += ")"
            axs[i][0].matshow(cur_input, aspect='auto')
            axs[i][0].xaxis.set_ticks_position('bottom')
            axs[i][0].set_xlabel("time [ms]")
            axs[i][0].set_ylabel("axon")
            axs[i][0].set_title(input_text)

            axs[i][1].plot(spike)
            axs[i][1].set_xlabel("time [ms]")
            axs[i][1].set_ylabel("spike")
            axs[i][1].set_title(f"spike (pred = {spike_pred})")

        plt.subplots_adjust(hspace=0.5)

    fig.savefig(f"{current_plot_folder}/epoch_{current_epoch}_batch_{batch_idx}.png")
    plt.close('all')

# that's the magic. magic is not always easy to understand. LOL
class NeuronUtilizer(pl.LightningModule):
    def __init__(self, model, model_shape, ds_shape, ds_extra_label_information=False, disable_model_last_layer=False, freeze_model=True,
     use_time_left_padding_if_needed=True, extra_time_left_padding_if_needed=0, time_left_padding_firing_rate=0.0,
     time_left_padding_before_wiring=False,
      use_wiring_layer=True, positive_wiring=True, wiring_zero_smaller_than=None, wiring_keep_max_k_from_input=None,
       wiring_keep_max_k_to_output=None, wiring_keep_weight_mean=None, wiring_keep_weight_std=None, wiring_keep_weight_max=None, functional_only_wiring=False, 
       wiring_bias=False, wiring_weight_init_mean=0, wiring_weight_init_bound=None, wiring_weight_init_sparsity=None, 
       wiring_enforce_every_in_train_epochs=None, wiring_enforce_every_in_train_batches=None, wiring_dales_law=False, wiring_weight_l1_reg=0.0, wiring_weight_l2_reg=0.0,
        population_k=1, use_population_masking_layer=False, positive_population_masking=False, functional_only_population_masking=False,
         population_masking_bias=False, population_masking_weight_init_mean=0, population_masking_weight_init_bound=None, population_masking_weight_init_sparsity=None,
          decoding_type=DecodingType.BINARY_LINEAR_SOFTMAX, decoding_time_from_end=None, require_no_spikes_before_decoding_time=False,
           grad_abs=False, positive_by_sigmoid=False, positive_by_softplus=False, optimizer_type=OptimizerType.ADAM, lr=1e-3, momentum=0.9, weight_decay=0.0, step_lr=False, loss_weights=None,
            differentiable_binarization_threshold_surrogate_spike=True, differentiable_binarization_threshold_surrogate_spike_beta=5,
             differentiable_binarization_threshold_straight_through=False, binarized_decoding_threshold=None, binarized_output_threshold=None,
             binarized_decoding_on_predict=True, binarized_output_on_predict=False, argmax_on_predict=True,
              verbosity=UtilizerVerbosity.NONE, enable_progress_bar=False,
               plots_folder=None, enable_plotting=False, plot_function=plot_function_neuron_model,
                plot_train_every_in_epochs=5, plot_valid_every_in_epochs=5,
                 plot_train_every_in_batches=10, plot_valid_every_in_batches=10):
        super(NeuronUtilizer, self).__init__()

        # TODO: ?
        # self.save_hyperparameters(ignore=['model'])

        self.save_hyperparameters()

        self.verbosity = verbosity

        self.enable_progress_bar = enable_progress_bar

        self.plots_folder = plots_folder
        self.enable_plotting = enable_plotting
        self.plot_function = plot_function
        self.plot_train_every_in_epochs = plot_train_every_in_epochs
        self.plot_valid_every_in_epochs = plot_valid_every_in_epochs
        self.plot_train_every_in_batches = plot_train_every_in_batches
        self.plot_valid_every_in_batches = plot_valid_every_in_batches

        ds_input_shape, ds_input_time, ds_output_shape, ds_output_time = ds_shape

        self.ds_input_shape = ds_input_shape
        self.ds_input_dim = np.prod(ds_input_shape)

        self.ds_input_time = ds_input_time

        self.ds_output_shape = ds_output_shape
        self.ds_output_dim = np.prod(ds_output_shape)

        self.ds_output_time = ds_output_time

        self.ds_extra_label_information = ds_extra_label_information

        self.decoding_type = decoding_type

        # TODO: decoding_time -> decoding_duration for both of these
        self.decoding_time_from_end = decoding_time_from_end
        self.require_no_spikes_before_decoding_time = require_no_spikes_before_decoding_time

        self.use_linear_decoding_layer = self.decoding_type in [DecodingType.LINEAR, DecodingType.LINEAR_BIAS,
         DecodingType.BINARY_LINEAR, DecodingType.BINARY_LINEAR_BIAS, DecodingType.LINEAR_SOFTMAX,
            DecodingType.LINEAR_BIAS_SOFTMAX, DecodingType.BINARY_LINEAR_SOFTMAX, DecodingType.BINARY_LINEAR_BIAS_SOFTMAX]
        self.linear_decoding_bias = self.decoding_type in [DecodingType.LINEAR_BIAS, DecodingType.BINARY_LINEAR_BIAS,
         DecodingType.LINEAR_BIAS_SOFTMAX, DecodingType.BINARY_LINEAR_BIAS_SOFTMAX]

        self.binarize_before_linear_decoding_layer = self.decoding_type in [DecodingType.BINARY_LINEAR, DecodingType.BINARY_LINEAR_BIAS,
         DecodingType.BINARY_LINEAR_SOFTMAX, DecodingType.BINARY_LINEAR_BIAS_SOFTMAX]

        self.use_softmax_after_linear_decoding_layer = self.decoding_type in [DecodingType.LINEAR_SOFTMAX, DecodingType.LINEAR_BIAS_SOFTMAX,
         DecodingType.BINARY_LINEAR_SOFTMAX, DecodingType.BINARY_LINEAR_BIAS_SOFTMAX]

        self.population_k = population_k
        
        self.set_model(model, model_shape, disable_model_last_layer,\
             freeze_model, use_time_left_padding_if_needed, extra_time_left_padding_if_needed, time_left_padding_firing_rate, time_left_padding_before_wiring)

        self.grad_abs = grad_abs
        self.positive_by_sigmoid = positive_by_sigmoid
        self.positive_by_softplus = positive_by_softplus

        self.use_wiring_layer = use_wiring_layer
        self.positive_wiring = positive_wiring
        self.wiring_zero_smaller_than = wiring_zero_smaller_than
        self.wiring_keep_max_k_from_input = wiring_keep_max_k_from_input
        self.wiring_keep_max_k_to_output = wiring_keep_max_k_to_output
        self.wiring_keep_weight_mean = wiring_keep_weight_mean
        self.wiring_keep_weight_std = wiring_keep_weight_std
        self.wiring_keep_weight_max = wiring_keep_weight_max
        self.functional_only_wiring = functional_only_wiring
        self.wiring_bias = wiring_bias
        self.wiring_weight_init_mean = wiring_weight_init_mean
        self.wiring_weight_init_bound = wiring_weight_init_bound
        self.wiring_weight_init_sparsity = wiring_weight_init_sparsity
        self.wiring_enforce_every_in_train_epochs = wiring_enforce_every_in_train_epochs
        self.wiring_enforce_every_in_train_batches = wiring_enforce_every_in_train_batches
        self.wiring_dales_law = wiring_dales_law
        self.wiring_weight_l1_reg = wiring_weight_l1_reg
        self.wiring_weight_l2_reg = wiring_weight_l2_reg
        if self.use_wiring_layer:
            self.wiring_layer = ConstrainedLinear(self.ds_input_dim, self.model_input_dim, bias=self.wiring_bias,
             diagonals_only=self.functional_only_wiring, positive_weight=self.positive_wiring, grad_abs=self.grad_abs, positive_by_sigmoid=self.positive_by_sigmoid, positive_by_softplus=self.positive_by_softplus,
              weight_init_mean=self.wiring_weight_init_mean, weight_init_bound=self.wiring_weight_init_bound,
               weight_init_sparsity=self.wiring_weight_init_sparsity, zero_smaller_than=self.wiring_zero_smaller_than,
                keep_max_k_from_input=self.wiring_keep_max_k_from_input, keep_max_k_to_output=self.wiring_keep_max_k_to_output,
                    keep_weight_mean=self.wiring_keep_weight_mean, keep_weight_std=self.wiring_keep_weight_std, keep_weight_max=self.wiring_keep_weight_max,
                    enforce_every_in_train_epochs=self.wiring_enforce_every_in_train_epochs,
                    enforce_every_in_train_batches=self.wiring_enforce_every_in_train_batches, dales_law=self.wiring_dales_law)

            if self.enable_plotting:
                fig = plt.figure(figsize=(15, 5))
                axs = fig.subplots(1, 2)
                weights = self.wiring_layer.get_weights().detach().cpu().numpy()
                if self.functional_only_wiring:
                    weights = np.diag(weights).reshape(-1, 1)
                                
                # TODO: code dup with code in plot_function_neuron_model
                axs[0].matshow(weights, aspect='auto')
                axs[0].xaxis.set_ticks_position('bottom')
                axs[0].set_xlabel("axon")
                axs[0].set_ylabel("synapse")
                axs[0].set_title("wiring")

                axs[1].hist(weights.reshape(-1), bins=100)
                axs[1].set_xlabel("weight")
                axs[1].set_ylabel("count")
                axs[1].set_title("weight histogram")

                fig.subplots_adjust(hspace=0.5)
                fig.suptitle("initial wiring")
                fig.savefig(f"{self.plots_folder}/initial_wiring.png")

                plt.close('all')

                # TODO: this is a hack, move elsewhere
                np.save(f"{self.plots_folder}/initial_wiring.npy", weights)

        self.use_population_masking_layer = use_population_masking_layer
        self.positive_population_masking = positive_population_masking
        self.functional_only_population_masking = functional_only_population_masking
        self.population_masking_bias = population_masking_bias
        self.population_masking_weight_init_mean = population_masking_weight_init_mean
        self.population_masking_weight_init_bound = population_masking_weight_init_bound
        self.population_masking_weight_init_sparsity = population_masking_weight_init_sparsity
        if self.use_population_masking_layer:
            self.population_masking_layer = ConstrainedLinear(self.model_input_dim, self.model_input_dim * self.population_k,
             bias=self.population_masking_bias, diagonals_only=self.functional_only_population_masking,
              positive_weight=self.positive_population_masking, grad_abs=self.grad_abs, positive_by_sigmoid=self.positive_by_sigmoid, positive_by_softplus=self.positive_by_softplus,
              weight_init_mean=self.population_masking_weight_init_mean,
               weight_init_bound=self.population_masking_weight_init_bound, weight_init_sparsity=self.population_masking_weight_init_sparsity)
            # TODO: plot init weights?

        self.cross_entropy = nn.CrossEntropyLoss(weight=loss_weights)
        self.mse = nn.MSELoss()

        self.optimizer_type = optimizer_type
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.step_lr = step_lr

        self.differentiable_binarization_threshold_surrogate_spike = differentiable_binarization_threshold_surrogate_spike
        self.differentiable_binarization_threshold_surrogate_spike_beta = differentiable_binarization_threshold_surrogate_spike_beta
        self.differentiable_binarization_threshold_straight_through = differentiable_binarization_threshold_straight_through

        self.binarized_decoding_threshold = binarized_decoding_threshold if binarized_decoding_threshold is not None else self.model_binarization_threshold

        self.binarized_output_threshold = binarized_output_threshold if binarized_output_threshold is not None else self.model_binarization_threshold

        self.binarized_decoding_on_predict = binarized_decoding_on_predict

        self.binarized_output_on_predict = binarized_output_on_predict

        self.argmax_on_predict = argmax_on_predict

        # TODO: binary always?
        self.train_accuracy = torchmetrics.Accuracy(task='binary')
        self.last_train_accuracy = None
        self.maximum_train_accuracy = 0
        
        # TODO: binary always?
        self.valid_accuracy  = torchmetrics.Accuracy(task='binary')
        self.last_valid_accuracy = None
        self.maximum_valid_accuracy = 0

        # TODO: binary always?
        self.train_auc = torchmetrics.AUROC(task='binary')
        self.last_train_auc = None
        self.maximum_train_auc = 0

        # TODO: binary always?
        self.valid_auc = torchmetrics.AUROC(task='binary')
        self.last_valid_auc = None
        self.maximum_valid_auc = 0

        self.train_mae = torchmetrics.MeanAbsoluteError()
        self.last_train_mae = None
        self.minimum_train_mae = float('inf')

        self.valid_mae = torchmetrics.MeanAbsoluteError()
        self.last_valid_mae = None
        self.minimum_valid_mae = float('inf')

    def set_model(self, model, model_shape, disable_model_last_layer=False, freeze_model=True,\
         use_time_left_padding_if_needed=True, extra_time_left_padding_if_needed=0, time_left_padding_firing_rate=0.0,
          time_left_padding_before_wiring=False):    
        self.model = model

        self.disable_model_last_layer = disable_model_last_layer

        model_input_shape, model_uses_time, model_output_shape, model_output_time_lambda, model_binarization_threshold = model_shape

        if self.disable_model_last_layer:
            model_output_shape = self.model.disable_last_layer()

        self.model_input_shape = model_input_shape
        self.model_input_dim = np.prod(model_input_shape)

        self.model_uses_time = model_uses_time

        self.model_output_shape = model_output_shape
        self.model_output_dim = np.prod(model_output_shape)

        self.time_left_padding_time = 0

        if isinstance(model_output_time_lambda, types.LambdaType):
            self.model_output_time_before_padding = model_output_time_lambda(self.ds_input_time)
            self.model_time_diff = self.ds_input_time - self.model_output_time_before_padding
        else:
            self.model_time_diff = -model_output_time_lambda
            self.model_output_time_before_padding = self.ds_input_time - self.model_time_diff 

        if self.verbosity >= UtilizerVerbosity.LOW:
            logger.info(f"Ds input time: {self.ds_input_time}")
            logger.info(f"Model output time before padding: {self.model_output_time_before_padding}")
            logger.info(f"Model time diff: {self.model_time_diff}")

        self.model_output_time = self.model_output_time_before_padding

        if not use_time_left_padding_if_needed:
            raise NotImplementedError("use_time_left_padding_if_needed=False is not implemented yet")

        if use_time_left_padding_if_needed and self.model_time_diff > 0:
            self.time_left_padding_time = self.model_time_diff

            if self.verbosity >= UtilizerVerbosity.LOW:
                logging.info(f"Using left padding of {self.time_left_padding_time} time steps because model time diff is {self.model_time_diff}.")

            self.model_output_time = self.ds_input_time

            if extra_time_left_padding_if_needed > 0:
                self.time_left_padding_time += extra_time_left_padding_if_needed

                if self.verbosity >= UtilizerVerbosity.LOW:
                    logging.info(f"Using extra left padding of {extra_time_left_padding_if_needed} time steps.")

                self.model_time_diff += extra_time_left_padding_if_needed

        self.time_left_padding_before_wiring = time_left_padding_before_wiring

        self.model_binarization_threshold = model_binarization_threshold

        self.time_left_padding_firing_rate = time_left_padding_firing_rate

        self.freeze_model = freeze_model
        if self.freeze_model:
            for param in self.model.parameters():
                param.requires_grad = False

        self.effective_decoding_time_from_end = self.decoding_time_from_end
        if self.effective_decoding_time_from_end is None:
            self.effective_decoding_time_from_end = self.model_output_time

        if self.use_linear_decoding_layer:
            # TODO: set init mean and bound?
            self.linear_decoding_layer = ConstrainedLinear(self.population_k * self.model_output_dim * self.effective_decoding_time_from_end,
             self.ds_output_dim * self.ds_output_time, bias=self.linear_decoding_bias)
            # TODO: plot init weights?

    def possibly_pad_x(self, x, is_before_wiring):
        if self.time_left_padding_before_wiring and not is_before_wiring:
            return x

        if not self.time_left_padding_before_wiring and is_before_wiring:
            return x
        
        if self.time_left_padding_time > 0:
            # poisson padding with fr self.time_left_padding_firing_rate
            # TODO: one day implement different padding schemes, or even move it out to a separate object Padder?

            # TODO: need torch.no_grad?
            padding_shape = list(x.shape)
            padding_shape[-1] = self.time_left_padding_time
            poisson_padding = torch.rand(padding_shape) < self.time_left_padding_firing_rate/1000.0
            poisson_padding = poisson_padding.to(x.device).float()

            x = torch.cat([poisson_padding, x], dim=-1)
            if self.verbosity >= UtilizerVerbosity.HIGH:
                logger.info(f"after poisson padding: {x.shape}")
        return x

    def forward(self, x):
        if self.verbosity >= UtilizerVerbosity.HIGH:
            logger.info(f"input shape: {x.shape} (0 is always batch size)")

        if not self.model_uses_time and self.ds_output_time == 1:
            x = x.unsqueeze(len(x.shape))
            if self.verbosity >= UtilizerVerbosity.HIGH:
                logger.info(f"after unsqueeze in order to add a time dimension: {x.shape}")            

        input_time = x.shape[-1]
        assert input_time == self.ds_input_time, f"input_time ({input_time}) != self.ds_input_time ({self.ds_input_time})"

        x = self.possibly_pad_x(x, is_before_wiring=True)

        # will not be padded with time_left_padding_before_wiring=False
        # TODO: better name?
        self.padded_first_x = x

        input_time = x.shape[-1]

        # 0 batch, * input_shape, -1 input_time

        if self.use_wiring_layer:
            x = x.reshape(x.shape[0], self.ds_input_dim, input_time)
            if self.verbosity >= UtilizerVerbosity.HIGH:
                logger.info(f"after flatten ds input shape for wiring: {x.shape}")

            x = x.transpose(1,2)
            if self.verbosity >= UtilizerVerbosity.HIGH:
                logger.info(f"after transpose dim and time for wiring: {x.shape}")

            if self.verbosity >= UtilizerVerbosity.LOW:
                x_shape_before = tuple(x.shape)
                wiring_layer_shape = tuple(self.wiring_layer.get_weight_shape())

            x = self.wiring_layer(x.float())

            if self.verbosity >= UtilizerVerbosity.LOW:
                x_shape_after = tuple(x.shape)
                logger.info(f"wiring_layer ({wiring_layer_shape}): {x_shape_before} -> {x_shape_after}")

            x = x.transpose(1,2)
            if self.verbosity >= UtilizerVerbosity.HIGH:
                logger.info(f"after transpose back time and dim from wiring: {x.shape}")

            x = x.reshape(x.shape[0], *self.model_input_shape, input_time)
            if self.verbosity >= UtilizerVerbosity.HIGH:
                logger.info(f"after unflatten ds input dim from wiring: {x.shape}")

        if self.use_population_masking_layer:
            x = x.reshape(x.shape[0], self.model_input_dim, input_time)
            if self.verbosity >= UtilizerVerbosity.HIGH:
                logger.info(f"after flatten model input shape for population masking: {x.shape}")
            
            # 0 batch, 1 input_dim, 2 time

            x = x.transpose(1,2)
            if self.verbosity >= UtilizerVerbosity.HIGH:
                logger.info(f"after transpose dim and time for population masking: {x.shape}")

            # 0 batch, 1 time, 2 input_dim

            if self.verbosity >= UtilizerVerbosity.LOW:
                x_shape_before = tuple(x.shape)
                population_masking_layer_shape = tuple(self.population_masking_layer.get_weight_shape())

            x = self.population_masking_layer(x)

            if self.verbosity >= UtilizerVerbosity.LOW:
                x_shape_after = tuple(x.shape)
                logger.info(f"population_masking_layer ({population_masking_layer_shape}): {x_shape_before} -> {x_shape_after}")

            # 0 batch, 1 time, 2 input_dim * population_k

            # add new dimensions for population
            x = x.reshape(x.shape[0], input_time, self.population_k, self.model_input_dim)
            if self.verbosity >= UtilizerVerbosity.HIGH:
                logger.info(f"after reshape for adding a population dimension: {x.shape}")

            # 0 batch, 1 time, 2 population_k, 3 input_dim

            x = x.transpose(1,2)
            if self.verbosity >= UtilizerVerbosity.HIGH:
                logger.info(f"after transpose population_k and time: {x.shape}")

            # 0 batch, 1 population_k, 2 time, 3 input_dim

            x = x.transpose(2,3)
            if self.verbosity >= UtilizerVerbosity.HIGH:
                logger.info(f"after transpose time and dim: {x.shape}")

            # 0 batch, 1 population_k, 2 input_dim, 3 time

            x = x.reshape(x.shape[0],  self.population_k, *self.model_input_shape, input_time)
            if self.verbosity >= UtilizerVerbosity.HIGH:
                logger.info(f"after unflatten model input dim back from population masking: {x.shape}")

            # 0 batch, 1 population_k, 2 input_shape, 3 time
        
        else:
            # 0 batch, 1 input_shape, 2 time
            x = x.reshape(x.shape[0], 1, *self.model_input_shape, input_time)
            if self.verbosity >= UtilizerVerbosity.HIGH:
                logger.info(f"after reshape for adding a population dimension: {x.shape}")

            x = x.repeat(1, self.population_k,  *[1 for _ in list(self.model_input_shape)], 1)
            if self.verbosity >= UtilizerVerbosity.HIGH:
                logger.info(f"after repeat for population_k: {x.shape}")

            # 0 batch, 1 population_k, 2 input_shape, 3 time

        # 0 batch, 1 population_k, 2 input_shape, 3 time

        x = x.reshape(x.shape[0] * x.shape[1], *self.model_input_shape, input_time)
        if self.verbosity >= UtilizerVerbosity.HIGH:
            logger.info(f"after reshape in order to treat population_k as batch: {x.shape}")

        # 0 batch * population_k, 1 input_shape, 2 time

        if not self.model_uses_time:
            if input_time != 1:
                # TODO: implement one day?
                raise ValueError("timeless model but ds_input_time != 1")

            x = x.squeeze(-1)
            if self.verbosity >= UtilizerVerbosity.HIGH:
                logger.info(f"after squeeze in order to remove time dimension: {x.shape}")

        x = self.possibly_pad_x(x, is_before_wiring=False)

        # TODO: better name?
        self.last_x_before_model = x

        x = self.model(x)
        if self.verbosity >= UtilizerVerbosity.HIGH:
            logger.info(f"after model batch * k times: {x.shape}")

        # TODO: better name?
        self.first_x_after_model = x

        if self.model_time_diff > 0:
            x_shape = list(x.shape)
            x_shape_without_time = x_shape[:-1]
            x_dim_without_time = np.prod(x_shape_without_time)
            x_shape_after_time_diff = list(x.shape)
            x_shape_after_time_diff[-1] = x_shape_after_time_diff[-1] - self.model_time_diff

            x = x.reshape(x_dim_without_time, -1)
            if self.verbosity >= UtilizerVerbosity.HIGH:
                logger.info(f"after reshape in order to apply time diff: {x.shape}")

            x = x[:, self.model_time_diff:]
            if self.verbosity >= UtilizerVerbosity.HIGH:
                logger.info(f"after time diff: {x.shape}")

            x = x.reshape(*x_shape_after_time_diff)
            if self.verbosity >= UtilizerVerbosity.HIGH:
                logger.info(f"after reshape in order to return to original shape: {x.shape}")

        if not self.model_uses_time:
            x = x.unsqueeze(len(x.shape))
            if self.verbosity >= UtilizerVerbosity.HIGH:
                logger.info(f"after unsqueeze in order to add a time dimension: {x.shape}")

        # 0 batch * population_k, 1 output_shape, 2 output_time
        
        x = x.reshape(x.shape[0] // self.population_k, self.population_k, *self.model_output_shape, x.shape[-1])
        if self.verbosity >= UtilizerVerbosity.HIGH:
            logger.info(f"after reshape in order to readd the population dimension: {x.shape}")

        # 0 batch, 1 population_k, 2 output_shape, 3 output_time

        x_before_decoding = x

        x = self.decoding(x)

        # returning probabilities by default, the user can apply threshold if needed
        # 0 batch, 1 output_shape, 2 output_time
        return x_before_decoding, x

    def decoding(self, x, binarized_decoding=None, binarized_decoding_threshold=None):
        binarization_threshold = binarized_decoding_threshold if binarized_decoding_threshold is not None else self.binarized_decoding_threshold

        # 0 batch, 1 population_k, 2 output_shape, 3 output_time
        
        x = x[:, :, :, -self.effective_decoding_time_from_end:]

        if self.use_linear_decoding_layer:
            if self.model_uses_time and self.population_k * self.model_output_dim > 1:
                # TODO: implement one day
                raise ValueError("linear decoding layer not implemented for model with time and population * output_dim > 1")

            x = x.reshape(x.shape[0], self.population_k * self.model_output_dim * self.effective_decoding_time_from_end)
            if self.verbosity >= UtilizerVerbosity.HIGH:
                logger.info(f"after flatten for decoding: {x.shape}")

            # 0 batch, 1 population_k * output_dim * output_time

            if not (self.binarize_before_linear_decoding_layer or binarized_decoding):
                if self.verbosity >= UtilizerVerbosity.HIGH:
                    logger.info(f"summing probabilities is just wrong, so although not (self.binarize_before_linear_decoding_layer or binarized_decoding), we'll do differentiable binarization")

            if self.binarize_before_linear_decoding_layer or binarized_decoding:
                # no sigmoid is needed, because we assume it was done by the model

                if self.differentiable_binarization_threshold_surrogate_spike:
                    spike_fn = get_surrogate_spike_gradient(self.differentiable_binarization_threshold_surrogate_spike_beta)
                    x = spike_fn(x-binarization_threshold).float()
                elif self.differentiable_binarization_threshold_straight_through:
                    # TODO: no binarization_threshold is used here?
                    p = x

                    randoms = np.random.rand(*p.shape)

                    y_hard = randoms < p.detach().cpu().numpy()

                    y_soft = (torch.from_numpy(y_hard).type(torch.float).to(p.device) - p).detach() + p

                    x = y_soft
                else:
                    raise ValueError("no differentiable binarization threshold method selected")
                if self.verbosity >= UtilizerVerbosity.HIGH:
                    logger.info(f"after binarization: {x.shape}")

            if self.verbosity >= UtilizerVerbosity.LOW:
                x_shape_before = tuple(x.shape)
                linear_decoding_layer_shape = tuple(self.linear_decoding_layer.get_weight_shape())

            # TODO?
            self.linear_decoding_layer = self.linear_decoding_layer.to(x.device)                

            x = self.linear_decoding_layer(x)

            if self.verbosity >= UtilizerVerbosity.LOW:
                x_shape_after = tuple(x.shape)
                logger.info(f"linear_decoding_layer ({linear_decoding_layer_shape}): {x_shape_before} -> {x_shape_after}")

            # 0 batch, 1 self.ds_output_dim * self.ds_output_time

            x = x.reshape(x.shape[0], self.ds_output_dim, self.ds_output_time)
            if self.verbosity >= UtilizerVerbosity.HIGH:
                logger.info(f"after flatten back: {x.shape}")

            # 0 batch, 1 output_dim, 2 output_time

            if self.use_softmax_after_linear_decoding_layer:
                x = F.softmax(x, dim=1)
                if self.verbosity >= UtilizerVerbosity.HIGH:
                    logger.info(f"after softmax: {x.shape}")
                
            x = x.reshape(x.shape[0], *self.ds_output_shape, self.ds_output_time)
            if self.verbosity >= UtilizerVerbosity.HIGH:
                logger.info(f"after unflatten output_dim: {x.shape}")

            # 0 batch, 1 output_shape, 2 output_time

        elif self.decoding_type == DecodingType.MAX_POOLING:
            # 0 batch, 1 population_k, 2 output_shape, 3 output_time
            if self.model_uses_time and self.population_k * self.model_output_dim > 1:
                # TODO: implement one day
                raise ValueError("max pooling decoding not implemented for model with time and population * output_dim > 1")
            if self.ds_output_shape != (2,) or self.ds_output_time != 1:
                # TODO: implement one day
                print(f"ds_output_shape: {self.ds_output_shape}, ds_output_time: {self.ds_output_time}")
                raise ValueError("max pooling decoding not implemented for ds_output_shape != (2,) or ds_output_time != 1")

            x = x.reshape(x.shape[0], self.population_k * self.model_output_dim * self.effective_decoding_time_from_end)
            if self.verbosity >= UtilizerVerbosity.HIGH:
                logger.info(f"after flatten for decoding: {x.shape}")

            # 0 batch, 1 population_k * output_dim * output_time

            x = torch.max(x, dim=1)[0]
            if self.verbosity >= UtilizerVerbosity.HIGH:
                logger.info(f"after max pooling: {x.shape}")

            # 0 batch

            x = x.unsqueeze(1)
            if self.verbosity >= UtilizerVerbosity.HIGH:
                logger.info(f"after unsqueeze in order to add a time dimension: {x.shape}")

            if binarized_decoding:
                x = (x > binarization_threshold).float()
                if self.verbosity >= UtilizerVerbosity.HIGH:
                    logger.info(f"after binarization: {x.shape}")

            temp_x = torch.zeros((x.shape[0], 2, 1)).to(x.device)
            temp_x[:,0,:] = 1 - x
            temp_x[:,1,:] = x
            x = temp_x
            if self.verbosity >= UtilizerVerbosity.HIGH:
                logger.info(f"after reshape to one hot: {x.shape}")

            # 0 batch, 1 output_shape, 2 output_time

        elif self.decoding_type == DecodingType.SUM_POOLING:
            # 0 batch, 1 population_k, 2 output_shape, 3 output_time
            if self.model_uses_time and self.population_k * self.model_output_dim > 1:
                # TODO: implement one day
                raise ValueError("sum pooling decoding not implemented for model with time and population * output_dim > 1")
            if self.ds_output_time != 1:
                # TODO: implement one day
                raise ValueError("sum pooling decoding not implemented for ds_output_time != 1")
            if len(self.ds_output_shape) != 1:
                # TODO: implement one day
                raise ValueError("sum pooling decoding not implemented for len(ds_output_shape) != 1")

            x = x.reshape(x.shape[0], self.population_k * self.model_output_dim * self.effective_decoding_time_from_end)
            if self.verbosity >= UtilizerVerbosity.HIGH:
                logger.info(f"after flatten for decoding: {x.shape}")

            # 0 batch, 1 population_k * output_dim * output_time

            if not binarized_decoding:
                if self.verbosity >= UtilizerVerbosity.HIGH:
                    logger.info(f"summing probabilities is just wrong, so although not binarized_decoding, we'll do differentiable binarization")

            if self.differentiable_binarization_threshold_surrogate_spike:
                spike_fn = get_surrogate_spike_gradient(self.differentiable_binarization_threshold_surrogate_spike_beta)
                x = spike_fn(x-binarization_threshold).float()
            elif self.differentiable_binarization_threshold_straight_through:
                # TODO: no binarization_threshold is used here?
                p = x

                randoms = np.random.rand(*p.shape)

                y_hard = randoms < p.detach().cpu().numpy()

                y_soft = (torch.from_numpy(y_hard).type(torch.float).to(p.device) - p).detach() + p

                x = y_soft
            else:
                raise ValueError("no differentiable binarization threshold method selected")
            if self.verbosity >= UtilizerVerbosity.HIGH:
                logger.info(f"after binarization: {x.shape}")
    
            x = torch.sum(x, dim=1)
            if self.verbosity >= UtilizerVerbosity.HIGH:
                logger.info(f"after sum pooling: {x.shape}")

            # 0 batch

            # TODO: this is not differentiable, so keeping the sum for now
            # x = torch.round(x).to(torch.long)
            # x = F.one_hot(x, num_classes=self.ds_output_shape[0]).to(x.device)
            # if self.verbosity >= UtilizerVerbosity.HIGH:
            #     logger.info(f"after reshape to one hot: {x.shape}")

            x = x.unsqueeze(-1)
            if self.verbosity >= UtilizerVerbosity.HIGH:
                logger.info(f"after unsqueeze in order to add a time dimension: {x.shape}")                

            # 0 batch, 1 output_shape, 2 output_time            

        elif self.decoding_type == DecodingType.NONE:
            if self.population_k != 1:
                raise ValueError("population_k != 1 but decoding type is NONE")
            
            x = x.squeeze(1)
            if self.verbosity >= UtilizerVerbosity.HIGH:
                logger.info(f"after squeeze in order to remove the population_k dimension: {x.shape}")

            # 0 batch, 1 output_shape, 2 output_time
        else:
            raise ValueError("unknown decoding type")

        # returning probabilities by default, the user can apply threshold if needed
        # 0 batch, 1 output_shape, 2 output_time
        return x

    def configure_optimizers(self):
        if self.optimizer_type == OptimizerType.ADAM:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer_type == OptimizerType.SGD:
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)

        optimizers = [optimizer]

        lr_schedulers = []
        if self.step_lr:
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
            lr_schedulers = [lr_scheduler]
            logger.info(f"using step lr scheduler with step_size={30} and gamma={0.5}")

        return optimizers, lr_schedulers

    def loss(self, z, y):
        loss = 0

        z_before_decoding = z[0]

        if self.require_no_spikes_before_decoding_time:
            # TODO: better name? no_spikes_before_decoding_time_loss

            # 0 batch, 1 population_k, 2 output_shape, 3 output_time
            z_before_decoding_before_decoding_time = z_before_decoding[:, :, :, :-self.effective_decoding_time_from_end]

            if z_before_decoding_before_decoding_time.shape[3] == 0:
                # there is nothing to put constraints on
                pass

            else:
                # loss is just the max of the spike probability before the decoding time
                # o(z_before_decoding_before_decoding_time.shape[1] * z_before_decoding_before_decoding_time.shape[2] * z_before_decoding_before_decoding_time.shape[3])
                before_decoding_loss = torch.sum(torch.max(z_before_decoding_before_decoding_time, dim=1)[0])

                # TODO: remove
                # print("before_decoding_loss", before_decoding_loss)

                # TODO: should have a no_spikes_before_decoding_time_loss_weight argument for this
                before_decoding_loss *= 0.1

                # print("after weight before_decoding_loss", before_decoding_loss)
                
                # # TODO: normalize by the size?
                # size_of_before_decoding_loss = z_before_decoding_before_decoding_time.shape[1] * z_before_decoding_before_decoding_time.shape[2] * z_before_decoding_before_decoding_time.shape[3]
                # before_decoding_loss /= size_of_before_decoding_loss

                # print("size_of_before_decoding_loss", size_of_before_decoding_loss)

                # print("after normalization before_decoding_loss", before_decoding_loss)

                loss += before_decoding_loss

        # after decoding
        z = z[1]
        # 0 batch, 1 output_shape, 2 decoding_time

        # TODO: enable more losses one day

        if self.ds_output_time != 1:
            if len(y.shape) == 2:
                # make it 3D
                y = y.unsqueeze(1)
            
            if y.shape[1] == 2:
                # already one hot, nothing to do
                pass

            elif y.shape[1] == 1:
                y = y.squeeze(1)

                y_one_hot = torch.zeros((y.shape[0], 2, y.shape[1]))
                y_one_hot[:,0,:] = 1 - y
                y_one_hot[:,1,:] = y

                y = y_one_hot

            else:
                raise ValueError("y shape[1] != 1 or 2")

            if len(z.shape) == 2:
                raise ValueError("has no output_shape after decoding (probably decoding was sum_pooling, for which time must be 1 right now)")
            
            # 0 batch, 1 output_shape, 2 output_time

            if z.shape[1] != 1:
                raise ValueError("ds_output_time != 1, and output_shape != 1")

            # TODO: does this assumptions always hold?
            # assuming we have probabilities, 
            # so for every time point, we can use cross entropy, under the assumption that this is a classification problem                

            z = z.squeeze(1)

            z_one_hot = torch.zeros((z.shape[0], 2, z.shape[1]))
            z_one_hot[:,0,:] = (1 - z)
            z_one_hot[:,1,:] = z

            z = z_one_hot
            
            # o(1)
            the_cross_entropy_loss = self.cross_entropy(z.float(), y.float())
            
            loss += the_cross_entropy_loss
        
        else:
            if z.shape[-1] != 1:
                raise ValueError("ds_output_time == 1 but time dimension after decoding is not 1")

            # There is no time, so we don't need the last dimension
            z = z.squeeze(-1)
            # 0 batch, 1 output_shape

            if len(z.shape) == 2:
                # (batch, one_hot_dim)

                # TODO: does this assumptions always hold?
                # assuming we have probabilities,
                # so we can use cross entropy, under the assumption that this is a classification problem

                # o(1)
                the_cross_entropy_loss = self.cross_entropy(z.float(), y.float())

                loss += the_cross_entropy_loss
            
            else:
                # print("y", y)
                # print("z", z)

                the_mse_loss = self.mse(z.float(), y.float())

                loss += the_mse_loss

        if self.wiring_weight_l1_reg > 0:
            loss += self.wiring_weight_l1_reg * torch.sum(torch.abs(self.wiring_layer.get_weights()))

        if self.wiring_weight_l2_reg > 0:
            loss += self.wiring_weight_l2_reg * torch.sum(self.wiring_layer.get_weights() ** 2)

        return y, loss

    def predict(self, z, binarized_decoding_on_predict=None, binarized_decoding_threshold=None,
     binarized_output_on_predict=None, binarized_output_threshold=None, argmax_on_predict=None):
        with torch.no_grad():
            if binarized_decoding_on_predict is None:
                binarized_decoding_on_predict = self.binarized_decoding_on_predict

            if binarized_decoding_threshold is None:
                binarized_decoding_threshold = self.binarized_decoding_threshold

            if binarized_output_on_predict is None:
                binarized_output_on_predict = self.binarized_output_on_predict

            if binarized_output_threshold is None:
                binarized_output_threshold = self.binarized_output_threshold

            if argmax_on_predict is None:
                argmax_on_predict = self.argmax_on_predict

            z_before_decoding, z = z[0], z[1]
            # 0 batch, 1 output_shape, 2 output_time

            if binarized_decoding_on_predict:
                z = self.decoding(z_before_decoding, binarized_decoding=True, binarized_decoding_threshold=binarized_decoding_threshold)

            if binarized_output_on_predict:
                z = (z > binarized_output_threshold).float()

            if self.ds_output_time != 1:
                if z.shape[1] != 1:
                    raise ValueError("ds_output_time != 1 but output_shape != 1")
                # replace the 1 output_shape with 2 output_shape

                # 0 batch, 1 output_shape, 2 output_time
                z = z.squeeze(1)
                # 0 batch, 1 output_time

                z_one_hot = torch.zeros((z.shape[0], 2, z.shape[1]))
                z_one_hot[:,0,:] = (1 - z)
                z_one_hot[:,1,:] = z

                z = z_one_hot
            else:
                # There is no time, so we don't need the last dimension
                # 0 batch, 1 output_shape, 2 output_time
                z = z.squeeze(-1)
                # 0 batch, 1 output_shape

            if len(z.shape) == 2:
                if argmax_on_predict:
                    z = torch.argmax(z, dim=1)

            if self.decoding_type == DecodingType.SUM_POOLING:
                # print("z", z)
                pass

            return z

    def update_metrics(self, accuracy, auc, mae, pred, y, argmax_on_predict=None):
        if argmax_on_predict is None:
            argmax_on_predict = self.argmax_on_predict

        if len(y.shape) == 2:
            # classification
            if argmax_on_predict:
                y = torch.argmax(y, dim=1)
            accuracy(pred, y)
            auc(pred, y)
        elif len(y.shape) == 1:
            # regression
            auc(pred, y) # TODO?
            mae(pred, y)
        else:
            raise ValueError("y shape is not 1 or 2")
        
        return y

    def calculate_metrics(self, data_loader, binarized_decoding_on_predict=None, binarized_decoding_threshold=None,
     binarized_output_on_predict=None, binarized_output_threshold=None, argmax_on_predict=None, save_to_folder=None):
        with torch.no_grad():
            # TODO: binary always?
            accuracy = torchmetrics.Accuracy(task='binary')
            auc = torchmetrics.AUROC(task='binary')
            mae = torchmetrics.MeanAbsoluteError()
            for batch_id, batch in enumerate(data_loader):
                x, y = batch
                y_extra = None
                if self.ds_extra_label_information:
                    y, y_extra = y

                z = self(x)
                pred = self.predict(z, binarized_decoding_on_predict=binarized_decoding_on_predict,
                 binarized_decoding_threshold=binarized_decoding_threshold, binarized_output_on_predict=binarized_output_on_predict,
                   binarized_output_threshold=binarized_output_threshold, argmax_on_predict=argmax_on_predict)
                y_for_accuracy = self.update_metrics(accuracy, auc, mae, pred, y, argmax_on_predict=argmax_on_predict)

                if save_to_folder is not None:
                    batch_folder = os.path.join(save_to_folder, f'batch{batch_id}')
                    os.makedirs(batch_folder, exist_ok=True)
                    for sample_id in range(x.shape[0]):
                        sample_folder = os.path.join(batch_folder, f'sample{sample_id}')
                        os.makedirs(sample_folder, exist_ok=True)
                        
                        # saving the padded x, not the original x
                        x_sample = self.padded_first_x[sample_id].detach().cpu().numpy()
                        # x_sample = x[sample_id].detach().cpu().numpy()

                        try:
                            sparse.save_npz(os.path.join(sample_folder, 'x.npz'), sparse.csr_matrix(x_sample))
                        except:
                            np.save(os.path.join(sample_folder, 'x.npy'), x_sample)

                        
                        y_sample = y[sample_id].detach().cpu().numpy()
                        np.save(os.path.join(sample_folder, 'y.npy'), y_sample)
                        if self.ds_extra_label_information:
                            y_extra_sample = y_extra[sample_id].detach().cpu().numpy()
                            np.save(os.path.join(sample_folder, 'y_extra.npy'), y_extra_sample)

                        z_before_decoding_sample = z[0][sample_id].detach().cpu().numpy()
                        np.save(os.path.join(sample_folder, 'z_before_decoding.npy'), z_before_decoding_sample)

                        z_sample = z[1][sample_id].detach().cpu().numpy()
                        np.save(os.path.join(sample_folder, 'z.npy'), z_sample)
                        
                        pred_sample = pred[sample_id].detach().cpu().numpy()
                        np.save(os.path.join(sample_folder, 'pred.npy'), pred_sample)

                logger.info(f"In calculate_metrics, batch {batch_id}, current accuracy: {accuracy.compute()}, current auc: {auc.compute()}, current mae: {mae.compute()}")
                self.plot_batch("calculate_metrics", x, y, y_extra, y_for_accuracy, z, pred, 0, batch_id)

            metrics = {'accuracy': accuracy.compute(), 'auc': auc.compute(), 'mae': mae.compute()}
            return metrics

    def on_train_epoch_start(self):
        if self.use_wiring_layer:
            self.wiring_layer.update_train_epoch(self.current_epoch)

        if self.use_population_masking_layer:
            self.population_masking_layer.update_train_epoch(self.current_epoch)

        if self.use_linear_decoding_layer:
            self.linear_decoding_layer.update_train_epoch(self.current_epoch)

    def on_train_batch_start(self, batch, batch_idx):
        if self.use_wiring_layer:
            self.wiring_layer.update_train_batch(batch_idx)

        if self.use_population_masking_layer:
            self.population_masking_layer.update_train_batch(batch_idx)

        if self.use_linear_decoding_layer:
            self.linear_decoding_layer.update_train_batch(batch_idx)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_extra = None
        if self.ds_extra_label_information:
            y, y_extra = y
        z = self(x)
        y_for_loss, loss = self.loss(z, y)
        self.log('train_loss', loss)
        pred = self.predict(z)
        y_for_accuracy = self.update_metrics(self.train_accuracy, self.train_auc, self.train_mae, pred, y_for_loss)
        self.log('train_accuracy_step', self.train_accuracy)
        self.log('train_auc_step', self.train_auc)
        self.log('train_mae_step', self.train_mae)

        self.plot_batch("train", x, y, y_extra, y_for_accuracy, z, pred, loss, batch_idx)

        return loss

    def on_train_epoch_end(self):
        with torch.no_grad():
            self.log('train_accuracy_epoch', self.train_accuracy)
            self.last_train_accuracy = self.train_accuracy.compute()
            self.log('train_auc_epoch', self.train_auc)
            self.last_train_auc = self.train_auc.compute()
            self.log('train_mae_epoch', self.train_mae)
            self.last_train_mae = self.train_mae.compute()
            if self.enable_progress_bar:
                print("\n")
            logger.info(f"on epoch {self.current_epoch}, train accuracy is {self.last_train_accuracy}")
            logger.info(f"on epoch {self.current_epoch}, train auc is {self.last_train_auc}")
            logger.info(f"on epoch {self.current_epoch}, train mae is {self.last_train_mae}")

            if self.last_train_accuracy > self.maximum_train_accuracy:
                self.maximum_train_accuracy = self.last_train_accuracy
                logger.info(f"on epoch {self.current_epoch}, reached new maximum train accuracy {self.maximum_train_accuracy}")

            if self.last_train_auc > self.maximum_train_auc:
                self.maximum_train_auc = self.last_train_auc
                logger.info(f"on epoch {self.current_epoch}, reached new maximum train auc {self.maximum_train_auc}")

            if self.last_train_mae < self.minimum_train_mae:
                self.minimum_train_mae = self.last_train_mae
                logger.info(f"on epoch {self.current_epoch}, reached new minimum train mae {self.minimum_train_mae}")

    def on_validation_epoch_start(self):
        if self.use_wiring_layer:
            self.wiring_layer.update_train_epoch(None)
            self.wiring_layer.update_train_batch(None)

        if self.use_population_masking_layer:
            self.population_masking_layer.update_train_epoch(None)
            self.population_masking_layer.update_train_batch(None)

        if self.use_linear_decoding_layer:
            self.linear_decoding_layer.update_train_epoch(None)
            self.linear_decoding_layer.update_train_batch(None)

    def validation_step(self, val_batch, batch_idx):
        with torch.no_grad():
            x, y = val_batch
            y_extra = None
            if self.ds_extra_label_information:
                y, y_extra = y
            z = self(x)
            y_for_loss, loss = self.loss(z, y)
            self.log('valid_loss', loss)
            pred = self.predict(z)
            y_for_accuracy = self.update_metrics(self.valid_accuracy, self.valid_auc, self.valid_mae, pred, y_for_loss)
            self.log('valid_accuracy_step', self.valid_accuracy)
            self.log('valid_auc_step', self.valid_auc)
            self.log('valid_mae_step', self.valid_mae)

            self.plot_batch("valid", x, y, y_extra, y_for_accuracy, z, pred, loss, batch_idx)
            
            return loss

    def on_validation_epoch_end(self):
        with torch.no_grad():
            self.log('valid_accuracy_epoch', self.valid_accuracy)
            self.last_valid_accuracy = self.valid_accuracy.compute()
            self.log('valid_auc_epoch', self.valid_auc)
            self.last_valid_auc = self.valid_auc.compute()
            self.log('valid_mae_epoch', self.valid_mae)
            self.last_valid_mae = self.valid_mae.compute()
            if self.enable_progress_bar:
                print("\n")
            logger.info(f"on epoch {self.current_epoch}, validation accuracy is {self.last_valid_accuracy}")
            logger.info(f"on epoch {self.current_epoch}, validation auc is {self.last_valid_auc}")
            logger.info(f"on epoch {self.current_epoch}, validation mae is {self.last_valid_mae}")

            if self.last_valid_accuracy > self.maximum_valid_accuracy:
                self.maximum_valid_accuracy = self.last_valid_accuracy
                logger.info(f"on epoch {self.current_epoch}, reached new maximum validation accuracy {self.maximum_valid_accuracy}")

            if self.last_valid_auc > self.maximum_valid_auc:
                self.maximum_valid_auc = self.last_valid_auc
                logger.info(f"on epoch {self.current_epoch}, reached new maximum validation auc {self.maximum_valid_auc}")

            if self.last_valid_mae < self.minimum_valid_mae:
                self.minimum_valid_mae = self.last_valid_mae
                logger.info(f"on epoch {self.current_epoch}, reached new minimum validation mae {self.minimum_valid_mae}")                
            
            self.log('epoch', self.current_epoch)

    def on_before_optimizer_step(self, optimizer):
        pass
        # # Compute the 2-norm for each layer
        # # If using mixed precision, the gradients are already unscaled here
        # print("on_before_optimizer_step start")
        # if self.use_wiring_layer:
        #     norms = grad_norm(self.wiring_layer, norm_type=2)
        #     print(norms)
        #     # self.log_dict(norms)
        #     for key, value in norms.items():
        #         print(f'wiring_layer_grad_norm_{key}', value)
        #         # self.log(f'wiring_layer_grad_norm_{key}', value)
        # print("on_before_optimizer_step end")

    def plot_batch(self, state, x, y, y_extra, y_for_accuracy, z, pred, loss, batch_idx):
        if not self.enable_plotting:
            return

        should_plot_epoch = False
        if state == "train":
            if self.plot_train_every_in_epochs:
                if (isinstance(self.plot_train_every_in_epochs, int) and self.current_epoch % self.plot_train_every_in_epochs == 0) or\
                    isinstance(self.plot_train_every_in_epochs, list) and self.current_epoch in self.plot_train_every_in_epochs:
                    should_plot_epoch = True
        elif state == "valid":
            if self.plot_valid_every_in_epochs:
                if (isinstance(self.plot_valid_every_in_epochs, int) and self.current_epoch % self.plot_valid_every_in_epochs == 0) or\
                    isinstance(self.plot_valid_every_in_epochs, list) and self.current_epoch in self.plot_valid_every_in_epochs:
                    should_plot_epoch = True
        elif state == "calculate_metrics":
            should_plot_epoch = True
        if not should_plot_epoch:
            return

        should_plot_batch = False
        if state == "train":
            if self.plot_train_every_in_batches:
                if (isinstance(self.plot_train_every_in_batches, int) and batch_idx % self.plot_train_every_in_batches == 0) or\
                    isinstance(self.plot_train_every_in_batches, list) and batch_idx in self.plot_train_every_in_batches:
                    should_plot_batch = True
        elif state == "valid":
            if self.plot_valid_every_in_batches:
                if (isinstance(self.plot_valid_every_in_batches, int) and batch_idx % self.plot_valid_every_in_batches == 0) or\
                    isinstance(self.plot_valid_every_in_batches, list) and batch_idx in self.plot_valid_every_in_batches:
                    should_plot_batch = True
        elif state == "calculate_metrics":
            should_plot_batch = True
        if not should_plot_batch:
            return

        current_plot_folder = f"{self.plots_folder}/{state}"
        os.makedirs(current_plot_folder, exist_ok=True)

        if self.enable_progress_bar:
            print("\n")
        logger.info(f"Plotting for state {state}, epoch {self.current_epoch}, batch {batch_idx}")

        self.plot_function(self, x, y, y_extra, y_for_accuracy, z, pred, loss, state, self.current_epoch, batch_idx, current_plot_folder)
