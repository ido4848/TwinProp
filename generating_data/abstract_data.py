import numpy as np
import sys
import pathlib
import matplotlib.pyplot as plt
import torch
import os
import pickle
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from sklearn import linear_model
import logging

sys.path.append(str(pathlib.Path(__file__).parent.absolute()))
sys.path.append(str(pathlib.Path(__file__).parent.parent.absolute()))

from generating_data import sequences

from utils.utils import setup_logger

logger = logging.getLogger(__name__)

def init_input(P, count_values_per_dimension, count_dimensions, stim_on, stim_off, r_mean, r_max, num_t, s,
 shuffle_inds=True, mutually_exclusive_synapses_for_each_pattern=False, arange_spikes=False):
    """
    Initialise input rates and spike time sequences for feature-binding task.

    Parameters
    ----------
    P : dict
        model parameters
    num_patterns : int
        number of input patterns to be classified
    stim_on, stim_off : int
        time of stimulus onset and termination (ms)
    r_mean : float
        average presynaptic population rate (Hz)
    r_max : float
        time averaged input rate to active synapses
    num_t : int
        number of precisely timed events per active synapse
    s : float
        interpolates between rate (s=0) and temporal (s=1) input signals (mostly
        unused parameter -- to be removed)

    Returns
    -------
    rates_e, rates_i : list
        excitatory and inhibitory input rates for all patterns
    S_E, S_I : list
        times of precisely timed events for all patterns
    """
    N_e, N_i = P['N_e'], P['N_i']
    N_e_bias, N_i_bias = P['N_e_bias'], P['N_i_bias']

    if N_e_bias is not None and N_e_bias > 0:
        assert N_e_bias < N_e, "N_e_bias must be less than N_e"
        N_e -= N_e_bias
    if N_i_bias is not None and N_i_bias > 0:
        assert N_i_bias < N_i, "N_i_bias must be less than N_i"
        N_i -= N_i_bias

    ind_e = np.arange(N_e)
    ind_i = np.arange(N_i)
    if shuffle_inds:
        np.random.shuffle(ind_e)
        np.random.shuffle(ind_i)
    rates_e, rates_i, n_es, n_is, ne_subs, ni_subs = sequences.assoc_rates(count_values_per_dimension, count_dimensions, N_e, N_i, r_mean, r_max,
     mutually_exclusive_synapses_for_each_pattern, shuffle_inds)
    rates_e = [r[ind_e] for r in rates_e]
    rates_i = [r[ind_i] for r in rates_i]
    if s > 0:
        S_E, S_I, n_es, n_is, ne_subs, ni_subs = sequences.assoc_seqs(count_values_per_dimension, count_dimensions, N_e, N_i, stim_on, stim_off,
                                        num_t, mutually_exclusive_synapses_for_each_pattern, shuffle_inds, ne_subs, ni_subs, arange_spikes)
        S_E = [s[ind_e] for s in S_E]
        S_I = [s[ind_i] for s in S_I]
        for s_e, r_e in zip(S_E, rates_e):
            s_e[r_e == 0] = np.inf
        for s_i, r_i in zip(S_I, rates_i):
            s_i[r_i == 0] = np.inf
    else:
        S_E, S_I = sequences.build_seqs(count_values_per_dimension, count_dimensions, N_e, N_i, stim_on, stim_off, 0)

    e_counter = 0
    e_axons = []
    for n_e in n_es:
        e_axons.append(ind_e[e_counter:e_counter + n_e])
        e_counter += n_e
    i_counter = 0
    i_axons = []
    for n_i in n_is:
        i_axons.append(ind_i[i_counter:i_counter + n_i])
        i_counter += n_i

    # add bias rates and sequences
    # TODO: possibly would want more sophisticated mean calculation (relative to others?)
    # TODO: possibly would want more sophisticated bias options (opt encoding?)

    if N_e_bias is not None and N_e_bias > 0:
        for i in range(len(rates_i)):
            rates_e[i] = np.append(rates_e[i], np.ones(N_e_bias) * r_mean * 1e-3)
            array_to_append = [np.inf] if len(S_E[i][0]) > 0 else []
            S_E[i] = np.append(S_E[i], np.array([array_to_append for _ in range(N_e_bias)]), axis=0)
        e_axons.append(np.arange(N_e, N_e + N_e_bias))

    if N_i_bias is not None and N_i_bias > 0:
        for i in range(len(rates_i)):
            rates_i[i] = np.append(rates_i[i], np.ones(N_i_bias) * r_mean * 1e-3)
            array_to_append = [np.inf] if len(S_I[i][0]) > 0 else []
            S_I[i] = np.append(S_I[i], np.array([array_to_append for _ in range(N_i_bias)]), axis=0)
        i_axons.append(np.arange(N_i, N_i + N_i_bias))    

    return rates_e, rates_i, S_E, S_I, e_axons, i_axons, ne_subs, ni_subs

def generate_general_classification_tensor(d, n, c=2):
    # generate c valued tensor with d dimensions each of them having n different values such that the entropy of the tensor is maximized
    if d < 2:
        raise ValueError("d must be at least 2")
    if n < 2:
        raise ValueError("n must be at least 2")

    if d == 2:
        mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                mat[i, j] = i%c ^ j%c
        return mat

    mat = np.zeros([n]*d)
    for i in range(n):
        d_minus_1_tensor = generate_general_classification_tensor(d-1, n, c)
        i_mod_c_tensor = np.zeros(d_minus_1_tensor.shape) + i%c
        mat[i] = np.bitwise_xor(d_minus_1_tensor==1, i_mod_c_tensor==1)
    return mat

def init_labels(count_values_per_dimension, count_dimensions, random_labels=False, random_permutation_labels=False):
    """
    Initialise classification labels. Nonlinear contingencies for 2x2 task,
    random assignment for 3x3 and above if random_labels is True, else checkerboard.

    Parameters
    ----------
    num_patterns : int
        number of input patterns to be classified

    Returns
    -------
    L : ndarray
        classification labels (+1/-1 for preferred/non-preferred input patterns).
    """
    gct = generate_general_classification_tensor(count_dimensions, count_values_per_dimension, 2)
    
    L = 2 * generate_general_classification_tensor(count_dimensions, count_values_per_dimension, 2) - 1
    L = L.flatten()

    if random_labels:
        # L = sequences.assign_labels(num_patterns)
        L = np.random.choice([-1, 1], len(L))

    if random_permutation_labels:
        L = np.random.permutation(L)

    logger.info(f"Initiated Labels: {L}")
    for i, val in enumerate(L):
        logger.info(f"{i}={i:0{count_dimensions}b} -> {val}")

    # train perceptron on labels
    len_labels = len(L)
    y = []
    X = []
    for i, val in enumerate(L):
        i_in_binary = f'{i:0{count_dimensions}b}'
        sample = []
        for j in i_in_binary:
            sample.append(int(j))
        X.append(sample)
        y.append(val)

    lr = linear_model.LogisticRegression()
    lr.fit(np.array(X), y)
    perceptron_score = lr.score(np.array(X), y)
    logger.info(f"Perceptron score: {perceptron_score}")

    return L, perceptron_score

def pad_S(S0):
    if len(S0) == 0:
        return np.array([])
    l = np.max([len(s) for s in S0])
    S = np.full((len(S0), l), np.inf)
    for k, s in enumerate(S0):
        S[k, :len(s)] = s
    return S

def round_to_nearest_100(x):
    return x + 100*(x%100>0) - x%100

def get_get_abstract_dataset(count_values_per_dimension=2, count_dimensions=2, input_encoding='opt',
 stim_dur=400, stim_on=0, r_mean=2.5, r_max=20, t_on=0, r_0=1.25, jitter=2.5, N_e=800, N_i=200,
  average_firing_rate_per_axon=None, shuffle_inds=True, mutually_exclusive_synapses_for_each_pattern=False, arange_spikes=False, 
  random_labels=False, random_permutation_labels=False, trial=0, N_e_bias=None, N_i_bias=None, num_t=None):
    param_sets = {'rate':[40., 0, 0.], 'temp':[2.5, 1, 1.], 'opt':[20., 1, 1.]}
    param_set_r_max, param_set_num_t, s = param_sets[input_encoding]
    
    if average_firing_rate_per_axon is not None:
        r_max = average_firing_rate_per_axon
        r_mean = average_firing_rate_per_axon
    elif r_max is None:
        r_max = param_set_r_max

    if num_t is None:
        num_t = param_set_num_t
    
    stim_off = stim_on + stim_dur # stimulus off
    t_off = stim_on						# background off
    temporal_extent = stim_off
    if num_t > 0:
        sigma = jitter*s*1e-3*r_max*(stim_off - stim_on)/num_t
    else:
        sigma = jitter
    P = {'N_e':N_e, 'N_i':N_i, 'N_e_bias':N_e_bias, 'N_i_bias': N_i_bias}

    num_patterns = count_values_per_dimension**count_dimensions

    def get_abstract_dataset():
        rates_e, rates_i, S_E, S_I, e_axons, i_axons, ne_subs, ni_subs = init_input(P, count_values_per_dimension, count_dimensions,
         stim_on, stim_off, r_mean, r_max, num_t, s, shuffle_inds, mutually_exclusive_synapses_for_each_pattern, arange_spikes)
        labels, perceptron_score = init_labels(count_values_per_dimension, count_dimensions, random_labels=random_labels, random_permutation_labels=random_permutation_labels)

        return r_max, r_mean, input_encoding, t_on, t_off, stim_on, stim_off, s, sigma, N_e, N_i, temporal_extent, rates_e, rates_i, S_E, S_I, e_axons, i_axons, ne_subs, ni_subs, labels, perceptron_score, num_t

    
    # dim_values_name = "x".join([str(count_values_per_dimension)]*count_dimensions)
    dim_values_name = f"{count_values_per_dimension}^{count_dimensions}"

    dataset_name = f"abstract_dataset_{dim_values_name}_{input_encoding}_stim_dur_{stim_dur}_stim_on_{stim_on}_r_mean_{r_mean}_r_max_{r_max}_N_e_{N_e}_N_i_{N_i}_shf_{shuffle_inds}_mut_{mutually_exclusive_synapses_for_each_pattern}_trial_{trial}"
    if N_e_bias is not None:
        dataset_name += f"_N_e_bias_{N_e_bias}"
    if N_i_bias is not None:
        dataset_name += f"_N_i_bias_{N_i_bias}"
    if arange_spikes:
        dataset_name += f"_arsp_{arange_spikes}"
    if random_labels:
        dataset_name += f"_randLbl_{random_labels}"
    if random_permutation_labels:
        dataset_name += f"_randPermLbl_{random_permutation_labels}"
    if num_t is not None:
        dataset_name += f"_num_t_{num_t}"
    dataset_params = {'N_e': N_e, 'N_i': N_i, 'stim_dur': stim_dur, 'r_max':r_max, 'r_mean':r_mean,
                       'N_e_bias':N_e_bias, 'N_i_bias': N_i_bias}
    return get_abstract_dataset, dataset_name, dim_values_name, dataset_params

class AbstractSampler:
    def __init__(self, r_0, input_encoding, t_on, t_off, stim_on, stim_off, s,
                  sigma, N_e, N_i, temporal_extent, rates_e, rates_i, S_E, S_I, labels,
                  N_e_bias=None, N_i_bias=None):
        self.r_0 = r_0
        self.input_encoding = input_encoding
        self.t_on = t_on
        self.t_off = t_off
        self.stim_on = stim_on
        self.stim_off = stim_off
        self.s = s
        self.sigma = sigma
        self.N_e = N_e
        self.N_i = N_i
        self.temporal_extent = temporal_extent
        self.rates_e = rates_e
        self.rates_i = rates_i
        self.S_E = S_E
        self.S_I = S_I
        self.labels = labels
        self.N_e_bias = N_e_bias
        if N_e_bias is None:
            self.N_e_bias = 0
        self.N_i_bias = N_i_bias
        if N_i_bias is None:
            self.N_i_bias = 0

        self.pre_syn = sequences.PreSyn(self.r_0, self.sigma)

    def sample(self, ind):
        # TODO: assuming bias axons are rate encoded
        e_s_lambda = lambda k: self.s if k < self.N_e - self.N_e_bias else 0
        S_e = [self.pre_syn.spike_train(self.t_on, self.t_off, self.stim_on, self.stim_off, e_s_lambda(k),
                            self.rates_e[ind][k], self.S_E[ind][k]) for k in range(self.N_e)]
        
        # TODO: assuming bias axons are rate encoded
        i_s_lambda = lambda k: self.s if k < self.N_i - self.N_i_bias else 0
        S_i = [self.pre_syn.spike_train(self.t_on, self.t_off, self.stim_on, self.stim_off, i_s_lambda(k),
                            self.rates_i[ind][k], self.S_I[ind][k]) for k in range(self.N_i)]                                
        # TODO: remove this old line
        # mat_size = int(round_to_nearest_100(max([a.max() for a in S_e if len(a) > 0])))
        mat_size = self.temporal_extent
        S_e = pad_S(S_e)
        S_i = pad_S(S_i)
        input_matrix = np.zeros((mat_size, self.N_e + self.N_i))
        for i in range(S_e.shape[0]):
            input_matrix[S_e[i][~np.isinf(S_e[i])].astype(int), i] = 1
        for i in range(S_i.shape[0]):
            input_matrix[S_i[i][~np.isinf(S_i[i])].astype(int), self.N_e+i] = 1

        input_matrix = input_matrix.transpose(1, 0)
        return input_matrix, self.labels[ind]

class AbstractRawSampler:
    def __init__(self, r_0, input_encoding, t_on, t_off, stim_on, stim_off, s,
                  sigma, N_e, N_i, temporal_extent, rates_e, rates_i, S_E, S_I, labels,
                  N_e_bias=None, N_i_bias=None):
        self.r_0 = r_0
        self.input_encoding = input_encoding
        self.t_on = t_on
        self.t_off = t_off
        self.stim_on = stim_on
        self.stim_off = stim_off
        self.s = s
        self.sigma = sigma
        self.N_e = N_e
        self.N_i = N_i
        self.temporal_extent = temporal_extent
        self.rates_e = rates_e
        self.rates_i = rates_i
        self.S_E = S_E
        self.S_I = S_I
        self.labels = labels
        self.N_e_bias = N_e_bias
        if N_e_bias is None:
            self.N_e_bias = 0
        self.N_i_bias = N_i_bias
        if N_i_bias is None:
            self.N_i_bias = 0

        self.X = []
        self.y = []

        # Right now it is here, but it might be in the sample function
        for ind in range(len(self.S_E)):
            mat = np.zeros((self.temporal_extent, self.N_e+self.N_i))

            for i in range(self.N_e):
                if self.input_encoding == 'opt':
                    if self.S_E[ind][i][0] < 0 or np.isinf(self.S_E[ind][i][0]):
                        continue
                    spike_indices = np.array(self.S_E[ind][i]).astype(int)
                    mat[spike_indices, i] = 1
                elif self.input_encoding == 'rate':
                    mat[:, i] = self.rates_e[ind][i]
                else:
                    raise ValueError('input_encoding must be "opt" or "rate"')

            for i in range(self.N_i):
                if self.input_encoding == 'opt':
                    if self.S_I[ind][i][0] < 0  or np.isinf(self.S_I[ind][i][0]):
                        continue
                    spike_indices = np.array(self.S_I[ind][i]).astype(int)
                    mat[spike_indices, self.N_e+i] = 1
                elif self.input_encoding == 'rate':
                    mat[:, self.N_e+i] = self.rates_i[ind][i]
                else:
                    raise ValueError('input_encoding must be "opt" or "rate"')
            
            mat = mat.transpose(1, 0)
            self.X.append(mat)
            self.y.append(self.labels[ind])

    def sample(self, ind):
        return self.X[ind], self.y[ind]

class AbstractDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, dataset_parent_path=None, dataset_path=None, one_hot_labels=True, samples_per_pattern=10, r_0=1.25, t_on=None,
                  t_off=None, jitter=None, sigma=None, use_raw_patterns=False, extra_label_information=False, *args, **kwargs):
        super(AbstractDataset, self).__init__()
        self.dataset_parent_path = dataset_parent_path
        self.dataset_path = dataset_path
        self.one_hot_labels = one_hot_labels
        self.use_raw_patterns = use_raw_patterns
        self.extra_label_information = extra_label_information
        self.get_dataset, self.dataset_name, self.dim_values_name, self.dataset_params = get_get_abstract_dataset(*args, **kwargs)
        # TODO: dataset_params vs dataset_dict? (the dict should only contain the heavy things [data] and not the params?)

        if self.dataset_path is None:
            if self.dataset_parent_path is None:
                raise ValueError("Either dataset_path or dataset_parent_path must be specified")
            self.dataset_path = os.path.join(self.dataset_parent_path, self.dataset_name)

        if not os.path.exists(self.dataset_path):
            logger.info(f"Dataset not found, generating... into {self.dataset_path}")
            os.makedirs(self.dataset_path)

            r_max, r_mean, input_encoding, t_on, t_off, stim_on, stim_off, s, sigma, N_e, N_i, temporal_extent, rates_e, rates_i, S_E, S_I, e_axons, i_axons, ne_subs, ni_subs, labels, perceptron_score, num_t = self.get_dataset()

            dataset_dict = {}
            dataset_dict["r_max"] = r_max
            dataset_dict["r_mean"] = r_mean
            dataset_dict["input_encoding"] = input_encoding
            dataset_dict["t_on"] = t_on
            dataset_dict["t_off"] = t_off
            dataset_dict["stim_on"] = stim_on
            dataset_dict["stim_off"] = stim_off
            dataset_dict["s"] = s
            dataset_dict["sigma"] = sigma
            dataset_dict["N_e"] = N_e
            dataset_dict["N_i"] = N_i
            dataset_dict["temporal_extent"] = temporal_extent
            dataset_dict["rates_e"] = rates_e
            dataset_dict["rates_i"] = rates_i
            dataset_dict["S_E"] = S_E
            dataset_dict["S_I"] = S_I
            dataset_dict["e_axons"] = e_axons
            dataset_dict["i_axons"] = i_axons
            dataset_dict["ne_subs"] = ne_subs
            dataset_dict["ni_subs"] = ni_subs
            dataset_dict["labels"] = labels
            dataset_dict["perceptron_score"] = perceptron_score
            dataset_dict["num_t"] = num_t

            self.dataset_dict = dataset_dict

            pickle.dump(dataset_dict, open(os.path.join(self.dataset_path, "dataset_dict.pkl"), "wb"))
        else:
            logger.info(f"Dataset found, loading from {self.dataset_path}")
            self.dataset_dict = pickle.load(open(os.path.join(self.dataset_path, "dataset_dict.pkl"), "rb"))

        self.count_labels = len(self.dataset_dict["labels"])
        self.count_samples = self.count_labels * samples_per_pattern
        self.samples_per_pattern = samples_per_pattern
        self.r_0 = r_0
        self.t_on = t_on
        if self.t_on is None:
            self.t_on = self.dataset_dict["t_on"]
        self.t_off = t_off        
        if self.t_off is None:
            self.t_off = self.dataset_dict["t_off"]

        self.input_encoding = self.dataset_dict["input_encoding"]
        self.s = self.dataset_dict["s"]
        self.stim_on = self.dataset_dict["stim_on"]
        self.stim_off = self.dataset_dict["stim_off"]

        if 'r_max' in self.dataset_dict:
            self.r_max = self.dataset_dict["r_max"]
        else:
            self.r_max = self.dataset_params["r_max"]

        if 'r_mean' in self.dataset_dict:
            self.r_mean = self.dataset_dict["r_mean"]
        else:
            self.r_mean = self.dataset_params["r_mean"]

        # allowing user to also explicitly set sigma instead of computing sigma from jitter
        self.sigma = sigma
        if self.sigma is None:
            self.jitter = jitter
            if self.jitter is not None:
                param_sets = {'rate':[40., 0, 0.], 'temp':[2.5, 1, 1.], 'opt':[20., 1, 1.]}
                _, num_t, _ = param_sets[self.input_encoding]
                if num_t > 0:
                    self.sigma = self.jitter*self.s*1e-3*self.r_max*(self.stim_off - self.stim_on)/num_t
                else:
                    self.sigma = jitter
            else:
                self.sigma = self.dataset_dict["sigma"]

        if self.use_raw_patterns:
            self.abstract_sampler = AbstractRawSampler(self.r_0, self.input_encoding,
                self.t_on, self.t_off,
                self.stim_on, self.stim_off,
                self.s, self.sigma,
                self.dataset_dict["N_e"], self.dataset_dict["N_i"],
                    self.dataset_dict["temporal_extent"], self.dataset_dict["rates_e"],
                    self.dataset_dict["rates_i"], self.dataset_dict["S_E"], self.dataset_dict["S_I"], self.dataset_dict["labels"]
                    , self.dataset_params["N_e_bias"], self.dataset_params["N_i_bias"])
        else:    
            self.abstract_sampler = AbstractSampler(self.r_0, self.input_encoding,
                self.t_on, self.t_off,
                self.stim_on, self.stim_off,
                self.s, self.sigma,
                self.dataset_dict["N_e"], self.dataset_dict["N_i"],
                    self.dataset_dict["temporal_extent"], self.dataset_dict["rates_e"],
                    self.dataset_dict["rates_i"], self.dataset_dict["S_E"], self.dataset_dict["S_I"], self.dataset_dict["labels"]
                    , self.dataset_params["N_e_bias"], self.dataset_params["N_i_bias"])

    def __len__(self):
        return self.count_samples

    def __getitem__(self, idx):
        ind = idx % self.count_labels
        samp, label = self.abstract_sampler.sample(ind)
        label = 0 if label == -1 else 1
        if self.one_hot_labels:
            one_hot_label = np.zeros(2)
            one_hot_label[label] = 1
            label = one_hot_label

        if self.extra_label_information:
            label = (label, ind)

        return samp, label

class NonSpikingAbstract():
    def __init__(self, valid_percentage=0.2, samples_per_pattern=10, jitter=None, valid_jitter=None, *args, **kwargs):
        self.valid_samples_per_pattern = int(samples_per_pattern * valid_percentage)
        self.train_samples_per_pattern = samples_per_pattern - self.valid_samples_per_pattern

        self.dataset_valid = AbstractDataset(dataset_parent_path='Data/',
         samples_per_pattern=self.valid_samples_per_pattern, jitter=valid_jitter, *args, **kwargs)

        self.dataset_train = AbstractDataset(dataset_parent_path='Data/',
            samples_per_pattern=self.train_samples_per_pattern, jitter=jitter, *args, **kwargs)

        self.dataset_params = self.dataset_train.dataset_params
        self.dim_values_name = self.dataset_train.dim_values_name

    def get_ds_shape(self):
        return ((self.dataset_params['N_e'] + self.dataset_params['N_i'], self.dataset_params['stim_dur']), 1, (2,), 1)

    def get_train_loader(self, batch_size, num_workers=1):
        return DataLoader(self.dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    def get_valid_loader(self, batch_size, num_workers=1):
        return DataLoader(self.dataset_valid, batch_size=batch_size, shuffle=False, num_workers=num_workers)

class SpikingAbstract():
    def __init__(self, valid_percentage=0.2, samples_per_pattern=10, jitter=None, valid_jitter=None, valid_jitters=None, *args, **kwargs):
        self.valid_samples_per_pattern = int(samples_per_pattern * valid_percentage)
        self.train_samples_per_pattern = samples_per_pattern - self.valid_samples_per_pattern

        self.dataset_valid = AbstractDataset(dataset_parent_path='Data/',
         samples_per_pattern=self.valid_samples_per_pattern, jitter=valid_jitter, *args, **kwargs)
        
        self.dataset_valids = None
        if valid_jitters is not None:
            self.dataset_valids = []
            for cur_valid_jitter in valid_jitters:
                self.dataset_valids.append(AbstractDataset(dataset_parent_path='Data/',
                    samples_per_pattern=self.valid_samples_per_pattern, jitter=cur_valid_jitter, *args, **kwargs))

        self.dataset_train = AbstractDataset(dataset_parent_path='Data/',
            samples_per_pattern=self.train_samples_per_pattern, jitter=jitter, *args, **kwargs)

        self.dataset_params = self.dataset_train.dataset_params
        self.dim_values_name = self.dataset_train.dim_values_name

    def get_ds_shape(self):
        return ((self.dataset_params['N_e'] + self.dataset_params['N_i'],), self.dataset_params['stim_dur'], (2,), 1)

    def get_train_loader(self, batch_size, num_workers=1):
        return DataLoader(self.dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    def get_valid_loader(self, batch_size, num_workers=1):
        return DataLoader(self.dataset_valid, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    def get_valid_loaders(self, batch_size, num_workers=1):
        if self.dataset_valids is None:
            raise ValueError("No valid datasets available, please set valid_jitters in the constructor")
        
        valid_loaders = []
        for cur_dataset_valid in self.dataset_valids:
            valid_loaders.append(DataLoader(cur_dataset_valid, batch_size=batch_size, shuffle=False, num_workers=num_workers))
        return valid_loaders