import numpy as np
import json
import sys
import pathlib
import matplotlib.pyplot as plt
import torch
import os
from enum import IntEnum
from tqdm import tqdm
from torch.nn import functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from typing import Sequence, Tuple, List, Optional
import math

import logging

try:
    from scipy.signal import firwin, lfilter
except ImportError:
    firwin = lfilter = None  # fallback later

from datasets.filter_dataset import FilterLabelsDataset, FilterItemsDataset
from datasets.hmax import HMAX
from datasets import sequences

from utils.utils import setup_logger

logger = logging.getLogger(__name__)

def pad_to_shape(x, target_shape):
    padding_list = []
    for x_dim, target_dim in zip(x.shape, target_shape):
        pad_value = (target_dim - x_dim)
        pad_tuple = ((pad_value//2, pad_value//2 + pad_value%2))
        padding_list.append(pad_tuple)
    
    return np.pad(x, tuple(padding_list), mode='constant')

def batch_pad_to_shape(x, target_shape):
    return np.array([pad_to_shape(x_i, target_shape) for x_i in x])
    
def stretch_to_shape(x, target_shape):
    kron_shape = [target_dim // x_dim for x_dim, target_dim in zip(x.shape, target_shape)]
    x_kron = np.kron(x, np.ones(kron_shape))
    return pad_to_shape(x_kron, target_shape)

def batch_stretch_to_shape(x, target_shape):
    return np.array([stretch_to_shape(x_i, target_shape) for x_i in x])

# now torch versions of the above

def torch_pad_to_shape(x, target_shape):
    padding_list = []
    for x_dim, target_dim in zip(x.shape, target_shape):
        pad_value = (target_dim - x_dim)
        pad_tuple = (pad_value//2, pad_value//2 + pad_value%2)
        padding_list = [pad_tuple[0], pad_tuple[1]] + padding_list
    
    return F.pad(x, tuple(padding_list))

def torch_batch_pad_to_shape(x, target_shape):
    return torch.stack([torch_pad_to_shape(x_i, target_shape) for x_i in x])

def torch_stretch_to_shape(x, target_shape):
    kron_shape = [target_dim // x_dim for x_dim, target_dim in zip(x.shape, target_shape)]
    x_kron = torch.kron(x, torch.ones(kron_shape))
    return torch_pad_to_shape(x_kron, target_shape)

def torch_batch_stretch_to_shape(x, target_shape):
    return torch.stack([torch_stretch_to_shape(x_i, target_shape) for x_i in x])

def subsample_to_spikes(block, average_firing_rate_per_axon, bg_activity_hz):
    if average_firing_rate_per_axon is None or average_firing_rate_per_axon < 0:
        pass
    else:
        # subsample the input spikes
        actual_mean_firing_rate_Hz = 1000 * block.float().mean()
        fraction_spikes_to_eliminate = average_firing_rate_per_axon / actual_mean_firing_rate_Hz

        block = block * (torch.rand(block.shape[0], block.shape[1]) < fraction_spikes_to_eliminate)
        
        # add bg activity
        block = ((torch.rand(*block.shape) < bg_activity_hz/1000.0) + block).bool()

    return block

def reduce_long_sequences(spikes, run_threshold=5, keep_probability=0.5):
    """
    Scan each axon (row) for consecutive sequences of spikes. If a sequence length 
    >= run_threshold, randomly thin that sequence by keeping each spike with 
    probability 'keep_probability' (and zeroing out the rest).

    Parameters:
    -----------
    spikes : 2D np.array [N_axons, T_orig]
        Original high-rate spike matrix (1 => spike, 0 => no spike).
    run_threshold : int
        Minimum consecutive spike length to trigger random thinning.
    keep_probability : float
        Probability of keeping a spike within a "too-long" run. 
        Must be between 0.0 and 1.0.

    Returns:
    --------
    reduced_spikes : 2D np.array [N_axons, T_orig]
        A new spike matrix with long spike runs reduced in firing rate.
    """
    N_axons, T_orig = spikes.shape
    reduced_spikes = np.copy(spikes)

    for axon_idx in range(N_axons):
        row = reduced_spikes[axon_idx]
        
        current_run_start = None
        for t in range(T_orig):
            if row[t] == 1:
                if current_run_start is None:
                    current_run_start = t  # beginning of a run
            else:
                # the run ended if we were in a run
                if current_run_start is not None:
                    run_length = t - current_run_start
                    if run_length >= run_threshold:
                        # Thin out the run
                        run_indices = np.arange(current_run_start, t)
                        # Keep each spike with probability keep_probability
                        keep_mask = (np.random.rand(run_length) < keep_probability)
                        # Zero out the ones we don't keep
                        row[run_indices] = 0
                        row[run_indices[keep_mask]] = 1
                    current_run_start = None
        
        # Edge case: if ended with a run that goes to the last time step
        if current_run_start is not None:
            run_length = T_orig - current_run_start
            if run_length >= run_threshold:
                run_indices = np.arange(current_run_start, T_orig)
                keep_mask = (np.random.rand(run_length) < keep_probability)
                row[run_indices] = 0
                row[run_indices[keep_mask]] = 1

    return reduced_spikes

def temporal_adaptation_spikes(
    spike_matrix: np.ndarray,
    alpha: float = 0.95,
    threshold: float = 0.2
) -> np.ndarray:
    """
    Simulates a simple form of adaptation: if a neuron's average recent activity is high,
    reduce its chance to fire. This is a simple exponential moving average approach.
    
    Args:
        spike_matrix: 2D array (neurons, time_bins).
        alpha: Decay parameter for the moving average. 
               Closer to 1 => slower adaptation, closer to 0 => faster adaptation.
        threshold: If the "adaptation state" is above this value, the neuron's output 
                   is silenced (set to 0).
    
    Returns:
        2D array (neurons, time_bins) with the adapted spiking.
    """
    spike_matrix = spike_matrix.T
    num_time, num_neurons = spike_matrix.shape
    output = np.zeros_like(spike_matrix)
    
    # We'll keep track of a running average for each neuron
    adaptation_state = np.zeros(num_neurons, dtype=float)
    
    for t in range(num_time):
        # Update adaptation state
        # For each neuron, adaptation_state[n] = alpha * adaptation_state[n] + (1-alpha)*spike
        # where 'spike' is 1 if that neuron fired at this time bin, else 0
        spike_current = spike_matrix[t, :]
        adaptation_state = alpha * adaptation_state + (1 - alpha) * spike_current
        
        # If adaptation_state[n] > threshold, we silence it 
        # (meaning it's "too adapted" => no output spike)
        # Otherwise, keep the spike as is
        # shape => (neurons,)
        silenced = adaptation_state > threshold
        
        # Keep spike only if not silenced
        output[t, :] = spike_current
        output[t, silenced] = 0

    output = output.T
    return output

def detect_onsets_offsets_spikes(
    spike_matrix: np.ndarray,
    window_size: int = 10,
    onset_threshold: float = 0.7,
    offset_threshold: float = 0.7
) -> np.ndarray:
    """
    Detects onsets and offsets in a spike matrix by comparing adjacent time windows.
    
    Args:
        spike_matrix: 2D array (neurons, time_bins), 0 or 1 for spiking activity.
        window_size: Number of time steps for computing average spike rates.
        onset_threshold: Minimum fractional increase in spike rate from one window to next
                         to consider it an "onset."
        offset_threshold: Minimum fractional decrease to consider it an "offset."
        
    Returns:
        2D array (neurons, time_bins) of 0/1. A "1" means an onset or offset was detected
        at that time step for that neuron. 
    """
    spike_matrix = spike_matrix.T
    num_time, num_neurons = spike_matrix.shape
    # Number of aggregated windows
    new_time_bins = num_time // window_size
    
    # Sum or average activity per window
    # shape => (new_time_bins, num_neurons)
    windowed_sum = np.array([
        np.sum(spike_matrix[t*window_size:(t+1)*window_size, :], axis=0)
        for t in range(new_time_bins)
    ])

    # We can convert to rates by dividing by window_size
    windowed_rate = windowed_sum / window_size

    # Compare consecutive windows to detect large changes
    # shape => (new_time_bins - 1, num_neurons)
    rate_diff = windowed_rate[1:] - windowed_rate[:-1]

    # Onset detection: rate_diff > onset_threshold * previous_rate
    # Offset detection: rate_diff < -offset_threshold * previous_rate
    # We'll mark a window as 1 if there's either an onset or offset
    # shape => (new_time_bins - 1, num_neurons)
    onset_offset_matrix = np.zeros_like(rate_diff)
    previous_rate = windowed_rate[:-1]

    onset_mask = rate_diff > (onset_threshold * previous_rate)
    offset_mask = rate_diff < (-offset_threshold * previous_rate)
    onset_offset_matrix[onset_mask] = 1
    onset_offset_matrix[offset_mask] = 1

    # Expand back to original time resolution if desired, or keep it at window level
    # For simplicity, we’ll keep it at window resolution
    # shape => (new_time_bins - 1, num_neurons)
    onset_offset_matrix = onset_offset_matrix.T
    return onset_offset_matrix

def detect_onsets_offsets_sustained_spikes(
    spike_matrix: np.ndarray,
    window_size: int = 10,
    overlap: float = 0.0,
    onset_threshold: float = 0.7,
    offset_threshold: float = 0.7,
    sustained_threshold: str | float = "median",
    split = False,
    also_sus_cont=False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns three binary matrices (onset, offset, sustained).

    * sustained[t,n] = 1  if rate[t,n] > θ  (θ can be float or "median"/"mean")
    * onset / offset rules identical to earlier version.
    """
    spike_matrix = spike_matrix.T

    if not (0.0 <= overlap < 1.0):
        raise ValueError("overlap must be in [0,1).")
    step = max(1, int(round(window_size * (1 - overlap))))
    T, N = spike_matrix.shape
    starts = np.arange(0, T - window_size + 1, step)
    rates = np.array([spike_matrix[s:s+window_size].sum(0) / window_size
                      for s in starts])        # shape (M, N)

    # sustained
    if isinstance(sustained_threshold, str):
        if sustained_threshold == "median":
            θ = np.median(rates)
        elif sustained_threshold == "mean":
            θ = rates.mean()
        else:
            raise ValueError("unknown sustained_threshold string")
    else:
        θ = sustained_threshold
    sustained = (rates[:-1] > θ).astype(np.int8)   # align with onset/offset rows

    if also_sus_cont:
        eps       = 1e-9
        peak_band = rates.max(axis=0, keepdims=True) + eps   # shape (1, N)
        sus_cont  = (rates[:-1] / peak_band).astype(np.float32)

    # onset / offset
    Δ = np.diff(rates, axis=0)
    prev = rates[:-1]
    onset  = (Δ >  onset_threshold  * prev).astype(np.int8)
    offset = (Δ < -offset_threshold * prev).astype(np.int8)

    if split:
        if also_sus_cont:
            return onset.T, offset.T, sustained.T, sus_cont.T
        else:
            return onset.T, offset.T, sustained.T

    # combine into one matrix
    onset_offset_sustained = np.concatenate(
        (onset, offset, sustained), axis=0
    )  # shape (3M, N)

    onset_offset_sustained = onset_offset_sustained.T

    return onset_offset_sustained
def envelope_extraction_spikes(
    spike_matrix: np.ndarray,
    kernel_size: int = 20,
    threshold_factor: float = 0.5
) -> np.ndarray:
    """
    Approximates "envelope" extraction by convolving each neuron's spike train
    with a smoothing window, then thresholding.
    
    Args:
        spike_matrix: 2D array (time_bins, neurons).
        kernel_size: Size of the smoothing window for approximating the envelope.
        threshold_factor: Multiply the maximum convolved value by this factor 
                         to create a threshold for generating spikes.
    
    Returns:
        2D array (time_bins, neurons) with 0/1 representing where envelope is above threshold.
    """
    spike_matrix = spike_matrix.T
    num_time, num_neurons = spike_matrix.shape
    output = np.zeros_like(spike_matrix)

    # Simple smoothing kernel (moving average)
    kernel = np.ones(kernel_size) / kernel_size

    for n in range(num_neurons):
        # Convolve spike train for neuron n
        conv_signal = np.convolve(spike_matrix[:, n], kernel, mode='same')
        # Determine threshold
        max_val = np.max(conv_signal) if np.max(conv_signal) != 0 else 1e-9
        thr = threshold_factor * max_val
        # Binarize
        output[:, n] = (conv_signal > thr).astype(int)

    output = output.T
    return output

def subsample_envelope_spikes(
    envelope: np.ndarray,
    time_window: int = 10,
    axon_group: int = 1,
    statistic: str = "mean",
    threshold: float = None
) -> np.ndarray:
    """
    Down‑samples an envelope matrix in time and/or across axons.

    Parameters
    ----------
    envelope : ndarray
        Shape (time_bins, axons), float or int.
    time_window : int
        Number of time bins to average / max‑pool over.
    axon_group : int
        Pool this many consecutive axons into one channel (1 = no pooling).
    statistic : {"mean","max"}
        Aggregation function for each window / axon group.
    threshold : float or None
        If given, binarise the pooled output: out>threshold → 1 else 0.
        You can pass a scalar (global threshold) or "median", "mean", etc.

    Returns
    -------
    ndarray
        Down‑sampled matrix, shape (⌊T/time_window⌋ , ⌊N/axon_group⌋).
    """
    envelope = envelope.T

    if statistic not in ("mean", "max"):
        raise ValueError("statistic must be 'mean' or 'max'.")

    T, N = envelope.shape
    T_ds = T // time_window
    N_ds = N // axon_group

    # Temporal pool
    env_t = envelope[:T_ds * time_window].reshape(T_ds, time_window, N)
    if statistic == "mean":
        env_t = env_t.mean(axis=1)
    else:
        env_t = env_t.max(axis=1)

    # Axon pool
    env_tx = env_t[:, :N_ds * axon_group].reshape(T_ds, N_ds, axon_group)
    if statistic == "mean":
        env_tx = env_tx.mean(axis=2)
    else:
        env_tx = env_tx.max(axis=2)

    if threshold is not None:
        if isinstance(threshold, str):
            if threshold == "median":
                thr = np.median(env_tx)
            elif threshold == "mean":
                thr = env_tx.mean()
            else:
                raise ValueError("unknown string threshold")
        else:
            thr = threshold
        env_tx = (env_tx > thr).astype(np.int8)

    env_tx = env_tx.T
    return env_tx


def binarise_subsample_envelope_spikes(env_pool):
    env_bin = (env_pool > np.median(env_pool, axis=0, keepdims=True)).astype(np.int8)    

    return env_bin

# ---------------------------------------------------------------------------
#  Utility: crude AM band‑pass filter on the envelope
# ---------------------------------------------------------------------------

def bandpass_am_filter(env: np.ndarray,
                       fs: float,
                       f_lo: float,
                       f_hi: float,
                       numtaps: int = 129) -> np.ndarray:
    """FIR band‑pass along the *time* axis for each envelope channel."""
    env = env.T
    if firwin is None or lfilter is None:
        # simple high‑minus‑low pass fallback if SciPy unavailable
        win_hi = int(max(1, fs / (2 * f_hi)))
        win_lo = int(max(1, fs / (2 * f_lo)))
        hi  = np.convolve(env,  np.ones(win_hi)/win_hi, mode="same")
        lo  = np.convolve(env,  np.ones(win_lo)/win_lo, mode="same")
        return np.clip(lo - hi, 0, None)
    nyq = fs / 2.0
    coeffs = firwin(numtaps, [f_lo/nyq, f_hi/nyq], pass_zero=False)
    # apply channel‑wise
    out = np.stack([lfilter(coeffs, 1.0, env[:, ch]) for ch in range(env.shape[1])], axis=1)
    out = out.T
    return out

# ---------------------------------------------------------------------------
#  Utility: local k‑winner‑take‑all along frequency axis (per time‑step)
# ---------------------------------------------------------------------------

def k_wta_pool(bits: np.ndarray,
               group_size: int = 3,
               k: int = 1) -> np.ndarray:
    """For every time row, partition channels into contiguous groups of *group_size*
    and keep the *k* largest (ties resolved arbitrarily) inside each group."""
    bits = bits.T
    T, N = bits.shape
    n_groups = N // group_size
    if n_groups == 0:
        return bits.copy()  # nothing to pool

    pooled = np.zeros((T, n_groups * k), dtype=np.int8)
    for g in range(n_groups):
        seg = bits[:, g*group_size:(g+1)*group_size]
        # argsort descending along freq axis
        topk_idx = np.argsort(seg, axis=1)[:, -k:]
        # build boolean mask for top‑k
        mask = np.zeros_like(seg)
        row = np.arange(T)[:, None]
        mask[row, topk_idx] = 1
        pooled[:, g*k:(g+1)*k] = (seg * mask).any(axis=2).astype(np.int8)
    pooled = pooled.T
    return pooled

def k_wta_pool2(binary: np.ndarray, k: int = 1, group: int = 8) -> np.ndarray:
    """Local k‑WTA (winner‑take‑all) along the frequency axis.

    Splits channels into non‑overlapping groups of size *group* and keeps the
    *k* strongest activations per time step inside each group.
    """
    binary = binary.T
    T, N = binary.shape
    if group <= 0 or k <= 0:
        raise ValueError("group and k must be positive")
    G = N // group
    out = np.zeros_like(binary)
    for g in range(G):
        seg = binary[:, g * group:(g + 1) * group]
        # indices of k largest values per row (ties arbitrary)
        idx_topk = np.argpartition(-seg, kth=k - 1, axis=1)[:, :k]
        rows = np.repeat(np.arange(T)[:, None], k, axis=1)
        cols = idx_topk + g * group
        out[rows, cols] = seg[rows, idx_topk]
    out = out.T
    return out

def multi_level_amplitude(
    envelope: np.ndarray,
    levels: Sequence[float] = (0.2, 0.5, 0.8),
    normalise_per_band: bool = True,
    eps: float = 1e-9
) -> np.ndarray:
    """
    Convert a continuous envelope (T × N) into multiple binary “intensity
    channels” per band.

    Each threshold τ in *levels* becomes one extra axon per original band:

        out[:,  k*N + n]  = 1  <=>  env[:, n]  >  τ_n

    Parameters
    ----------
    envelope : ndarray, shape (T, N)
        Continuous non-negative envelope values.
    levels : iterable of float
        Ascending thresholds in the range (0, 1] *after normalisation*.
    normalise_per_band : bool
        If True (default) divide each band by its own max, mimicking
        cochlear-nerve automatic gain control.  Set False if env is already
        normalised globally.
    eps : float
        Tiny number to avoid division by zero.

    Returns
    -------
    ndarray, shape (T, N × len(levels))
        Binary matrix; channels are ordered [τ1-all-bands, τ2-all-bands, ...].
    """
    env = envelope.copy()
    env = env.T
    if normalise_per_band:
        peak = env.max(axis=0, keepdims=True) + eps
        env /= peak                       # now each band ∈ [0, 1]

    T, N = env.shape
    out = []
    for τ in levels:
        out.append((env > τ).astype(np.int8))
    out = np.concatenate(out, axis=1)    # shape (T, N·|levels|)
    out = out.T
    return out

def sequence_detector(
    binary_matrix: np.ndarray,
    pairs: Optional[List[Tuple[int, int]]] = None,
    max_lag: int = 3,
    causal: bool = True
) -> np.ndarray:
    """
    Detect ordered pairs of events across bands.

    A feature fires at time *t* if:

        binary_matrix[t,  src] == 1           and
        binary_matrix[t+δ, dst] == 1
        for some δ ∈ [1, max_lag]  (causal=True)     OR
        δ ∈ [-max_lag, -1]        (causal=False, i.e. either order)

    Parameters
    ----------
    binary_matrix : ndarray, shape (T, N)
        Binarised activity (spikes or thresholded envelope).
    pairs : list[(src, dst)] or None
        Which band pairs to test.  If None, every ordered pair (i, j≠i) is used.
    max_lag : int
        Maximum number of time bins between the two events.
    causal : bool
        If True (default) look only forward in time (src ➜ dst).
        If False detect either order (src before dst *or* dst before src).

    Returns
    -------
    ndarray, shape (T,  len(pairs)  *  (1 or 2))
        Binary sequence features.  Order of columns matches *pairs*; if
        causal=False two columns per pair are returned
        (src→dst  followed by  dst→src).
    """
    bits = binary_matrix.astype(bool)
    bits = bits.T
    T, N = bits.shape

    # Build list of (src, dst, direction) tuples
    if pairs is None:
        pairs = [(i, j) for i in range(N) for j in range(N) if i != j]
    tasks = []
    for (i, j) in pairs:
        tasks.append((i, j, +1))              # src→dst
        if not causal:
            tasks.append((j, i, +1))          # dst→src (same forward code)

    n_feat = len(tasks)
    out = np.zeros((T, n_feat), dtype=np.int8)

    # Pre-compute shifted matrices for each lag
    shifted = {
        lag: np.pad(bits[:-lag], ((lag, 0), (0, 0))) if lag > 0
        else np.pad(bits[-lag:], ((0, -lag), (0, 0)))
        for lag in range(1, max_lag + 1)
    }

    for k, (src, dst, _) in enumerate(tasks):
        seq = np.zeros(T, dtype=bool)
        for lag in range(1, max_lag + 1):
            seq |= bits[:, src] & shifted[lag][:, dst]
        out[:, k] = seq.astype(np.int8)

    out = out.T
    return out

# ---------------------------------------------------------------------------
#  Main hierarchical pipeline
# ---------------------------------------------------------------------------

def hierarchical_audio_pipeline(spike_matrix: np.ndarray,
                          samp_hz: float = 1000.0,
                          *,
                          max_axons: int = 6000,
                          env_time_window: int = 10,   # ms used for base envelope pooling
                          freq_group: int = 3,
                          envelope_levels: tuple[float, ...] = (0.3, 0.6, 0.8),
                          onset_windows: tuple[int, int] = (10, 40),
                          am_bands: tuple[tuple[int, int], ...] = ((4, 16), (16, 64)),
                          k_wta_k: int = 1,
                          split=False) -> np.ndarray:
    """Four‑layer auditory HMAX‑like pipeline (S1→C1→S2→C2) with automatic
    dimensionality control.

    Parameters
    ----------
    spike_matrix : np.ndarray, shape (T, N)
        Binary bushy‑cell spikes (0/1).
    samp_hz : float
        Sampling rate of *time bins* (e.g. 1000 → 1 ms bins).
    max_axons : int
        Maximum number of output channels.

    Returns
    -------
    np.ndarray, shape (new_T, ≤max_axons)
        Final binary feature matrix ready to feed the dendritic neuron.
    """
    spike_matrix = spike_matrix.T
    T, N = spike_matrix.shape

    # ------------------------------------------------------------------
    # S‑1 : Base envelope + onset/offset/sustained; AM filters
    # ------------------------------------------------------------------
    # Continuous envelope via running mean
    window_bins = int(round(env_time_window / 1000 * samp_hz))
    env = np.convolve(spike_matrix.astype(float).reshape(-1),
                      np.ones(window_bins) / window_bins, mode="same").reshape(T, N)

    # Multi‑level envelope binary channels (per‑band normalisation)
    env_levels = multi_level_amplitude(env, levels=envelope_levels, normalise_per_band=True)

    # Onset/offset/sustained at two scales
    onset_feats_list = []
    for w in onset_windows:
        o, f, s = detect_onsets_offsets_sustained_spikes(spike_matrix,   # use raw spikes for timing precision
                                                  window_size=int(round(w/1000 * samp_hz)),
                                                  overlap=0.0,
                                                  onset_threshold=0.5,
                                                  offset_threshold=0.5,
                                                  sustained_threshold="median", split=True)
        onset_feats_list.extend([o, f, s])
    onoffsus = np.concatenate(onset_feats_list, axis=1)   # (T', 3*len(onset_windows)*N)

    # AM band‑pass filters on the envelope (continuous → then binarise 0.3 * per‑band max)
    am_feats = []
    for f_lo, f_hi in am_bands:
        bp = bandpass_am_filter(env, fs=samp_hz, f_lo=f_lo, f_hi=f_hi)
        thr = 0.3 * (bp.max(axis=0, keepdims=True) + 1e-9)
        am_feats.append((bp > thr).astype(np.int8))
    am_bin = np.concatenate(am_feats, axis=1)

    # Concatenate S1 feature map (time aligned).  Need same T across all matrices.
    min_T = min(env_levels.shape[0], onoffsus.shape[0], am_bin.shape[0])
    S1 = np.concatenate([env_levels[:min_T], onoffsus[:min_T], am_bin[:min_T]], axis=1)
    S1 = S1.T

    # ------------------------------------------------------------------
    # C‑1 : local k‑WTA pooling along frequency dimension
    # ------------------------------------------------------------------
    C1 = k_wta_pool2(S1.T, group=freq_group, k=k_wta_k)
    C1 = C1.T

    # ------------------------------------------------------------------
    # S‑2 : sequence detectors on C1 (nearest neighbour + octave up/down)
    # ------------------------------------------------------------------
    N_c1 = C1.shape[1]
    pairs = []
    for i in range(N_c1):
        # nearest neighbour
        if i+1 < N_c1:
            pairs.append((i, i+1))
        # octave up/down approximated by doubling index spacing (if valid)
        if i*2 < N_c1:
            pairs.append((i, i*2))
        if i % 2 == 0 and i//2 < N_c1:
            pairs.append((i, i//2))
    S2 = sequence_detector(C1, pairs=pairs, max_lag=3, causal=True)

    # ------------------------------------------------------------------
    # C‑2 : temporal average pool  (50 ms)  → then median threshold
    # ------------------------------------------------------------------
    pool_bins = int(round(50/1000 * samp_hz))
    new_T = min_T // pool_bins
    pooled = C1[:new_T*pool_bins].reshape(new_T, pool_bins, -1).mean(axis=1)
    C2 = (pooled > np.median(pooled)).astype(np.int8)

    # ------------------------------------------------------------------
    # Final concatenation  (align to new_T)
    # ------------------------------------------------------------------
    S2_aligned = S2[:new_T]
    features = np.concatenate([C2, S2_aligned], axis=1)

    # ------------------------------------------------------------------
    # Dimensionality control — group extra channels if > max_axons
    # ------------------------------------------------------------------
    if features.shape[1] > max_axons:
        group = int(np.ceil(features.shape[1] / max_axons))
        n_groups = features.shape[1] // group
        features = features[:, :n_groups*group].reshape(new_T, n_groups, group).max(axis=2)

    
    features = features.astype(np.int8)
    features = features.T

    if split:
        return S1.T, C1.T, S2.T, features

    else:
        return features

# --- 1. modulation filters  ------------------------------------------
def bandpass_fir(lo, hi, fs, ntaps=129):
    from scipy.signal import firwin
    return firwin(ntaps, [lo, hi], pass_zero=False, fs=fs)

def am_filter_bank(env, fs, bands, thr_factor):
    T, N = env.shape
    out = []
    for lo, hi in bands:
        kern = bandpass_fir(lo, hi, fs)
        am = np.apply_along_axis(lambda x: np.convolve(x, kern, "same"),
                                 0, env)
        # per–band threshold
        thr = thr_factor * am.max(axis=0, keepdims=True)
        out.append((am > thr).astype(np.int8))
    return np.concatenate(out, axis=1)   # (T, N * len(am_bands))

# --- 2. neighbour coincidence (lag 0 & ±1 frame) ---------------------
def neighbour_and(bits, lag1=True, octave=False):
    T, N = bits.shape
    neigh = bits[:, :-1] & bits[:, 1:]              # same-frame n & n+1
    outs = [neigh]
    if octave and N > 2:
        outs.append(bits[:, :-2] & bits[:, 2:])     # n & n+2 (≈octave)
    if lag1:
        shifted = np.pad(bits[:-1], ((1,0),(0,0)))  # lag +1
        outs.append(shifted[:, :-1] & bits[:, 1:])
        if octave:
            outs.append(shifted[:, :-2] & bits[:, 2:])
    return np.concatenate(outs, axis=1)

# --------------------------------------------------------------
# helper: time-only max-pool
# --------------------------------------------------------------
def time_max_pool(bits: np.ndarray, pool_ms: int, fs: float) -> np.ndarray:
    """
    Max-pool along the time axis only.
    bits : (T, N) binary
    """
    bits = bits.T
    T, N = bits.shape
    step = int(round(pool_ms * fs / 1000.0))
    n_win = T // step
    pooled = bits[:n_win * step,:].reshape(n_win, step, N).max(axis=1)
    return pooled.T

# --------------------------------------------------------------
# helper: local k-WTA along freq axis
# --------------------------------------------------------------
def freq_k_wta(bits: np.ndarray, k: int, band: int = 3) -> np.ndarray:
    """
    Keep at most *k* winners in each ±band patch.
    band = 3 means a patch size of 7 channels (-3…+3).
    """
    bits = bits.T
    if k >= band * 2 + 1:
        return bits.T  # nothing to prune

    T, N = bits.shape
    out  = bits.copy()
    half = band
    for n in range(N):
        left = max(0, n - half)
        right = min(N, n + half + 1)
        patch = bits[:, left:right]
        # winners = indices of the k largest columns in this patch
        # treat binary as counts per column
        col_sum = patch.sum(0)
        if np.count_nonzero(col_sum) <= k:
            continue
        keep_cols = col_sum.argsort()[-k:]
        mask = np.ones_like(patch, dtype=bool)
        mask[:, keep_cols] = False
        out[:, left:right][mask] = 0
    return out.T

def freq_group_or(bits: np.ndarray, max_axons: int) -> np.ndarray:
    """
    Compress columns by OR-ing g adjacent channels, where
    g = ceil(N / max_axons).  Guarantees ≤ max_axons columns.
    bits: (T, N) binary
    """
    bits = bits.T
    T, N = bits.shape
    if N <= max_axons:
        return bits

    g = math.ceil(N / max_axons)
    N_new = N // g
    trimmed = bits[:, :N_new * g]                 # drop remainder
    grouped = trimmed.reshape(T, N_new, g).max(axis=2)
    return grouped.astype(np.int8).T

# --------------------------------------------------------------
# helper: adaptive frequency grouping (logical OR)
# --------------------------------------------------------------
def adaptive_group(bits: np.ndarray, max_channels: int) -> np.ndarray:
    """
    If bits.shape[1] > max_channels, OR neighbour channels together
    so that the output has <= max_channels columns.
    """
    bits = bits.T
    T, N = bits.shape
    if N <= max_channels:
        return bits.T

    g = math.ceil(N / max_channels)                # minimal group size
    N_new = N // g
    bits_trim = bits[:, :N_new * g]
    grouped = bits_trim.reshape(T, N_new, g).max(axis=2)
    return grouped.astype(np.int8).T

def hierarchical_audio_pipeline2(bushy_data, window_size=20, overlap=0.0,
                          onset_threshold=0.5, offset_threshold=0.5,
                          sustained_threshold="median", 
                          fs = 1000.0,
                          am_bands=((4, 8), (8, 16), (16, 32), (32, 64)),
                          thr_factor=0.3, c1_k=2, c1_group=4, c2_k=2, c2_band=4,
                          c2_or_factor=3,
                          c2_am_pool_ms=4, c2_co_pool_ms=8, max_axons=6000):
    on_10_0_5, off_10_0_5, sus_10_0_5, sus_10_0_5_cont = detect_onsets_offsets_sustained_spikes(
        spike_matrix=bushy_data,
        window_size=window_size,
        overlap=overlap,
        onset_threshold=onset_threshold,
        offset_threshold=offset_threshold,
        sustained_threshold=sustained_threshold,
        split=True,
        also_sus_cont=True)
    # print(f"There are {on_10_0_5.shape[0] * on_10_0_5.shape[1]} onset features.")
    # print(f"There are {off_10_0_5.shape[0] * off_10_0_5.shape[1]} offset features.")
    # print(f"There are {sus_10_0_5.shape[0] * sus_10_0_5.shape[1]} sustained features.")
    # print(f"There are {sus_10_0_5_cont.shape[0] * sus_10_0_5_cont.shape[1]} sustained features cont.")

    # am_bits = am_filter_bank(c1_env, fs=fs)   # c1_env is the *continuous* envelope that won the WTA

    # co_bits = neighbour_and(am_bits, lag1=True, octave=True)

    # # --- 3. concatenate AM + coincidence ---------------------------------
    # s2 = np.concatenate([am_bits, co_bits], axis=1)

    # uses the continuous envelope from S1
    am_bits = am_filter_bank(sus_10_0_5_cont, fs=fs,
                            bands=am_bands,
                            thr_factor=thr_factor)        # returns binary
    # print(f"There are {am_bits.shape[0] * am_bits.shape[1]} AM features.")

    on_10_0_5_pool = k_wta_pool2(on_10_0_5, k=c1_k, group=c1_group)
    # print(f"There are {on_10_0_5_pool.shape[0] * on_10_0_5_pool.shape[1]} pooled onset features.")

    off_10_0_5_pool = k_wta_pool2(off_10_0_5, k=c1_k, group=c1_group)
    # print(f"There are {off_10_0_5_pool.shape[0] * off_10_0_5_pool.shape[1]} pooled offset features.")

    sus_10_0_5_pool = k_wta_pool2(sus_10_0_5, k=c1_k, group=c1_group)
    # print(f"There are {sus_10_0_5_pool.shape[0] * sus_10_0_5_pool.shape[1]} pooled sustained features.")

    base_bits = np.concatenate([on_10_0_5_pool, off_10_0_5_pool, sus_10_0_5_pool], axis=1)
    # print(f"There are {base_bits.shape[0] * base_bits.shape[1]} base features.")

    co_bits   = neighbour_and(base_bits, lag1=True, octave=True)
    # print(f"There are {co_bits.shape[0] * co_bits.shape[1]} coincidence features.")

    # s2 = np.concatenate([am_bits, co_bits], axis=1)   # still binary
    # # print(f"There are {s2.shape[0] * s2.shape[1]} combined features.")


    # 1. local k-WTA along freq axis (band = ±3 ch, k = 1)
    am_kw = freq_k_wta(am_bits, k=c2_k, band=c2_band)
    co_kw = freq_k_wta(co_bits, k=c2_k, band=c2_band)
    # print(f"There are {am_kw.shape[0] * am_kw.shape[1]} AM kw features.")
    # print(f"There are {co_kw.shape[0] * co_kw.shape[1]} coincidence kw features.")

    am_gor = freq_group_or(am_kw, max_axons=am_kw.shape[0]//c2_or_factor)
    co_gor = freq_group_or(co_kw, max_axons=co_kw.shape[0]//c2_or_factor)
    # print(f"There are {am_gor.shape[0] * am_gor.shape[1]} AM gor features.")
    # print(f"There are {co_gor.shape[0] * co_gor.shape[1]} coincidence gor features.")

    # 1. time-only max-pool (keeps tonotopy)
    am_t  = time_max_pool(am_gor, c2_am_pool_ms, fs)
    co_t  = time_max_pool(co_gor, c2_co_pool_ms, fs)
    # print(f"There are {am_t.shape[0] * am_t.shape[1]} AM t features.")
    # print(f"There are {co_t.shape[0] * co_t.shape[1]} coincidence t features.")

    # 3. concatenate
    C2 = np.concatenate([am_t, co_t], axis=1)
    # print(f"There are {C2.shape[0] * C2.shape[1]} combined features.")

    # 4. adaptive frequency grouping if still too wide
    C2_grouped = adaptive_group(C2, max_channels=max_axons)
    # print(f"There are {C2_grouped.shape[0] * C2_grouped.shape[1]} grouped features.")

    return C2_grouped

import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert, sosfilt
from math import ceil
from typing import Sequence, Tuple


# -----------------------------------------------------------------------------#
# 1. Cochlear filter‑bank
# -----------------------------------------------------------------------------#
def _erb_space(low_freq: float, high_freq: float, n: int) -> np.ndarray:
    """Return `n` centre freqs (Hz) equally spaced on the ERB scale."""
    ear_q, min_bw = 9.26449, 24.7
    low_erb = ear_q * np.log(1 + low_freq / (ear_q * min_bw))
    high_erb = ear_q * np.log(1 + high_freq / (ear_q * min_bw))
    erb_points = np.linspace(low_erb, high_erb, n)
    return (ear_q * min_bw) * (np.exp(erb_points / ear_q) - 1)


def _design_filter_bank(
    fs: int,
    num_bands: int = 64,
    low_lim: float = 50,
    high_lim: float = 8000,
) -> Tuple[np.ndarray, list]:
    """Return (centre_freqs, list_of_IIR_SOS)."""
    centres = _erb_space(low_lim, high_lim, num_bands)
    sos_filters = []
    for fc in centres:
        low, high = max(fc / np.sqrt(2), 10), min(fc * np.sqrt(2), fs / 2 - 50)
        sos = butter(4, [low / (fs / 2), high / (fs / 2)], "bandpass", output="sos")
        sos_filters.append(sos)
    return centres, sos_filters


def _cochlear_filterbank(x: np.ndarray, fs: int, sos_filters: Sequence) -> np.ndarray:
    """(C, N) array of cochlear channel signals."""
    return np.stack([sosfilt(sos, x) for sos in sos_filters], axis=0)


# -----------------------------------------------------------------------------#
# 2. Inner‑hair‑cell envelope
# -----------------------------------------------------------------------------#
def _ihc_envelope(coch_sig: np.ndarray, fs: int, lp_cutoff: float = 1000) -> np.ndarray:
    rectified = np.maximum(coch_sig, 0.0)
    env = np.abs(hilbert(rectified, axis=-1))
    sos = butter(2, lp_cutoff / (fs / 2), "lowpass", output="sos")
    env = sosfiltfilt(sos, env, axis=-1)
    return np.log1p(3.0 * env)  # logarithmic compression


# -----------------------------------------------------------------------------#
# 3. Modulation filter‑bank
# -----------------------------------------------------------------------------#
def _make_temporal_gabor(rate_hz: float, fs: int, length_ms: int = 250) -> np.ndarray:
    """1‑D zero‑mean Gabor kernel (cosine carrier)."""
    n = ceil(length_ms * fs / 1000)
    if n % 2 == 0:
        n += 1
    t = (np.arange(n) - n // 2) / fs
    sigma = 1.0 / (2 * np.pi * rate_hz)
    kernel = np.cos(2 * np.pi * rate_hz * t) * np.exp(-t**2 / (2 * sigma**2))
    return kernel - kernel.mean()


def _modulation_filterbank(
    env: np.ndarray,
    fs: int,
    rates: Sequence = (4, 8, 16, 32, 64, 128),
) -> np.ndarray:
    feats = []
    for r in rates:
        g = _make_temporal_gabor(r, fs)
        tmp = np.apply_along_axis(lambda m: np.convolve(m, g, "same"), -1, env)
        feats.append(tmp)
    return np.maximum(np.stack(feats, 0), 0.0)  # (M, C, N)


# -----------------------------------------------------------------------------#
# 4. 2‑D max‑pool
# -----------------------------------------------------------------------------#
def _pool_2d(feats: np.ndarray, freq_pool: int, time_pool: int) -> np.ndarray:
    M, C, N = feats.shape
    C_out, N_out = C // freq_pool, N // time_pool
    pooled = np.empty((M, C_out, N_out), dtype=feats.dtype)
    for m in range(M):
        for i in range(C_out):
            for j in range(N_out):
                pooled[m, i, j] = feats[
                    m,
                    i * freq_pool : (i + 1) * freq_pool,
                    j * time_pool : (j + 1) * time_pool,
                ].max()
    return pooled


# -----------------------------------------------------------------------------#
# 5. Public API
# -----------------------------------------------------------------------------#
def wav_to_binary_features_func(
    wav: np.ndarray,
    fs: int = 16000,
    *,
    num_bands: int = 64,
    low_lim: float = 50,
    high_lim: float = 8000,
    rates: Sequence = (4, 8, 16, 32, 64, 128),
    freq_pool: int = 2,
    time_pool_ms: int = 50,
    rng: np.random.Generator | None = None,
    max_features: int | None = 6000,
    k_thresh: float = 1.0,
    q: float | None = None,
    comp_alpha: float | None = None,
    signed_branch: bool = False,
    multi_thresh: bool = False,
    output_all: bool = False,
) -> np.ndarray:
    """
    Convert a 1‑D waveform into a 0/1 feature vector.

    Parameters
    ----------
    wav : ndarray
        Mono audio waveform.
    fs : int
        Sample rate (Hz).
    num_bands : int
        Cochlear channels.
    rates : Sequence[int]
        Modulation rates (Hz).
    freq_pool, time_pool_ms : int
        Pool size across frequency & time.
    max_features : int or None
        If not None, randomly subsample features to this length.

    Returns
    -------
    binary : ndarray[int8], shape (F,)
    """
    centres, sos_filters = _design_filter_bank(fs, num_bands, low_lim, high_lim)
    coch = _cochlear_filterbank(wav, fs, sos_filters)
    env = _ihc_envelope(coch, fs)
    if comp_alpha:
        env = np.log1p(comp_alpha * env)
    feats = _modulation_filterbank(env, fs, rates)
    if signed_branch:
        pos = np.maximum(feats, 0)
        neg = np.maximum(-feats, 0)     # offset energy
        feats = np.concatenate([pos, neg], axis=0)   # doubles M
    time_pool = int(time_pool_ms * fs / 1000)
    pooled = _pool_2d(feats, freq_pool, time_pool)

    if multi_thresh:
        thr_lo = pooled.mean() + 0.3 * pooled.std()
        thr_hi = pooled.mean() + 0.9 * pooled.std()
        b_lo   = (pooled > thr_lo).astype(np.uint8)
        b_hi   = (pooled > thr_hi).astype(np.uint8)
        binary = np.concatenate([b_lo, b_hi], axis=0).flatten()

    else:
        if q is not None:
            thr = np.quantile(pooled, q)   # single scalar per clip
            binary = (pooled > thr).astype(np.uint8)
        else:
            thr = pooled.mean() + k_thresh * pooled.std()
            binary = (pooled > thr).astype(np.int8).flatten()

    if max_features is not None and binary.size > max_features:
        raise ValueError(f"There are {binary.size} features which is more than {max_features}")

    if output_all:
        return coch, env, feats, pooled, binary

    return binary

class BinarizationMethod(IntEnum):
    NONE = 0
    THRESHOLD = 1
    THRESHOLD_MEAN = 2
    BUCKETIZE_POISSON = 3
    BUCKETIZE_TEMPORAL = 4
    BUCKETIZE_RAW = 5
    THRESHOLD_POISSON = 6
    THRESHOLD_TEMPORAL = 7
    THRESHOLD_MEAN_POISSON = 8
    THRESHOLD_MEAN_TEMPORAL = 9


class GaborMethod(IntEnum):
    NONE = 0
    GABOR_S1 = 1
    GABOR_C1 = 2
    GABOR_S2 = 3
    GABOR_C2 = 4
    GABOR_STACK = 5

global_hmax = None

def get_transform_2d_to_spikes(h_crop_range=None, w_crop_range=None, gabor_method=GaborMethod.NONE,\
    vertical_duplicate=False, binarization_method=BinarizationMethod.THRESHOLD, binarization_threshold=0.5,\
        count_axons=1000, count_const_firing_axons=40, time_in_ms=400, average_firing_rate_per_axon=20, bg_activity_hz=0,
        did_gabor=False, bucketize_bins=10, temporal_delay_from_start=10, to_presampled=False, trial=None,\
             average_burst_firing_rate_per_axon=200, jitter=2.5, arange_spikes=True, temporal_locations_file=None, gabor_c1_only_unit_0=False,
             subsample_modulo=None, reduce_fr=False, temporal_adaptation=False, detect_onsets_offsets=False, detect_onsets_offsets_window_size=None, detect_onsets_offsets_threshold=None, kron=True,
             detect_onsets_offsets_sustained=False, detect_onsets_offsets_sustained_window_size=None, detect_onsets_offsets_sustained_overlap=None,
             detect_onsets_offsets_sustained_onset_threshold=None, detect_onsets_offsets_sustained_offset_threshold=None,
             envelope_extraction=False, envelope_extraction_kernel_size=None, envelope_extraction_threshold=None,
             subsample_envelope=False, subsample_envelope_time_window=None, subsample_envelope_axon_group=None, subsample_envelope_statistic=None,
             binarise_subsample_envelope=False, binarised_subsample_envelope_and_detect_onsets_offsets_sustained_not_binary=False,
             hierarchical_audio_processing=False, hierarchical_audio_processing2=False, ds_shorter_name=False, wav_to_binary_features=False,):

    def transform_2d_to_spikes(im, additional_args=None):
        if additional_args is not None:
            if 'temporal_locations_file' in additional_args:
                temporal_locations_file = additional_args['temporal_locations_file']

        if len(im.shape) == 3:
            if im.shape[0] == 1:
                im = im[0]
            else:
                raise ValueError(f"Unsupported data shape {im.shape}")

        # first apply some preprocessing on the data, like cropping, gabor, etc.

        if h_crop_range is not None:
            if did_gabor:
                raise ValueError("h_crop_range is not None, but did_gabor")
            im = im[h_crop_range[0]:h_crop_range[1], :]
        if w_crop_range is not None:
            if did_gabor:
                raise ValueError("w_crop_range is not None, but did_gabor")
            im = im[:, w_crop_range[0]:w_crop_range[1]]

        if not did_gabor:
            if gabor_method is not GaborMethod.NONE:
                # add dim to im
                im = im[None]
                im = im[None]

                global global_hmax
                if global_hmax is None:
                    global_hmax = HMAX('universal_patch_set.mat')

                if gabor_method == GaborMethod.GABOR_S1:
                    global_hmax.until_s1 = True
                    # TODO: do that on GPU or save to disk?
                    s1 = global_hmax.run_all_layers(im)

                    s1_0_shape = list(s1[0].shape)
                    
                    # TODO: needed?
                    # s1_0_shape[-1] = time_in_ms
                    s1_stretched = torch_batch_stretch_to_shape(s1, tuple(s1_0_shape))

                    s1_im = s1_stretched.reshape(-1, s1_stretched.shape[-1])

                    im = s1_im
                elif gabor_method == GaborMethod.GABOR_C1:
                    global_hmax.until_s1 = False
                    global_hmax.until_c1 = True
                    # TODO: do that on GPU or save to disk?
                    s1, c1 = global_hmax.run_all_layers(im)


                    if gabor_c1_only_unit_0:
                        c1_stretched = c1[0]

                    else:
                        c1_0_shape = list(c1[0].shape)
                        # TODO: needed?
                        # c1_0_shape[-1] = time_in_ms
                        c1_stretched = torch_batch_stretch_to_shape(c1, tuple(c1_0_shape))

                    c1_im = c1_stretched.reshape(-1, c1_stretched.shape[-1])
                
                    im = c1_im
                elif gabor_method == GaborMethod.GABOR_S2:
                    global_hmax.until_s1 = False
                    global_hmax.until_c1 = False
                    global_hmax.until_s2 = True
                    # TODO: do that on GPU or save to disk?
                    s1, c1, s2 = global_hmax.run_all_layers(im)

                    s2_full_list = []
                    for i in range(len(s2)):
                        s2_full_list += s2[i]
                    s2_stretched = torch_batch_stretch_to_shape(s2_full_list, s2_full_list[0].shape)

                    s2_im = s2_stretched.reshape(-1, s2_stretched.shape[-1])

                    im = s2_im
                elif gabor_method == GaborMethod.GABOR_C2:
                    global_hmax.until_s1 = False
                    global_hmax.until_c1 = False
                    global_hmax.until_s2 = False
                    c2 = global_hmax(im)

                    # for debugging purposes

                    # s1, c1, s2, _ = global_hmax.run_all_layers(im)
                    # s1_0_shape = list(s1[0].shape)
                    # s1_0_shape[-1] = time_in_ms
                    # s1_stretched = torch_batch_stretch_to_shape(s1, tuple(s1_0_shape))

                    # s1_im = s1_stretched.reshape(-1, s1_stretched.shape[-1])
                    # print("s1_im", s1_im.shape)

                    # c1_0_shape = list(c1[0].shape)
                    # c1_0_shape[-1] = time_in_ms
                    # c1_stretched = torch_batch_stretch_to_shape(c1, tuple(c1_0_shape))

                    # c1_im = c1_stretched.reshape(-1, c1_stretched.shape[-1])
                    # print("c1_im", c1_im.shape)

                    # s2_full_list = []
                    # for i in range(len(s2)):
                    #     s2_full_list += s2[i]
                    # s2_stretched = torch_batch_stretch_to_shape(s2_full_list, s2_full_list[0].shape)

                    # s2_im = s2_stretched.reshape(-1, s2_stretched.shape[-1])
                    # print("s2_im", s2_im.shape)

                    # print("c2", c2.shape) # torch.Size([1, 8, 400])
                    c2_im = c2.reshape(-1, c2.shape[-1])
                    # print("c2_im", c2_im.shape) # torch.Size([8, 400])
                
                    im = c2_im
                else:
                    raise ValueError("Unsupported gabor method")
                
        # applying transformations of spikes -> features

        if envelope_extraction:
            # print("before envelope_extraction", im.shape)
            kernel_size = envelope_extraction_kernel_size if envelope_extraction_kernel_size is not None else 20
            threshold_factor = envelope_extraction_threshold if envelope_extraction_threshold is not None else 0.5
            im = envelope_extraction_spikes(im.numpy(), kernel_size=kernel_size, threshold_factor=threshold_factor)
            im = torch.tensor(im)
            # print("after envelope_extraction", im.shape)

        if subsample_envelope:
            # print("before subsample_envelope", im.shape)
            time_window = subsample_envelope_time_window if subsample_envelope_time_window is not None else 25
            axon_group = subsample_envelope_axon_group if subsample_envelope_axon_group is not None else 3
            statistic = subsample_envelope_statistic if subsample_envelope_statistic is not None else "mean"
            im = subsample_envelope_spikes(im.numpy(), time_window=time_window, axon_group=axon_group, statistic=statistic)
            im = torch.tensor(im)
            # print("after subsample_envelope", im.shape)

        if binarise_subsample_envelope:
            # print("before binarise_subsample_envelope", im.shape)
            if binarised_subsample_envelope_and_detect_onsets_offsets_sustained_not_binary:
                not_binary_im = im
            im = binarise_subsample_envelope_spikes(im.numpy())
            im = torch.tensor(im)
            # print("after binarise_subsample_envelope", im.shape)
        
        # old and weird
        if detect_onsets_offsets:
            # print("before detect_onsets_offsets", im.shape)
            doo_ws = detect_onsets_offsets_window_size if detect_onsets_offsets_window_size is not None else 20
            doo_th = detect_onsets_offsets_threshold if detect_onsets_offsets_threshold is not None else 0.9
            im = detect_onsets_offsets_spikes(im.numpy(), window_size=doo_ws, onset_threshold=doo_th, offset_threshold=doo_th)
            im = torch.tensor(im)
            # print("after detect_onsets_offsets", im.shape)

        if detect_onsets_offsets_sustained:
            # print("before detect_onsets_offsets_sustained", im.shape)
            if subsample_envelope:
                doo_ws = subsample_envelope_time_window if subsample_envelope_time_window is not None else 1
                doo_overlap = subsample_envelope_axon_group if subsample_envelope_axon_group is not None else 0.0
                doo_onset_th = subsample_envelope_statistic if subsample_envelope_statistic is not None else 0.5
                doo_offset_th = subsample_envelope_statistic if subsample_envelope_statistic is not None else 0.5

            else:
                doo_ws = detect_onsets_offsets_sustained_window_size if detect_onsets_offsets_sustained_window_size is not None else 20
                doo_overlap = detect_onsets_offsets_sustained_overlap if detect_onsets_offsets_sustained_overlap is not None else 0.0
                doo_onset_th = detect_onsets_offsets_sustained_onset_threshold if detect_onsets_offsets_sustained_onset_threshold is not None else 0.5
                doo_offset_th = detect_onsets_offsets_sustained_offset_threshold if detect_onsets_offsets_sustained_offset_threshold is not None else 0.5

            if binarised_subsample_envelope_and_detect_onsets_offsets_sustained_not_binary:
                bin_im = im.numpy()
                doo_im = detect_onsets_offsets_sustained_spikes(not_binary_im.numpy(), window_size=doo_ws, overlap=doo_overlap, onset_threshold=doo_onset_th, offset_threshold=doo_offset_th)
                # print("bin_im", bin_im.shape)
                # print("doo_im", doo_im.shape)
                im = np.concatenate((bin_im, doo_im), axis=1)
            else:
                im = detect_onsets_offsets_sustained_spikes(im.numpy(), window_size=doo_ws, overlap=doo_overlap, onset_threshold=doo_onset_th, offset_threshold=doo_offset_th)
            im = torch.tensor(im)
            # print("after detect_onsets_offsets_sustained", im.shape)


        if hierarchical_audio_processing:
            im = hierarchical_audio_pipeline(
                spike_matrix=im.numpy(),
                samp_hz=1000.0,
                max_axons=6000,
                env_time_window=10,
                freq_group=3,
                envelope_levels=(0.3, 0.6, 0.8),
                onset_windows=(10, 40),
                am_bands=((4, 16), (16, 64)),
                k_wta_k=1,
            )
            im = torch.tensor(im)

        if hierarchical_audio_processing2:
            im = hierarchical_audio_pipeline2(im.numpy())
            im = torch.tensor(im)

        if wav_to_binary_features:
            im = im.t().numpy().reshape(-1)
            if len(im) != 16000:
                # pad with zeros
                logger.warning(f"wav is of shape {im.shape}, but should be 16000, padding with zeros.")
                im = np.pad(im, (0, 16000 - len(im)), 'constant', constant_values=0)
            im = wav_to_binary_features_func(im, fs=16000, freq_pool=2, time_pool_ms=50, comp_alpha=4, q=0.8, output_all=False, max_features=None)
            im = torch.tensor(im)

        # now binarize the data

        if binarization_method == BinarizationMethod.THRESHOLD \
            or binarization_method == BinarizationMethod.THRESHOLD_POISSON \
                or binarization_method == BinarizationMethod.THRESHOLD_TEMPORAL:
            im = im > binarization_threshold
        elif binarization_method == BinarizationMethod.THRESHOLD_MEAN \
            or binarization_method == BinarizationMethod.THRESHOLD_MEAN_POISSON \
                or binarization_method == BinarizationMethod.THRESHOLD_MEAN_TEMPORAL:
            im = im > im.mean()
        elif binarization_method == BinarizationMethod.NONE:
            pass
        elif binarization_method == BinarizationMethod.BUCKETIZE_POISSON \
            or binarization_method == BinarizationMethod.BUCKETIZE_TEMPORAL \
                or binarization_method == BinarizationMethod.BUCKETIZE_RAW:
            im = torch.bucketize(im, torch.linspace(0, 1, bucketize_bins))
        else:
            raise ValueError("Unsupported binarization method")

        # now convert the data to spikes
        
        actual_count_axons = count_axons
        actual_count_const_firing_axons = count_const_firing_axons
        actual_count_axons_for_data = actual_count_axons - actual_count_const_firing_axons
        if vertical_duplicate:
            actual_count_axons = actual_count_axons // 2
            actual_count_const_firing_axons = actual_count_const_firing_axons // 2         
            actual_count_axons_for_data = actual_count_axons - actual_count_const_firing_axons           

        if binarization_method == BinarizationMethod.BUCKETIZE_POISSON \
            or binarization_method == BinarizationMethod.THRESHOLD_POISSON \
                or binarization_method == BinarizationMethod.THRESHOLD_MEAN_POISSON \
                    or binarization_method == BinarizationMethod.BUCKETIZE_TEMPORAL \
                        or binarization_method == BinarizationMethod.THRESHOLD_TEMPORAL \
                            or binarization_method == BinarizationMethod.THRESHOLD_MEAN_TEMPORAL:
            # every pixel is a different axon (maybe more than one axon per pixel)
            im = im.flatten()

            if subsample_modulo is not None:
                # subsample the data to the right size
                im = im[::subsample_modulo]

            effective_bins = 2
            if binarization_method == BinarizationMethod.BUCKETIZE_POISSON \
                or binarization_method == BinarizationMethod.BUCKETIZE_TEMPORAL:
                effective_bins = bucketize_bins - 1

            effective_bins_normalization_constant = effective_bins - 1

            if binarization_method == BinarizationMethod.BUCKETIZE_POISSON \
                or binarization_method == BinarizationMethod.THRESHOLD_POISSON \
                    or binarization_method == BinarizationMethod.THRESHOLD_MEAN_POISSON:
                # poisson process with fr which is the pixel value

                if actual_count_axons_for_data % im.shape[0] != 0:
                    raise ValueError(f"actual_count_axons ({actual_count_axons}) - actual_count_const_firing_axons ({actual_count_const_firing_axons}) (which is {actual_count_axons_for_data}) in ms must be divisible by data height ({im.shape[0]})")

                # print("actual_count_axons_for_data", actual_count_axons_for_data)
                # print("im.shape[0]", im.shape[0])
                axons_per_pixel = actual_count_axons_for_data // im.shape[0]
                # print("axons_per_pixel", axons_per_pixel)

                fr = im.clone()

                # duplicate each row axons_per_pixel times
                fr = fr.repeat_interleave(axons_per_pixel)

                if vertical_duplicate:
                    fr = torch.cat((fr, fr), dim=0)

                if to_presampled:
                    spike_encoding = np.array([[np.inf] for _ in range(fr.shape[0])])
                    rate_encoding = (average_firing_rate_per_axon/1000)*fr.numpy()/effective_bins_normalization_constant

                else:
                    im = torch.zeros((fr.shape[0], time_in_ms))

                    # TODO: use average_firing_rate_per_axon
                    for i in range(fr.shape[0]):
                        im[i] = torch.rand((time_in_ms,)) < (fr[i]/1000)
    
            elif binarization_method == BinarizationMethod.BUCKETIZE_TEMPORAL \
                or binarization_method == BinarizationMethod.THRESHOLD_TEMPORAL \
                    or binarization_method == BinarizationMethod.THRESHOLD_MEAN_TEMPORAL:
                # temporal encoding of the pixels, every pixel is a set of different axons with bursts that are the pixel value
                
                if actual_count_axons_for_data % im.shape[0] != 0:
                    raise ValueError(f"actual_count_axons ({actual_count_axons}) - actual_count_const_firing_axons ({actual_count_const_firing_axons}) (which is {actual_count_axons_for_data}) in ms must be divisible by data height ({im.shape[0]})")

                # print("actual_count_axons_for_data", actual_count_axons_for_data)
                count_variables = im.shape[0]
                # print("count_variables", count_variables)
                # print("im.shape[0]", im.shape[0])
                axons_per_pixel = actual_count_axons_for_data // count_variables
                # print("actual_count_axons_for_data", actual_count_axons_for_data)
                # print("axons_per_pixel", axons_per_pixel)
                # print("axons_per_pixel", axons_per_pixel)
                distance_between_temporal_locations = (time_in_ms - temporal_delay_from_start) // axons_per_pixel
                # print(f"time_in_ms", time_in_ms)
                # print(f"temporal_delay_from_start", temporal_delay_from_start)
                # print("distance_between_temporal_locations", distance_between_temporal_locations)
                
                burst_fr = im.clone()

                if not arange_spikes:
                    if temporal_locations_file is None:
                        raise ValueError("temporal_locations_file must be specified if arange_spikes is False")
                    
                    if not os.path.exists(temporal_locations_file):
                        # create the temporal locations file, choose random locations
                        random_temporal_locations = np.random.choice(np.arange(temporal_delay_from_start, time_in_ms), size=(actual_count_axons_for_data,), replace=True)

                        np.save(temporal_locations_file, random_temporal_locations)   
                    
                    temporal_locations = np.load(temporal_locations_file)
                else:
                    temporal_locations = np.tile(np.arange(temporal_delay_from_start, axons_per_pixel*distance_between_temporal_locations, distance_between_temporal_locations), count_variables).astype(int)
                    # print(f"temporal_locations.shape", temporal_locations.shape)

                burst_fr_and_temporal_location = torch.zeros((actual_count_axons_for_data, time_in_ms))
                # print("burst_fr_and_temporal_location.shape", burst_fr_and_temporal_location.shape)

                for i in range(actual_count_axons_for_data // axons_per_pixel):
                    for j in range(axons_per_pixel):
                        # old version
                        # burst_fr_and_temporal_location[i*axons_per_pixel+j, temporal_delay_from_start+j*distance_between_temporal_locations] = burst_fr[i]
                        burst_fr_and_temporal_location[i*axons_per_pixel+j, temporal_locations[i*axons_per_pixel+j]] = burst_fr[i]

                if vertical_duplicate:
                    burst_fr_and_temporal_location = torch.cat((burst_fr_and_temporal_location, burst_fr_and_temporal_location), dim=0)
                if to_presampled:
                    spike_encoding = np.array([np.nonzero(row)[0] or np.array([np.inf]) for row in burst_fr_and_temporal_location.numpy()])
                    rate_encoding = np.array([(average_burst_firing_rate_per_axon/1000)*np.max(row) / effective_bins_normalization_constant for row in burst_fr_and_temporal_location.numpy()])

                else:
                    r_0 = bg_activity_hz
                    stim_on = 0
                    stim_off = time_in_ms
                    num_t = 1 # TODO: change if needed
                    r_max = average_firing_rate_per_axon
                    s = 1
                    t_on = 0
                    t_off = 0
                    sigma = jitter*s*1e-3*r_max*(stim_off - stim_on)/num_t
                    pre_syn = sequences.PreSyn(r_0, sigma)

                    im = torch.zeros((burst_fr_and_temporal_location.shape[0], time_in_ms))

                    for i in range(burst_fr_and_temporal_location.shape[0]):
                        row = burst_fr_and_temporal_location[i].numpy()
                        spike_times = np.nonzero(row)[0]
                        r = r_max * np.max(row) / effective_bins_normalization_constant
                        generated_spike_times = pre_syn.spike_train(t_on, t_off, stim_on, stim_off, s, r, spike_times)
                        im[i, generated_spike_times.astype(int)] = 1

        else:
            # the data itself is the spikes, but we need to stretch it to the right shape and then subsample it

            if kron:
                if time_in_ms % im.shape[1] != 0:
                    raise ValueError(f"Time in ms ({time_in_ms}) must be divisible by data width ({im.shape[1]})")

                if (actual_count_axons_for_data) % im.shape[0] != 0:
                    raise ValueError(f"actual_count_axons ({actual_count_axons}) - actual_count_const_firing_axons ({actual_count_const_firing_axons}) (which is {actual_count_axons_for_data}) in ms must be divisible by data height ({im.shape[0]})")
                
                temporal_extent_factor = time_in_ms // im.shape[1]
                spatial_extent_factor = (actual_count_axons_for_data) // im.shape[0]

                im = torch.kron(im, torch.ones((spatial_extent_factor, temporal_extent_factor), dtype=bool))

            if subsample_modulo is not None:
                raise ValueError("subsample_modulo is not supported for this binarization method")

            # applying transformations of spikes -> spikes

            if reduce_fr:
                im = reduce_long_sequences(im.numpy(), run_threshold=5, keep_probability=0.5)
                im = torch.from_numpy(im)

            if temporal_adaptation:
                im = temporal_adaptation_spikes(im.numpy(), alpha=0.95, threshold=0.2)
                im = torch.from_numpy(im)

            if to_presampled:
                raise ValueError("to_presampled is not supported for this binarization method")

            if vertical_duplicate:
                im = torch.cat((subsample_to_spikes(im, average_firing_rate_per_axon, bg_activity_hz), subsample_to_spikes(im, average_firing_rate_per_axon, bg_activity_hz)), dim=0)
            else:
                im = subsample_to_spikes(im, average_firing_rate_per_axon, bg_activity_hz)


        # pad with ones and zeros on top (for "bias" learning)

        if to_presampled:
            top_spike_encoding = np.array([[np.inf] for _ in range(actual_count_const_firing_axons)])
            top_rate_encoding = np.array([average_firing_rate_per_axon / 1000 if i < actual_count_const_firing_axons//2 else 0 for i in range(actual_count_const_firing_axons)])

            if vertical_duplicate:
                spike_encoding_up = spike_encoding[:spike_encoding.shape[0]//2]
                spike_encoding_down = spike_encoding[spike_encoding.shape[0]//2:]

                spike_encoding = np.concatenate((top_spike_encoding, spike_encoding_up, top_spike_encoding, spike_encoding_down), axis=0)

                rate_encoding_up = rate_encoding[:rate_encoding.shape[0]//2]
                rate_encoding_down = rate_encoding[rate_encoding.shape[0]//2:]

                rate_encoding = np.concatenate((top_rate_encoding, rate_encoding_up, top_rate_encoding, rate_encoding_down), axis=0)

            else:
                spike_encoding = np.concatenate((top_spike_encoding, spike_encoding), axis=0)
                rate_encoding = np.concatenate((top_rate_encoding, rate_encoding), axis=0)

        else:
            top_pad_train = torch.ones((actual_count_const_firing_axons, im.shape[1]), dtype=bool)
            top_pad_train[-actual_count_const_firing_axons//2:,:] = 0

            if vertical_duplicate:
                top_pad_train_2 = top_pad_train.clone()

                im_up = im[:im.shape[0]//2]
                im_down = im[im.shape[0]//2:]

                top_pad_train_subsampled = subsample_to_spikes(torch.tile(top_pad_train, [1,1]), average_firing_rate_per_axon, bg_activity_hz)
                top_pad_train_2_subsampled = subsample_to_spikes(torch.tile(top_pad_train_2, [1,1]), average_firing_rate_per_axon, bg_activity_hz)

                im = torch.cat((top_pad_train_subsampled, im_up, top_pad_train_2_subsampled, im_down), dim=0)

            else:
                top_pad_train_subsampled = subsample_to_spikes(torch.tile(top_pad_train, [1,1]), average_firing_rate_per_axon, bg_activity_hz)

                im = torch.cat((top_pad_train_subsampled, im), dim=0)

        # TODO: add synaptic unreliability (release_probability)?

        if to_presampled:
            return spike_encoding, rate_encoding

        else:
            return im

    params_dict = {
        "h_crop_range": h_crop_range,
        "w_crop_range": w_crop_range,
        "gabor_method": gabor_method,
        "vertical_duplicate": vertical_duplicate,
        "binarization_method": binarization_method,
        "binarization_threshold": binarization_threshold,
        "count_axons": count_axons,
        "count_const_firing_axons": count_const_firing_axons,
        "time_in_ms": time_in_ms,
        "average_firing_rate_per_axon": average_firing_rate_per_axon,
        "bg_activity_hz": bg_activity_hz,
        "did_gabor": did_gabor,
        "bucketize_bins": bucketize_bins,
        "temporal_delay_from_start": temporal_delay_from_start,
        "to_presampled": to_presampled,
        "trial": trial,
        "average_burst_firing_rate_per_axon": average_burst_firing_rate_per_axon,
        "jitter": jitter, 
        "arange_spikes": arange_spikes,
        "temporal_locations_file": temporal_locations_file,
        "gabor_c1_only_unit_0": gabor_c1_only_unit_0,
        "subsample_modulo": subsample_modulo,
        "reduce_fr": reduce_fr,
        "temporal_adaptation": temporal_adaptation, 
        "detect_onsets_offsets": detect_onsets_offsets,
        "detect_onsets_offsets_window_size": detect_onsets_offsets_window_size,
        "detect_onsets_offsets_threshold": detect_onsets_offsets_threshold,
        "kron": kron,
        "detect_onsets_offsets_sustained": detect_onsets_offsets_sustained,
        "detect_onsets_offsets_sustained_window_size": detect_onsets_offsets_sustained_window_size,
        "detect_onsets_offsets_sustained_overlap": detect_onsets_offsets_sustained_overlap,
        "detect_onsets_offsets_sustained_onset_threshold": detect_onsets_offsets_sustained_onset_threshold,
        "detect_onsets_offsets_sustained_offset_threshold": detect_onsets_offsets_sustained_offset_threshold,
        "envelope_extraction": envelope_extraction,
        "envelope_extraction_kernel_size": envelope_extraction_kernel_size,
        "envelope_extraction_threshold": envelope_extraction_threshold,
        "subsample_envelope": subsample_envelope,
        "subsample_envelope_time_window": subsample_envelope_time_window,
        "subsample_envelope_axon_group": subsample_envelope_axon_group,
        "subsample_envelope_statistic": subsample_envelope_statistic,
        "binarise_subsample_envelope": binarise_subsample_envelope,
        "binarised_subsample_envelope_and_detect_onsets_offsets_sustained_not_binary": binarised_subsample_envelope_and_detect_onsets_offsets_sustained_not_binary,
        "hierarchical_audio_processing": hierarchical_audio_processing,
        "hierarchical_audio_processing2": hierarchical_audio_processing2,
        "wav_to_binary_features": wav_to_binary_features,
        "ds_shorter_name": ds_shorter_name,
    }

    if ds_shorter_name:
        transform_name = f"s"
        if h_crop_range is not None:
            transform_name += f"_hc_{h_crop_range}"
        if w_crop_range is not None:
            transform_name += f"_wc_{w_crop_range}"
        if gabor_method is not GaborMethod.NONE:
            transform_name += f"_g_{gabor_method}"
        if vertical_duplicate:
            transform_name += f"_vd_{vertical_duplicate}"
        if binarization_method is not BinarizationMethod.NONE:
            transform_name += f"_b_{binarization_method}"
        if binarization_threshold is not None:
            transform_name += f"_bt_{binarization_threshold}"
        if count_axons is not None:
            transform_name += f"_a_{count_axons}"
        if count_const_firing_axons is not None:
            transform_name += f"_ac_{count_const_firing_axons}"
        if time_in_ms is not None:
            transform_name += f"_tim_{time_in_ms}"
        if average_firing_rate_per_axon is not None:
            transform_name += f"_fr_{average_firing_rate_per_axon}"
        if did_gabor:
            transform_name += f"_dg_{did_gabor}"
        if bucketize_bins is not None:
            transform_name += f"_bb_{bucketize_bins}"
        if temporal_delay_from_start is not None:
            transform_name += f"_td_{temporal_delay_from_start}"
        if to_presampled:
            transform_name += f"_tp_{to_presampled}"
        if trial is not None:
            transform_name += f"_t_{trial}"
        if average_burst_firing_rate_per_axon is not None:
            transform_name += f"_bfr_{average_burst_firing_rate_per_axon}"
        if bg_activity_hz is not None:
            transform_name += f"_bg_{bg_activity_hz}"
        if jitter is not None:
            transform_name += f"_j_{jitter}"
    else:
        transform_name = f"hc_{h_crop_range}_wc_{w_crop_range}_g_{gabor_method}_vd_{vertical_duplicate}" \
            + f"_b_{binarization_method}_bt_{binarization_threshold}_a_{count_axons}_ac_{count_const_firing_axons}" \
            + f"_tim_{time_in_ms}_fr_{average_firing_rate_per_axon}_dg_{did_gabor}" \
            + f"_bb_{bucketize_bins}_td_{temporal_delay_from_start}_tp_{to_presampled}_t_{trial}" \
            + f"_bfr_{average_burst_firing_rate_per_axon}_bg_{bg_activity_hz}_j_{jitter}"

    # TODO: a hack, remove in the future
    if arange_spikes is False:
        transform_name += f"_arsp_{arange_spikes}"

    if gabor_c1_only_unit_0 is not None:
        transform_name += f"_c1ou0_{gabor_c1_only_unit_0}"

    if subsample_modulo is not None:
        transform_name += f"_sm_{subsample_modulo}"

    if reduce_fr:
        transform_name += f"_rf_{reduce_fr}"

    if temporal_adaptation:
        transform_name += f"_ta_{temporal_adaptation}"

    if detect_onsets_offsets:
        transform_name += f"_do_{detect_onsets_offsets}"
    if detect_onsets_offsets_window_size is not None:
        transform_name += f"_dow_{detect_onsets_offsets_window_size}"
    if detect_onsets_offsets_threshold is not None:
        transform_name += f"_doth_{detect_onsets_offsets_threshold}"

    if kron:
        transform_name += f"_kron_{kron}"

    if detect_onsets_offsets_sustained:
        transform_name += f"_dos_{detect_onsets_offsets_sustained}"
    if detect_onsets_offsets_sustained_window_size is not None:
        transform_name += f"_dosw_{detect_onsets_offsets_sustained_window_size}"
    if detect_onsets_offsets_sustained_overlap is not None:
        transform_name += f"_doso_{detect_onsets_offsets_sustained_overlap}"
    if detect_onsets_offsets_sustained_onset_threshold is not None:
        transform_name += f"_dosot_{detect_onsets_offsets_sustained_onset_threshold}"
    if detect_onsets_offsets_sustained_offset_threshold is not None:
        transform_name += f"_dosof_{detect_onsets_offsets_sustained_offset_threshold}"

    if envelope_extraction:
        transform_name += f"_ee_{envelope_extraction}"
    if envelope_extraction_kernel_size is not None:
        transform_name += f"_eeks_{envelope_extraction_kernel_size}"
    if envelope_extraction_threshold is not None:
        transform_name += f"_eeth_{envelope_extraction_threshold}"

    if subsample_envelope:
        transform_name += f"_se_{subsample_envelope}"
    if subsample_envelope_time_window is not None:
        transform_name += f"_setw_{subsample_envelope_time_window}"
    if subsample_envelope_axon_group is not None:
        transform_name += f"_seag_{subsample_envelope_axon_group}"
    if subsample_envelope_statistic is not None:
        transform_name += f"_ses_{subsample_envelope_statistic}"

    if binarise_subsample_envelope:
        transform_name += f"_bse_{binarise_subsample_envelope}"

    if binarised_subsample_envelope_and_detect_onsets_offsets_sustained_not_binary:
        transform_name += f"_bseando_{binarised_subsample_envelope_and_detect_onsets_offsets_sustained_not_binary}"

    if hierarchical_audio_processing:
        transform_name += f"_hap_{hierarchical_audio_processing}"

    if hierarchical_audio_processing2:
        transform_name += f"_hap2_{hierarchical_audio_processing2}"

    if wav_to_binary_features:
        transform_name += f"_wbf_{wav_to_binary_features}"

    return transform_2d_to_spikes, params_dict, transform_name


class SpikingDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, dataset_parent_path=None, dataset_path=None, original_dataset=None, spiking_dataset_basename=None,
    keep_labels=None, transform_labels=None, one_hot_size=-1, max_count_samples=None, initial_image_size=None, extra_label_information=False, **kwargs):
        super(SpikingDataset, self).__init__()
        self.dataset_parent_path = dataset_parent_path
        self.dataset_path = dataset_path
        self.spiking_dataset_basename = spiking_dataset_basename
        self.presampled = kwargs['to_presampled']
        self.transform_2d_to_spikes, self.params_dict, self.transform_name = get_transform_2d_to_spikes(**kwargs)
        self.transform_name_suffix = f"_msc_{max_count_samples}_kl_{keep_labels}_ohs_{one_hot_size}_tl_{True if transform_labels is not None else False}"
        self.extra_label_information = extra_label_information
        if initial_image_size is not None:
            self.transform_name_suffix += f"_iis_{initial_image_size}"

        if self.dataset_path is None:
            if self.dataset_parent_path is None:
                raise ValueError("Must specify either dataset_path or dataset_parent_path")
            self.dataset_path = os.path.join(self.dataset_parent_path, self.spiking_dataset_basename + "_" + self.transform_name + self.transform_name_suffix)

        logger.info(f"self.dataset_path is {self.dataset_path}")

        if not os.path.exists(self.dataset_path):
            if original_dataset is None:
                raise ValueError("Either original_dataset or dataset_path must be specified")

            logger.info(f"Dataset not found, generating... into {self.dataset_path}")
            os.makedirs(self.dataset_path)

            additional_args = {}
            temporal_locations_file = self.params_dict['temporal_locations_file']
            if temporal_locations_file is not None and not os.path.exists(temporal_locations_file):
                raise ValueError(f"temporal_locations_file {temporal_locations_file} was set, but it does not exist")
            if temporal_locations_file is None:
                temporal_locations_file = os.path.join(self.dataset_path, "temporal_locations.npy")
                additional_args['temporal_locations_file'] = temporal_locations_file

            label_predicate = lambda x: x in keep_labels if keep_labels is not None else lambda x: True
            original_dataset_with_labels = FilterLabelsDataset(original_dataset, label_predicate, one_hot=True, one_hot_size=one_hot_size, transform_labels=transform_labels,
                                                                max_count_samples=max_count_samples, extra_label_information=extra_label_information, return_original_index=True)
            # original_dataset_with_labels = FilterLabelsDataset(original_dataset, label_predicate, one_hot=True, one_hot_size=one_hot_size, transform_labels=transform_labels)
            self.one_hot_size = original_dataset_with_labels.one_hot_size

            dataset_indices = range(len(original_dataset_with_labels))

            count_samples = 0
            # go over all images in the original dataset and convert them to spikes presampled
            for i, ind in enumerate(dataset_indices):
                item, label, original_index = original_dataset_with_labels[ind]
                transform_output = self.transform_2d_to_spikes(item, additional_args=additional_args)
                if self.presampled:
                    spike_encoding, rate_encoding = transform_output
                    np.save(os.path.join(self.dataset_path, f"{i}_spikes.npy"), spike_encoding)
                    np.save(os.path.join(self.dataset_path, f"{i}_rate.npy"), rate_encoding)
                else:
                    spikes = transform_output
                    np.save(os.path.join(self.dataset_path, f"{i}_spikes.npy"), spikes)

                if self.extra_label_information:
                    label, label_extra = label
                    np.save(os.path.join(self.dataset_path, f"{i}_label_extra.npy"), label_extra)
                np.save(os.path.join(self.dataset_path, f"{i}_label.npy"), label)

                np.save(os.path.join(self.dataset_path, f"{i}_index.npy"), ind)
                np.save(os.path.join(self.dataset_path, f"{i}_original_index.npy"), original_index)
                count_samples += 1

                if i % 100 == 0:
                    logger.info(f"Generated {i} samples")

            self.count_samples = count_samples

            # save number of samples
            with open(os.path.join(self.dataset_path, "num_samples.txt"), "w") as f:
                f.write(str(self.count_samples))

            # save one hot size
            with open(os.path.join(self.dataset_path, "one_hot_size.txt"), "w") as f:
                f.write(str(self.one_hot_size))

            # save params dict
            with open(os.path.join(self.dataset_path, "params_dict.json"), "w") as f:
                json.dump(self.params_dict, f)
        else:
            logger.info(f"Dataset found, loading... from {self.dataset_path}")

            with open(os.path.join(self.dataset_path, "num_samples.txt"), "r") as f:
                self.count_samples = int(f.read())

            one_hot_size_file_path = os.path.join(self.dataset_path, "one_hot_size.txt")
            if os.path.exists(one_hot_size_file_path):
                with open(one_hot_size_file_path, "r") as f:
                    self.one_hot_size = int(f.read())
            else:
                # TODO: a hack that might not always work, remove one day
                self.one_hot_size = 2

            params_dict_file_path = os.path.join(self.dataset_path, "params_dict.json")
            if os.path.exists(params_dict_file_path):
                with open(params_dict_file_path, "r") as f:
                    self.params_dict = json.load(f)
            else:
                logger.warning(f"params_dict.json not found in {self.dataset_path}, using params_dict from arguments")

    def __len__(self):
        return self.count_samples

    def __getitem__(self, index):
        label = np.load(os.path.join(self.dataset_path, f"{index}_label.npy"))
        if self.extra_label_information:
            label_extra = np.load(os.path.join(self.dataset_path, f"{index}_label_extra.npy"))
            label = (label, label_extra)
        
        if self.presampled:
            spike_encoding = np.load(os.path.join(self.dataset_path, f"{index}_spikes.npy"))
            rate_encoding = np.load(os.path.join(self.dataset_path, f"{index}_rate.npy"))

            time_in_ms = self.params_dict['time_in_ms']

            r_0 = self.params_dict['bg_activity_hz']
            jitter = self.params_dict['jitter']
            stim_on = 0 # TODO: change if needed
            stim_off = time_in_ms
            num_t = 1 # TODO: change if needed
            r_max = self.params_dict['average_firing_rate_per_axon']
            t_on = 0
            t_off = 0

            binarization_method = self.params_dict['binarization_method']
            vertical_duplicate = self.params_dict['vertical_duplicate']
            count_const_firing_axons = self.params_dict['count_const_firing_axons']

            spikes = np.zeros((rate_encoding.shape[0], time_in_ms))

            for i in range(rate_encoding.shape[0]):
                if binarization_method == BinarizationMethod.BUCKETIZE_POISSON \
                    or binarization_method == BinarizationMethod.THRESHOLD_POISSON \
                        or binarization_method == BinarizationMethod.THRESHOLD_MEAN_POISSON:
                    s = 0
                elif binarization_method == BinarizationMethod.BUCKETIZE_TEMPORAL \
                    or binarization_method == BinarizationMethod.THRESHOLD_TEMPORAL \
                        or binarization_method == BinarizationMethod.THRESHOLD_MEAN_TEMPORAL:
                    if vertical_duplicate:
                        actual_count_const_firing_axons = count_const_firing_axons // 2
                        s = 0 if i % (rate_encoding.shape[0]//2) < actual_count_const_firing_axons else 1
                    else:
                        s = 0 if i < count_const_firing_axons else 1
                r = rate_encoding[i]
                spike_times = spike_encoding[i]
                
                sigma = jitter*s*1e-3*r_max*(stim_off - stim_on)/num_t
                pre_syn = sequences.PreSyn(r_0, sigma)
                generated_spike_times = pre_syn.spike_train(t_on, t_off, stim_on, stim_off, s, r, spike_times)
                spikes[i, generated_spike_times.astype(int)] = 1
        else:
            spikes = np.load(os.path.join(self.dataset_path, f"{index}_spikes.npy"))
                
        return spikes, label
