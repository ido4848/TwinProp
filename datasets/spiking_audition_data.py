import numpy as np
import sys
import pathlib
import matplotlib.pyplot as plt
import torch
import os
from enum import IntEnum
from tqdm import tqdm
from torch.nn import functional as F
from torchvision.datasets import MNIST, CIFAR10, ImageFolder
from torchvision import transforms
from torchaudio.datasets import SPEECHCOMMANDS
import tonic
import tonic.transforms as tonic_transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split

import logging

from datasets.to_spikes import get_transform_2d_to_spikes, BinarizationMethod, SpikingDataset
from utils.utils import setup_logger

logger = logging.getLogger(__name__)

class Shd():
    def __init__(self, keep_labels=None, transform_labels=None, one_hot_size=-1, valid_percentage=0.2, split_seed=None,
     time_in_ms=200, count_axons=1500, binarization_method=BinarizationMethod.NONE, count_const_firing_axons=100, average_firing_rate_per_axon=None,
       max_count_samples=None, to_presampled=True, **kwargs):
        self.keep_labels = keep_labels
        self.transform_labels = transform_labels
        self.one_hot_size = one_hot_size
        self.valid_percentage = valid_percentage
        self.split_seed = split_seed
        
        kwargs = dict(kwargs)
        kwargs['time_in_ms'] = time_in_ms
        kwargs['count_axons'] = count_axons
        kwargs['count_const_firing_axons'] = count_const_firing_axons
        kwargs['average_firing_rate_per_axon'] = average_firing_rate_per_axon
        kwargs['binarization_method'] = binarization_method
        kwargs['vertical_duplicate'] = True
        
        self.kwargs = kwargs
        self.time_in_ms = kwargs['time_in_ms']
        self.count_axons = kwargs['count_axons']

        sensor_size = tonic.datasets.SHD.sensor_size
        frame_transform = tonic_transforms.ToFrame(sensor_size=sensor_size, n_time_bins=self.time_in_ms)

        def transpose_transform(im):
            return im.transpose(0,2).transpose(1,2)
        
        def binary_transform(im):
            return (im > 0).float()

        self.pre_transform = tonic_transforms.Compose([frame_transform, transforms.ToTensor(), transforms.Lambda(transpose_transform), transforms.Lambda(binary_transform)])

        shd_dataset = tonic.datasets.SHD(save_to="Data/shd_tonic", train=True, transform=self.pre_transform)

        self.dataset = SpikingDataset(dataset_parent_path='Data/',
            spiking_dataset_basename = f'shd', original_dataset=shd_dataset,
            keep_labels=keep_labels, transform_labels=transform_labels, one_hot_size=one_hot_size, max_count_samples=max_count_samples,
            to_presampled=to_presampled, **kwargs)

        self.one_hot_size = self.dataset.one_hot_size  

        self.valid_size = int(len(self.dataset) * self.valid_percentage)
        self.train_size = len(self.dataset) - self.valid_size
        if self.split_seed is None:
            self.dataset_train, self.dataset_valid = random_split(self.dataset, [self.train_size, self.valid_size])
        else:
            logger.info(f"Splitting dataset with seed {self.split_seed}")
            self.dataset_train, self.dataset_valid = random_split(self.dataset, [self.train_size, self.valid_size],
                generator=torch.Generator().manual_seed(split_seed))

    def get_ds_shape(self):
        return ((self.count_axons,), self.time_in_ms, (self.one_hot_size,), 1)

    def get_train_loader(self, batch_size, num_workers=1):
        return DataLoader(self.dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    def get_valid_loader(self, batch_size, num_workers=1):
        return DataLoader(self.dataset_valid, batch_size=batch_size, shuffle=False, num_workers=num_workers)

class NonSpikingShd():
    def __init__(self, *args, **kwargs):
        self.shd = Shd(*args, **kwargs)

    def get_ds_shape(self):
        return ((self.shd.count_axons, self.shd.time_in_ms), 1, (self.shd.one_hot_size,), 1)

    def get_train_loader(self, batch_size, num_workers=1):
        return DataLoader(self.shd.dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    def get_valid_loader(self, batch_size, num_workers=1):
        return DataLoader(self.shd.dataset_valid, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    def train_size(self):
        return len(self.shd.dataset_train)

    def valid_size(self):
        return len(self.shd.dataset_valid)

class Ssc():
    def __init__(self, keep_labels=None, transform_labels=None, one_hot_size=-1, valid_percentage=0.2, split_seed=None,
     time_in_ms=200, count_axons=1500, binarization_method=BinarizationMethod.NONE, count_const_firing_axons=100, average_firing_rate_per_axon=None,
       max_count_samples=None, to_presampled=True, **kwargs):
        self.keep_labels = keep_labels
        self.transform_labels = transform_labels
        self.one_hot_size = one_hot_size
        self.valid_percentage = valid_percentage
        self.split_seed = split_seed
        
        kwargs = dict(kwargs)
        kwargs['time_in_ms'] = time_in_ms
        kwargs['count_axons'] = count_axons
        kwargs['count_const_firing_axons'] = count_const_firing_axons
        kwargs['average_firing_rate_per_axon'] = average_firing_rate_per_axon
        kwargs['binarization_method'] = binarization_method
        kwargs['vertical_duplicate'] = True
        
        self.kwargs = kwargs
        self.time_in_ms = kwargs['time_in_ms']
        self.count_axons = kwargs['count_axons']

        sensor_size = tonic.datasets.SSC.sensor_size
        frame_transform = tonic_transforms.ToFrame(sensor_size=sensor_size, n_time_bins=self.time_in_ms)

        def transpose_transform(im):
            return im.transpose(0,2).transpose(1,2)
        
        def binary_transform(im):
            return (im > 0).float()

        self.pre_transform = tonic_transforms.Compose([frame_transform, transforms.ToTensor(), transforms.Lambda(transpose_transform), transforms.Lambda(binary_transform)])

        ssc_dataset = tonic.datasets.SSC(save_to="Data/ssc", split='train', transform=self.pre_transform)

        self.dataset = SpikingDataset(dataset_parent_path='Data/',
            spiking_dataset_basename = f'ssc', original_dataset=ssc_dataset,
            keep_labels=keep_labels, transform_labels=transform_labels, one_hot_size=one_hot_size, max_count_samples=max_count_samples,
            to_presampled=to_presampled, **kwargs)

        self.one_hot_size = self.dataset.one_hot_size  

        self.valid_size = int(len(self.dataset) * self.valid_percentage)
        self.train_size = len(self.dataset) - self.valid_size
        if self.split_seed is None:
            self.dataset_train, self.dataset_valid = random_split(self.dataset, [self.train_size, self.valid_size])
        else:
            logger.info(f"Splitting dataset with seed {self.split_seed}")
            self.dataset_train, self.dataset_valid = random_split(self.dataset, [self.train_size, self.valid_size],
                generator=torch.Generator().manual_seed(split_seed))

    def get_ds_shape(self):
        return ((self.count_axons,), self.time_in_ms, (self.one_hot_size,), 1)

    def get_train_loader(self, batch_size, num_workers=1):
        return DataLoader(self.dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    def get_valid_loader(self, batch_size, num_workers=1):
        return DataLoader(self.dataset_valid, batch_size=batch_size, shuffle=False, num_workers=num_workers)

class NonSpikingSsc():
    def __init__(self, *args, **kwargs):
        self.ssc = Ssc(*args, **kwargs)

    def get_ds_shape(self):
        return ((self.ssc.count_axons, self.ssc.time_in_ms), 1, (self.ssc.one_hot_size,), 1)

    def get_train_loader(self, batch_size, num_workers=1):
        return DataLoader(self.ssc.dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    def get_valid_loader(self, batch_size, num_workers=1):
        return DataLoader(self.ssc.dataset_valid, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    def train_size(self):
        return len(self.ssc.dataset_train)

    def valid_size(self):
        return len(self.ssc.dataset_valid)

class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None):
        super(self.__class__, self).__init__("Data/speech_commands", download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]

class SubsetSC2(torch.utils.data.dataset.Dataset):
    def __init__(self, subset: str = None, numerical_labels: bool = False, return_all: bool = True):
        self.subset_sc = SubsetSC(subset)
        self.numerical_labels = numerical_labels
        self.return_all = return_all

        self.labels = ["Backward", "Bed", "Bird", "Cat", "Dog", "Down", "Eight", "Five", "Follow",
                        "Forward", "Four", "Go", "Happy", "House", "Learn", "Left", "Marvin", "Nine",
                          "No", "Off", "On", "One", "Right", "Seven", "Sheila", "Six", "Stop", "Three",
                            "Tree", "Two", "Up", "Visual", "Wow", "Yes", "Zero"]

        self.labels = [label.lower() for label in self.labels]

    def __len__(self):
        return len(self.subset_sc)
    
    def __getitem__(self, index):
        waveform, sample_rate, label, speaker_id, utterance_number = self.subset_sc[index]
        if self.numerical_labels:
            label = self.labels.index(label.lower())
        if self.return_all:
            return waveform, sample_rate, label, speaker_id, utterance_number
        else:
            return waveform, label

# # Create training and testing split of the data. We do not use validation in this tutorial.
# train_set = SubsetSC("training")
# test_set = SubsetSC("testing")

# waveform, sample_rate, label, speaker_id, utterance_number = train_set[0]

class SpikingSc():
    def __init__(self, keep_labels=None, transform_labels=None, one_hot_size=-1, valid_percentage=0.2, split_seed=None,
     time_in_ms=200, count_axons=1500, binarization_method=BinarizationMethod.NONE, count_const_firing_axons=100, average_firing_rate_per_axon=None,
       max_count_samples=None, to_presampled=True, **kwargs):
        self.keep_labels = keep_labels
        self.transform_labels = transform_labels
        self.one_hot_size = one_hot_size
        self.valid_percentage = valid_percentage
        self.split_seed = split_seed
        
        kwargs = dict(kwargs)
        kwargs['time_in_ms'] = time_in_ms
        kwargs['count_axons'] = count_axons
        kwargs['count_const_firing_axons'] = count_const_firing_axons
        kwargs['average_firing_rate_per_axon'] = average_firing_rate_per_axon
        kwargs['binarization_method'] = binarization_method
        kwargs['vertical_duplicate'] = True
        
        self.kwargs = kwargs
        self.time_in_ms = kwargs['time_in_ms']
        self.count_axons = kwargs['count_axons']

        sc_dataset = SubsetSC2(subset='training', numerical_labels=True, return_all=False)

        self.dataset = SpikingDataset(dataset_parent_path='Data/',
            spiking_dataset_basename = f'spiking_sc', original_dataset=sc_dataset,
            keep_labels=keep_labels, transform_labels=transform_labels, one_hot_size=one_hot_size, max_count_samples=max_count_samples,
            to_presampled=to_presampled, **kwargs)

        self.one_hot_size = self.dataset.one_hot_size  

        self.valid_size = int(len(self.dataset) * self.valid_percentage)
        self.train_size = len(self.dataset) - self.valid_size
        if self.split_seed is None:
            self.dataset_train, self.dataset_valid = random_split(self.dataset, [self.train_size, self.valid_size])
        else:
            logger.info(f"Splitting dataset with seed {self.split_seed}")
            self.dataset_train, self.dataset_valid = random_split(self.dataset, [self.train_size, self.valid_size],
                generator=torch.Generator().manual_seed(split_seed))

    def get_ds_shape(self):
        return ((self.count_axons,), self.time_in_ms, (self.one_hot_size,), 1)

    def get_train_loader(self, batch_size, num_workers=1):
        return DataLoader(self.dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    def get_valid_loader(self, batch_size, num_workers=1):
        return DataLoader(self.dataset_valid, batch_size=batch_size, shuffle=False, num_workers=num_workers)

class NonSpikingSpikingSc():
    def __init__(self, *args, **kwargs):
        self.spiking_sc = SpikingSc(*args, **kwargs)

    def get_ds_shape(self):
        return ((self.spiking_sc.count_axons, self.spiking_sc.time_in_ms), 1, (self.spiking_sc.one_hot_size,), 1)

    def get_train_loader(self, batch_size, num_workers=1):
        return DataLoader(self.spiking_sc.dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    def get_valid_loader(self, batch_size, num_workers=1):
        return DataLoader(self.spiking_sc.dataset_valid, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    def train_size(self):
        return len(self.spiking_sc.dataset_train)

    def valid_size(self):
        return len(self.spiking_sc.dataset_valid)        