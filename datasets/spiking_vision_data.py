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
from torch.utils.data import DataLoader
from torch.utils.data import random_split

import logging

from datasets.filter_dataset import FilterLabelsDataset, FilterItemsDataset
from datasets.hmax import HMAX
from datasets import sequences
from datasets.to_spikes import get_transform_2d_to_spikes, BinarizationMethod, GaborMethod, SpikingDataset

from utils.utils import setup_logger

logger = logging.getLogger(__name__)

        
class SpikingMnist():
    def __init__(self, keep_labels=None, transform_labels=None, one_hot_size=-1, valid_percentage=0.2, split_seed=None,
     time_in_ms=420, count_axons=880, max_count_samples=None, to_presampled=True, **kwargs):
        
        kwargs = dict(kwargs)
        kwargs['time_in_ms'] = time_in_ms
        kwargs['count_axons'] = count_axons
        kwargs['vertical_duplicate'] = True
        
        self.kwargs = kwargs
        self.count_axons = kwargs['count_axons']
        self.time_in_ms = kwargs['time_in_ms']
        
        self.valid_percentage = valid_percentage
        self.split_seed = split_seed

        mnist_dataset = MNIST('Data/', train=True, download=True, transform=transforms.Compose(
                    [transforms.ToTensor()]))
        
        self.dataset = SpikingDataset(dataset_parent_path='Data/',
            spiking_dataset_basename = f'spiking_mnist', original_dataset=mnist_dataset,
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

class NonSpikingSpikingMnist():
    def __init__(self, *args, **kwargs):
        self.spiking_mnist = SpikingMnist(*args, **kwargs)

    def get_ds_shape(self):
        return ((self.spiking_mnist.count_axons, self.spiking_mnist.time_in_ms), 1, (self.spiking_mnist.one_hot_size,), 1)

    def get_train_loader(self, batch_size, num_workers=1):
        return DataLoader(self.spiking_mnist.dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    def get_valid_loader(self, batch_size, num_workers=1):
        return DataLoader(self.spiking_mnist.dataset_valid, batch_size=batch_size, shuffle=False, num_workers=num_workers)

class SpikingCatAndDog():
    def __init__(self, keep_labels=None, transform_labels=None, one_hot_size=-1, valid_percentage=0.2, split_seed=None,
     time_in_ms=200, gabor_method=GaborMethod.GABOR_C1, initial_image_size=None, max_count_samples=None, to_presampled=True, **kwargs):
        kwargs = dict(kwargs)

        if gabor_method == GaborMethod.GABOR_C1:
            kwargs['gabor_method'] = GaborMethod.GABOR_C1
            gabor_method_text = "c1"
            # kwargs['count_axons'] = 2200 * 2 # recommended
            # kwargs['count_const_firing_axons'] = 120 * 2 # recommended
            kwargs['time_in_ms'] = time_in_ms
            kwargs['vertical_duplicate'] = True
            image_size = initial_image_size or 256
        elif gabor_method == GaborMethod.GABOR_S1:
            kwargs['gabor_method'] = GaborMethod.GABOR_S1
            gabor_method_text = "s1"
            # kwargs['count_axons'] = 4200 * 2 # recommended
            # kwargs['count_const_firing_axons'] = 104 * 2 # recommended
            kwargs['time_in_ms'] = time_in_ms
            kwargs['vertical_duplicate'] = True
            image_size = initial_image_size or 64
        elif gabor_method == GaborMethod.GABOR_S2:
            kwargs['gabor_method'] = GaborMethod.GABOR_S2
            gabor_method_text = "s2"
            # kwargs['count_axons'] = 4200 * 2 # recommended
            # kwargs['count_const_firing_axons'] = 104 * 2 # recommended
            kwargs['time_in_ms'] = time_in_ms
            kwargs['vertical_duplicate'] = True
            image_size = initial_image_size or 64
        elif gabor_method == GaborMethod.GABOR_C2:
            kwargs['gabor_method'] = GaborMethod.GABOR_C2
            gabor_method_text = "c2"
            # kwargs['count_axons'] = 4200 * 2 # recommended
            # kwargs['count_const_firing_axons'] = 104 * 2 # recommended
            kwargs['time_in_ms'] = time_in_ms
            kwargs['vertical_duplicate'] = True
            image_size = initial_image_size or 256
        elif gabor_method == GaborMethod.NONE:
            kwargs['gabor_method'] = GaborMethod.NONE
            gabor_method_text = "none"
            # kwargs['count_axons'] = 2200 * 2 # recommended
            # kwargs['count_const_firing_axons'] = 152 * 2 # recommended
            kwargs['time_in_ms'] = time_in_ms
            kwargs['vertical_duplicate'] = True
            image_size = initial_image_size or 256
        else:
            raise ValueError("Unsupported gabor method")

        self.kwargs = kwargs
        self.count_axons = kwargs['count_axons']
        self.time_in_ms = kwargs['time_in_ms']
        
        self.valid_percentage = valid_percentage
        self.split_seed = split_seed

        cat_and_dog_path = "Data/cat_and_dog"

        self.pre_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize((0),(1))
            ])
        cat_and_dog_dataset = ImageFolder(root=f"{cat_and_dog_path}/training_set/training_set", transform=self.pre_transform)

        # TOOD: use?
        # def get_train_transform():
        # return T.Compose([
        #     T.Resize((256, 256)),
        #     T.Grayscale(num_output_channels=1),
        #     T.RandomHorizontalFlip(p=0.5),
        #     T.RandomRotation(15),
        #     T.RandomCrop(204),
        #     T.ToTensor(),
        #     T.Normalize((0, 0, 0),(1, 1, 1))
        # ])

        self.dataset = SpikingDataset(dataset_parent_path='Data/',
         spiking_dataset_basename = f'spiking_cats_and_dogs', original_dataset=cat_and_dog_dataset,
          keep_labels=keep_labels, transform_labels=transform_labels, one_hot_size=one_hot_size, max_count_samples=max_count_samples,
          to_presampled=to_presampled, initial_image_size=initial_image_size, **kwargs)

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

class OldSpikingCatAndDog():
    def __init__(self, keep_labels=None, transform_labels=None, one_hot_size=-1, valid_percentage=0.2, split_seed=None,
     time_in_ms=200, gabor_method=GaborMethod.GABOR_C1, initial_image_size=None, **kwargs):
        self.keep_labels = keep_labels
        self.transform_labels = transform_labels
        self.one_hot_size = one_hot_size
        self.valid_percentage = valid_percentage
        self.split_seed = split_seed

        kwargs = dict(kwargs)

        if gabor_method == GaborMethod.GABOR_C1:
            kwargs['gabor_method'] = GaborMethod.GABOR_C1
            gabor_method_text = "c1"
            # kwargs['count_axons'] = 2200 * 2 # recommended
            # kwargs['count_const_firing_axons'] = 120 * 2 # recommended
            kwargs['time_in_ms'] = time_in_ms
            kwargs['vertical_duplicate'] = True
            image_size = initial_image_size or 256
        elif gabor_method == GaborMethod.GABOR_S1:
            kwargs['gabor_method'] = GaborMethod.GABOR_S1
            gabor_method_text = "s1"
            # kwargs['count_axons'] = 4200 * 2 # recommended
            # kwargs['count_const_firing_axons'] = 104 * 2
            kwargs['time_in_ms'] = time_in_ms
            kwargs['vertical_duplicate'] = True
            image_size = initial_image_size or 64
        elif gabor_method == GaborMethod.GABOR_S2:
            kwargs['gabor_method'] = GaborMethod.GABOR_S2
            gabor_method_text = "s2"
            # kwargs['count_axons'] = 4200 * 2 # recommended
            # kwargs['count_const_firing_axons'] = 104 * 2
            kwargs['time_in_ms'] = time_in_ms
            kwargs['vertical_duplicate'] = True
            image_size = initial_image_size or 64
        elif gabor_method == GaborMethod.GABOR_C2:
            kwargs['gabor_method'] = GaborMethod.GABOR_C2
            gabor_method_text = "c2"
            # kwargs['count_axons'] = 4200 * 2 # recommended
            # kwargs['count_const_firing_axons'] = 104 * 2
            kwargs['time_in_ms'] = time_in_ms
            kwargs['vertical_duplicate'] = True
            image_size = initial_image_size or 256
        elif gabor_method == GaborMethod.NONE:
            kwargs['gabor_method'] = GaborMethod.NONE
            gabor_method_text = "none"
            # kwargs['count_axons'] = 2200 * 2 # recommended
            # kwargs['count_const_firing_axons'] = 152 * 2
            kwargs['time_in_ms'] = time_in_ms
            kwargs['vertical_duplicate'] = True
            image_size = initial_image_size or 256
        else:
            raise ValueError("Unsupported gabor method")

        self.kwargs = kwargs
        self.count_axons = kwargs['count_axons']
        self.time_in_ms = kwargs['time_in_ms']

        preprocessed_cat_and_dog_path = f"Data/cat_and_dog_gabor_{gabor_method_text}_image_size_{image_size}_time_in_ms_{time_in_ms}"

        if os.path.exists(preprocessed_cat_and_dog_path):
            logging.info("Loading already preprocessed cat and dog dataset from {preprocessed_cat_and_dog_path}")
            self.kwargs['did_gabor'] = True
            kwargs['did_gabor'] = True
            
            self.transform_image_to_spikes, self.params_dict, self.transform_name = get_transform_2d_to_spikes(**kwargs)
            self.full_transform = transforms.Compose([
                    transforms.Grayscale(num_output_channels=1),
                    transforms.ToTensor(),
                    transforms.Lambda(self.transform_image_to_spikes)
                ])
            dataset = ImageFolder(root=f"{preprocessed_cat_and_dog_path}/training_set/training_set", transform=self.full_transform)

        else:
            cat_and_dog_path = "Data/cat_and_dog"

            self.transform_image_to_spikes, self.params_dict, self.transform_name = get_transform_2d_to_spikes(**kwargs)
            self.full_transform = transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.Grayscale(num_output_channels=1),
                    transforms.ToTensor(),
                    transforms.Normalize((0),(1)),
                    transforms.Lambda(self.transform_image_to_spikes)
                ])
            dataset = ImageFolder(root=f"{cat_and_dog_path}/training_set/training_set", transform=self.full_transform)

        # TODO: use?
        # normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                              std=[0.229, 0.224, 0.225])

        # # TODO: maybe also transforms.RandomHorizontalFlip(),
        #     #transforms.RandomCrop(32, 4),
        # train_transforms_list = [
        #         transforms.RandomHorizontalFlip(),
        #         transforms.RandomCrop(32, 4),
        #         transforms.ToTensor(),]
        # if normalize:
        #     train_transforms_list.append(normalize_transform)
        # else:
        #     logger.info("Not normalizing train")
        # train_transform = transforms.Compose(train_transforms_list)

        # valid_transforms_list = [transforms.ToTensor(),]
        # if normalize:
        #     valid_transforms_list.append(normalize_transform)
        # else:         
        #     logger.info("Not normalizing valid")
        # valid_transform = transforms.Compose(valid_transforms_list)

        # TODO: use?
        # def get_train_transform():
        # return T.Compose([
        #     T.Resize((256, 256)),
        #     T.Grayscale(num_output_channels=1),
        #     T.RandomHorizontalFlip(p=0.5),
        #     T.RandomRotation(15),
        #     T.RandomCrop(204),
        #     T.ToTensor(),
        #     T.Normalize((0, 0, 0),(1, 1, 1))
        # ])


        self.label_predicate = lambda x: x in keep_labels if keep_labels is not None else lambda x: True
        dataset = FilterLabelsDataset(dataset, self.label_predicate, one_hot=True, one_hot_size=self.one_hot_size, transform_labels=transform_labels)
        self.dataset = dataset
        self.one_hot_size = dataset.one_hot_size
        self.dataset_size = len(self.dataset)

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

class NonSpikingSpikingCatAndDog():
    def __init__(self, *args, **kwargs):
        self.spiking_cat_and_dog = SpikingCatAndDog(*args, **kwargs)

    def get_ds_shape(self):
        return ((self.spiking_cat_and_dog.count_axons, self.spiking_cat_and_dog.time_in_ms), 1, (self.spiking_cat_and_dog.one_hot_size,), 1)

    def get_train_loader(self, batch_size, num_workers=1):
        return DataLoader(self.spiking_cat_and_dog.dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    def get_valid_loader(self, batch_size, num_workers=1):
        return DataLoader(self.spiking_cat_and_dog.dataset_valid, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    def train_size(self):
        return len(self.spiking_cat_and_dog.dataset_train)

    def valid_size(self):
        return len(self.spiking_cat_and_dog.dataset_valid)        

class SpikingAfhq():
    def __init__(self, keep_labels=None, transform_labels=None, one_hot_size=-1, valid_percentage=0.2, split_seed=None,
     time_in_ms=200, gabor_method=GaborMethod.GABOR_C1, initial_image_size=None, max_count_samples=None, to_presampled=True, **kwargs):
        kwargs = dict(kwargs)

        if gabor_method == GaborMethod.GABOR_C1:
            kwargs['gabor_method'] = GaborMethod.GABOR_C1
            gabor_method_text = "c1"
            # kwargs['count_axons'] = 2200 * 2 # recommended
            # kwargs['count_const_firing_axons'] = 120 * 2 # recommended
            kwargs['time_in_ms'] = time_in_ms
            kwargs['vertical_duplicate'] = True
            image_size = initial_image_size or 256
        elif gabor_method == GaborMethod.GABOR_S1:
            kwargs['gabor_method'] = GaborMethod.GABOR_S1
            gabor_method_text = "s1"
            # kwargs['count_axons'] = 4200 * 2 # recommended
            # kwargs['count_const_firing_axons'] = 104 * 2
            kwargs['time_in_ms'] = time_in_ms
            kwargs['vertical_duplicate'] = True
            image_size = initial_image_size or 64
        elif gabor_method == GaborMethod.GABOR_S2:
            kwargs['gabor_method'] = GaborMethod.GABOR_S2
            gabor_method_text = "s2"
            # kwargs['count_axons'] = 4200 * 2 # recommended
            # kwargs['count_const_firing_axons'] = 104 * 2
            kwargs['time_in_ms'] = time_in_ms
            kwargs['vertical_duplicate'] = True
            image_size = initial_image_size or 64
        elif gabor_method == GaborMethod.GABOR_C2:
            kwargs['gabor_method'] = GaborMethod.GABOR_C2
            gabor_method_text = "c2"
            # kwargs['count_axons'] = 4200 * 2 # recommended
            # kwargs['count_const_firing_axons'] = 104 * 2
            kwargs['time_in_ms'] = time_in_ms
            kwargs['vertical_duplicate'] = True
            image_size = initial_image_size or 256
        elif gabor_method == GaborMethod.NONE:
            kwargs['gabor_method'] = GaborMethod.NONE
            gabor_method_text = "none"
            # kwargs['count_axons'] = 2200 * 2 # recommended
            # kwargs['count_const_firing_axons'] = 152 * 2
            kwargs['time_in_ms'] = time_in_ms
            kwargs['vertical_duplicate'] = True
            image_size = initial_image_size or 256
        else:
            raise ValueError("Unsupported gabor method")

        self.kwargs = kwargs
        self.count_axons = kwargs['count_axons']
        self.time_in_ms = kwargs['time_in_ms']
        
        self.valid_percentage = valid_percentage
        self.split_seed = split_seed

        afhq_path = "Data/afhq"

        self.pre_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize((0),(1))
            ])
        afhq_dataset = ImageFolder(root=f"{afhq_path}/train", transform=self.pre_transform)

        # TOOD: use?
        # def get_train_transform():
        # return T.Compose([
        #     T.Resize((256, 256)),
        #     T.Grayscale(num_output_channels=1),
        #     T.RandomHorizontalFlip(p=0.5),
        #     T.RandomRotation(15),
        #     T.RandomCrop(204),
        #     T.ToTensor(),
        #     T.Normalize((0, 0, 0),(1, 1, 1))
        # ])

        self.dataset = SpikingDataset(dataset_parent_path='Data/',
         spiking_dataset_basename = f'spiking_afhq', original_dataset=afhq_dataset,
          keep_labels=keep_labels, transform_labels=transform_labels, one_hot_size=one_hot_size, max_count_samples=max_count_samples,
          to_presampled=to_presampled, initial_image_size=initial_image_size, **kwargs)

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

class OldSpikingAfhq():
    def __init__(self, keep_labels=None, transform_labels=None, one_hot_size=-1, valid_percentage=0.2, split_seed=None,
     time_in_ms=200, gabor_method=GaborMethod.GABOR_C1, initial_image_size=None, **kwargs):
        self.keep_labels = keep_labels
        self.transform_labels = transform_labels
        self.one_hot_size = one_hot_size
        self.valid_percentage = valid_percentage
        self.split_seed = split_seed

        kwargs = dict(kwargs)

        if gabor_method == GaborMethod.GABOR_C1:
            kwargs['gabor_method'] = GaborMethod.GABOR_C1
            gabor_method_text = "c1"
            # kwargs['count_axons'] = 2200 * 2 # recommended
            kwargs['count_const_firing_axons'] = 120 * 2 # recommended
            kwargs['time_in_ms'] = time_in_ms
            kwargs['vertical_duplicate'] = True
            image_size = initial_image_size or 256
        elif gabor_method == GaborMethod.GABOR_S1:
            kwargs['gabor_method'] = GaborMethod.GABOR_S1
            gabor_method_text = "s1"
            # kwargs['count_axons'] = 4200 * 2 # recommended
            kwargs['count_const_firing_axons'] = 104 * 2
            kwargs['time_in_ms'] = time_in_ms
            kwargs['vertical_duplicate'] = True
            image_size = initial_image_size or 64
        elif gabor_method == GaborMethod.GABOR_S2:
            kwargs['gabor_method'] = GaborMethod.GABOR_S2
            gabor_method_text = "s2"
            # kwargs['count_axons'] = 4200 * 2 # recommended
            kwargs['count_const_firing_axons'] = 104 * 2
            kwargs['time_in_ms'] = time_in_ms
            kwargs['vertical_duplicate'] = True
            image_size = initial_image_size or 64
        elif gabor_method == GaborMethod.GABOR_C2:
            kwargs['gabor_method'] = GaborMethod.GABOR_C2
            gabor_method_text = "c2"
            # kwargs['count_axons'] = 4200 * 2 # recommended
            kwargs['count_const_firing_axons'] = 104 * 2
            kwargs['time_in_ms'] = time_in_ms
            kwargs['vertical_duplicate'] = True
            image_size = initial_image_size or 256
        elif gabor_method == GaborMethod.NONE:
            kwargs['gabor_method'] = GaborMethod.NONE
            gabor_method_text = "none"
            # kwargs['count_axons'] = 2200 * 2 # recommended
            kwargs['count_const_firing_axons'] = 152 * 2
            kwargs['time_in_ms'] = time_in_ms
            kwargs['vertical_duplicate'] = True
            image_size = initial_image_size or 256
        else:
            raise ValueError("Unsupported gabor method")

        print("image_size", image_size)            

        self.kwargs = kwargs
        self.count_axons = kwargs['count_axons']
        self.time_in_ms = kwargs['time_in_ms']
        
        preprocessed_afhq_path = f"Data/afhq_gabor_{gabor_method_text}_image_size_{image_size}_time_in_ms_{time_in_ms}"

        if os.path.exists(preprocessed_afhq_path):
            logging.info("Loading already preprocessed afhq dataset from {preprocessed_afhq_path}")
            self.kwargs['did_gabor'] = True
            kwargs['did_gabor'] = True
            
            self.transform_image_to_spikes, self.params_dict, self.transform_name = get_transform_2d_to_spikes(**kwargs)
            self.full_transform = transforms.Compose([
                    transforms.Grayscale(num_output_channels=1),
                    transforms.ToTensor(),
                    transforms.Lambda(self.transform_image_to_spikes)
                ])
            dataset = ImageFolder(root=f"{preprocessed_afhq_path}/train", transform=self.full_transform)

        else:
            afhq_path = "Data/afhq"

            self.transform_image_to_spikes, self.params_dict, self.transform_name = get_transform_2d_to_spikes(**kwargs)
            self.full_transform = transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.Grayscale(num_output_channels=1),
                    transforms.ToTensor(),
                    transforms.Normalize((0),(1)),
                    transforms.Lambda(self.transform_image_to_spikes)
                ])
            dataset = ImageFolder(root=f"{afhq_path}/train", transform=self.full_transform)

        # TOOD: use?
        # def get_train_transform():
        # return T.Compose([
        #     T.Resize((256, 256)),
        #     T.Grayscale(num_output_channels=1),
        #     T.RandomHorizontalFlip(p=0.5),
        #     T.RandomRotation(15),
        #     T.RandomCrop(204),
        #     T.ToTensor(),
        #     T.Normalize((0, 0, 0),(1, 1, 1))
        # ])
        
        self.label_predicate = lambda x: x in keep_labels if keep_labels is not None else lambda x: True
        dataset = FilterLabelsDataset(dataset, self.label_predicate, one_hot=True, one_hot_size=self.one_hot_size, transform_labels=transform_labels)
        self.dataset = dataset
        self.one_hot_size = dataset.one_hot_size
        self.dataset_size = len(self.dataset)

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

class NonSpikingSpikingAfhq():
    def __init__(self, *args, **kwargs):
        self.spiking_afhq = SpikingAfhq(*args, **kwargs)

    def get_ds_shape(self):
        return ((self.spiking_afhq.count_axons, self.spiking_afhq.time_in_ms), 1, (self.spiking_afhq.one_hot_size,), 1)

    def get_train_loader(self, batch_size, num_workers=1):
        return DataLoader(self.spiking_afhq.dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    def get_valid_loader(self, batch_size, num_workers=1):
        return DataLoader(self.spiking_afhq.dataset_valid, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    def train_size(self):
        return len(self.spiking_afhq.dataset_train)

    def valid_size(self):
        return len(self.spiking_afhq.dataset_valid)