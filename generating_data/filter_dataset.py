import numpy as np
import torch
import logging

from utils.utils import setup_logger, MAXIMAL_RANDOM_SEED

logger = logging.getLogger(__name__)

class FilterLabelsDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, orig_dataset, predicate, one_hot=False, one_hot_size=-1, transform_labels=None, max_count_samples=None, extra_label_information=False, shuffle_seed=None, return_original_index=False):
        self.orig_dataset = orig_dataset
        self.predicate = predicate
        self.one_hot = one_hot
        self.one_hot_size = one_hot_size
        self.transform_labels = transform_labels
        self.max_label = -1
        self.extra_label_information = extra_label_information
        self.shuffle_seed = shuffle_seed
        self.return_original_index = return_original_index

        self.indices = []

        indices_to_iterate = range(len(orig_dataset))

        if self.shuffle_seed is None:
            self.shuffle_seed = np.random.randint(0, MAXIMAL_RANDOM_SEED + 1)
        logger.info(f'Shuffle seed: {self.shuffle_seed}')
        np.random.seed(self.shuffle_seed)

        indices_to_iterate = np.random.permutation(indices_to_iterate)

        appended_indices = 0

        if hasattr(self.orig_dataset, 'targets') and len(self.orig_dataset.targets) > 0:
            for i in indices_to_iterate:
                target = self.orig_dataset.targets[i]
                if predicate(target):
                    self.indices.append(i)
                    if self.transform_labels is not None:
                        target = self.transform_labels(target)
                    if target > self.max_label:
                        self.max_label = target
                    appended_indices += 1
                    if max_count_samples is not None and appended_indices >= max_count_samples:
                        break

        else:
            for i in indices_to_iterate:
                _, target = self.orig_dataset[i]
                if predicate(target):
                    self.indices.append(i)
                    if self.transform_labels is not None:
                        target = self.transform_labels(target)
                    if target > self.max_label:
                        self.max_label = target
                    appended_indices += 1
                    if max_count_samples is not None and appended_indices >= max_count_samples:
                        break

        if self.one_hot:
            if self.one_hot_size == -1:
                self.one_hot_size = self.max_label + 1                                        
            else:
                if self.one_hot_size < self.max_label + 1:
                    raise ValueError(f'One-hot encoding size {self.one_hot_size} is too small for maximum label {self.max_label}')
        
            logger.info(f'One-hot encoding size set to {self.one_hot_size} with maximum label {self.max_label}')

    def __getitem__(self, index):
        original_index = self.indices[index]
        item, target = self.orig_dataset[original_index]
        target_orig = target
        if self.transform_labels is not None:
            target = self.transform_labels(target)
        if self.one_hot:
            target = torch.nn.functional.one_hot(torch.tensor(target), self.one_hot_size)
        if self.extra_label_information:
            target = (target, target_orig)

        if self.return_original_index:
            return item, target, original_index
        else:
            return item, target

    def __len__(self):
        return len(self.indices)

class FilterItemsDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, orig_dataset, indices):
        self.orig_dataset = orig_dataset
        self.indices = indices
        
    def __getitem__(self, index):
        return self.orig_dataset[self.indices[index]]

    def __len__(self):
        return len(self.indices)