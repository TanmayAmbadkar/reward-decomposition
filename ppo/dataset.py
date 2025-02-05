
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class RecurrentPPODataset(Dataset):

    def __init__(self, *dataset):
        """
        Args:
        dataset: iterable containing lists of tensors, where each tensor in a
            list represents an element in a batch.
        
        Example usage for batch size NL:
        >>> obs = [ obs_1, obs_2, ..., obs_N ]
        >>> actions = [ actions_1, actions_2, ..., actions_N ]
        >>> dataset = PPODataset(obs, actions)
        >>> dataset[5]
        (obs_5, actions_5)
        """
        super().__init__()

        self._len = len(dataset[0])
        self._dataset = []
        for data in dataset:
            entry = np.empty(self._len, dtype=object)
            for i in range(self._len):
                entry[i] = data[i]
            self._dataset.append(entry)

    def __getitem__(self, idx):
        items = []
        for data in self._dataset:
            entry = data[idx]
            # only pad if selecting multiple tensors
            if isinstance(entry, np.ndarray):
                entry = pad_sequence(entry)
            # else:
            #     entry = entry.reshape(-1, 1)
            items.append(entry)
        return tuple(items)

    def __len__(self):
        return self._len