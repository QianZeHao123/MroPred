# | default_exp lstm_dataset
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import cupy as cp
import bisect


class mroRnnDataset(Dataset):
    def __init__(
        self,
        data_rnn_origin: pd.DataFrame,
        rnn_features: list,
        rnn_target: list,
        group: str = "train",
        max_seq_length: int = 10,
    ):
        self.max_seq_length = max_seq_length
        self.data = self._prepare_data(data_rnn_origin, rnn_features, rnn_target, group)
        self.ids = list(self.data.keys())

        self.cumulative_lengths = []
        total_length = 0
        for id in self.ids:
            features, _ = self.data[id]
            total_length += len(features)
            self.cumulative_lengths.append(total_length)
        self.total_length = total_length

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        data_idx = bisect.bisect_right(self.cumulative_lengths, idx)
        if data_idx == 0:
            pos = idx
        else:
            pos = idx - self.cumulative_lengths[data_idx - 1]
        id = self.ids[data_idx]
        features, targets = self.data[id]

        # slice the seq
        features = features[: pos + 1]
        targets = targets[: pos + 1]

        if len(features) > self.max_seq_length:
            features = features[-self.max_seq_length :]
            targets = targets[-self.max_seq_length :]

        return features, targets

    def _prepare_data(
        self,
        data_rnn_origin: pd.DataFrame,
        rnn_features: list,
        rnn_target: list,
        group: str = "train",
    ):
        data: pd.DataFrame = data_rnn_origin[data_rnn_origin["group"] == group]

        features_data = data[rnn_features].values
        target_data = data[rnn_target].values
        id_data = data["id"].values

        sort_indices = np.argsort(id_data)
        sorted_features = features_data[sort_indices]
        sorted_targets = target_data[sort_indices]
        sorted_ids = id_data[sort_indices]

        _, unique_indices = np.unique(sorted_ids, return_index=True)
        split_indices = np.append(unique_indices[1:], len(sorted_ids))

        split_features = np.split(sorted_features, split_indices)
        split_targets = np.split(sorted_targets, split_indices)
        unique_ids = np.unique(sorted_ids)

        processed_data = {}
        for id_val, feat, targ in zip(unique_ids, split_features, split_targets):
            features_tensor = torch.tensor(feat, dtype=torch.float32)
            target_tensor = torch.tensor(targ, dtype=torch.float32)
            processed_data[id_val] = (features_tensor, target_tensor)

        return processed_data
