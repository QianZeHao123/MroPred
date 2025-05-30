import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class mroRnnDataset(Dataset):
    def __init__(
        self,
        data_rnn_origin: pd.DataFrame,
        rnn_features: list,
        rnn_target: list,
        group: str = "train",
        seq_length: int = 10,
    ):
        self.seq_length = seq_length
        self.data = self._prepare_data(data_rnn_origin, rnn_features, rnn_target, group)

        # Build a list of (id, start_idx, end_idx) for all valid sequences
        self.sequence_indices = []
        for id, (features, _) in self.data.items():
            length = len(features)
            if length >= seq_length:
                # For each possible starting index to get a sequence of length seq_length
                for start in range(length - seq_length + 1):
                    end = start + seq_length
                    self.sequence_indices.append((id, start, end))

    def __len__(self):
        return len(self.sequence_indices)

    def __getitem__(self, idx):
        id, start, end = self.sequence_indices[idx]
        features, targets = self.data[id]

        feature_seq = features[start:end]
        target_seq = targets[start:end]

        return feature_seq, target_seq

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
