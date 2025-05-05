# | default_exp lstm_dataset
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import cupy as cp

using_cupy = True


class mroRnnDataset(Dataset):
    def __init__(
        self,
        data_rnn_origin: pd.DataFrame,
        rnn_features: list,
        rnn_target: list,
        group: str = "train",
        max_seq_length: int = 10,
    ):
        """
        Initializes the mroRnnDataset.

        Args:
            data_rnn_origin (pd.DataFrame): The original DataFrame containing the data.
            rnn_features (list): List of feature names to be used for the RNN.
            rnn_target (list): List of target names to be predicted by the RNN.
            group (str, optional): The group to filter the data by (e.g., 'train', 'val'). Defaults to "train".
            max_seq_length (int, optional): The maximum sequence length for the RNN. Defaults to 10.
        """
        self.max_seq_length = max_seq_length
        self.data = self._prepare_data(data_rnn_origin, rnn_features, rnn_target, group)
        self.ids = list(self.data.keys())

    def __len__(self):
        """
        Returns the total number of data points in the dataset.
        """
        count = 0
        for id in self.ids:
            features: torch.Tensor
            targets: torch.Tensor
            features, targets = self.data[id]
            count += len(features)
        return count

    def __getitem__(self, idx):
        """
        Retrieves a single item from the dataset at the given index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing the features and targets tensors.
        """
        current_idx = 0
        pos = None

        for id in self.ids:
            features, targets = self.data[id]
            group_len = len(features)
            if idx < current_idx + group_len:
                pos = idx - current_idx
                break
            current_idx += group_len

        features = features[: pos + 1, :]
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
        """
        Converts DataFrame to Tensor and groups by ID (accelerated with CuPy, handles string IDs).

        Args:
            data_rnn_origin (pd.DataFrame): The original DataFrame containing the data.
            rnn_features (list): List of feature names to be used for the RNN.
            rnn_target (list): List of target names to be predicted by the RNN.
            group (str, optional): The group to filter the data by (e.g., 'train', 'val'). Defaults to "train".

        Returns:
            dict: A dictionary where keys are IDs and values are tuples of (features_tensor, target_tensor).
        """

        # 1. Filter data for the specified group (vectorized)
        data: pd.DataFrame = data_rnn_origin[data_rnn_origin["group"] == group]

        # 2. Get all unique IDs (vectorized)
        ids = data["id"].unique()

        # 3. Create a dictionary to store the results
        processed_data = {}

        # 4. Convert to NumPy/CuPy array (one-time conversion)
        features_data = data[rnn_features].values
        target_data = data[rnn_target].values
        id_data = data["id"].values

        # Use a dictionary to map string IDs to integers
        unique_ids = np.unique(id_data)
        id_to_index = {id_val: idx for idx, id_val in enumerate(unique_ids)}
        indexed_id_data = np.array(
            [id_to_index[id_val] for id_val in id_data], dtype=np.int64
        )

        if using_cupy:
            features_data = cp.asarray(features_data)
            target_data = cp.asarray(target_data)
            indexed_id_data = cp.asarray(indexed_id_data)

        # 5. Loop through IDs (still needed, but operations inside the loop are optimized)
        for id_val in ids:
            # Find the indices corresponding to this ID (vectorized)
            # Use the mapped integer ID
            indexed_id_val = id_to_index[id_val]
            indices = cp.where(indexed_id_data == indexed_id_val)[0]

            # Extract data using indices (avoid Pandas operations)
            features = features_data[indices]
            target = target_data[indices]

            # Convert to Tensor (on CPU)
            features_tensor = torch.tensor(cp.asnumpy(features), dtype=torch.float32)
            target_tensor = torch.tensor(cp.asnumpy(target), dtype=torch.float32)

            # Store in the dictionary
            processed_data[id_val] = (features_tensor, target_tensor)

        return processed_data
