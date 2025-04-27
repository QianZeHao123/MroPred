import pandas as pd
from torch import optim
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import pandas as pd
import torch
from typing import List
import cupy as cp
import pandas as pd
from torch.utils.data import Dataset
from pandas.api.typing import DataFrameGroupBy
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


try:
    import cupy as cp
    import cupy.cuda.runtime as runtime

    # get the num of GPUs
    n_gpus = cp.cuda.runtime.getDeviceCount()
    print(f"Detected {n_gpus} GPU(s):")
    # print the info of GPUs
    for i in range(n_gpus):
        device = cp.cuda.Device(i)
        attrs = runtime.getDeviceProperties(i)
        print(
            f"GPU {i}: {attrs['name'].decode()} with {attrs['totalGlobalMem'] // (1024**3)} GB memory"
        )
    cupy_gpu_id = 7
    cp.cuda.Device(cupy_gpu_id).use()
    print(f"Using GPU {cupy_gpu_id}")
    using_cupy = True
except ImportError:
    print("CuPy is not available, falling back to NumPy.")
    using_cupy = False
    cp = np  # rename cupy to np for compatibility


file_name = "./Data/mro_daily_clean.csv"
data = pd.read_csv(file_name, index_col=0, engine="pyarrow")


column_need_std = [
    "hard_braking",
    "hard_acceleration",
    "speeding_sum",
    "day_mileage",
    "engn_size",
    "est_hh_incm_prmr_cd",
    "purchaser_age_at_tm_of_purch",
    "tavg",
    "random_avg_traffic",
]

column_after_std = [
    "hard_braking_std",
    "hard_acceleration_std",
    "speeding_sum_std",
    "day_mileage_std",
    "engn_size_std",
    "est_hh_incm_prmr_cd_std",
    "purchaser_age_at_tm_of_purch_std",
    "tavg_std",
    "random_avg_traffic_std",
]

column_need_encode = [
    "gmqualty_model",
    "umf_xref_finc_gbl_trim",
    "input_indiv_gndr_prmr_cd",
]

column_after_encode = [
    "gmqualty_model_encode",
    "umf_xref_finc_gbl_trim_encode",
    "input_indiv_gndr_prmr_cd_encode",
]

column_after_encode_std = [
    "gmqualty_model_encode_std",
    "umf_xref_finc_gbl_trim_encode_std",
    "input_indiv_gndr_prmr_cd_encode_std",
]

# standardize data
scaler = StandardScaler()
data[column_after_std] = scaler.fit_transform(data[column_need_std])

# encode data
label_encoders = {}
for i, col in enumerate(column_need_encode):
    le = LabelEncoder()
    data[column_after_encode[i]] = le.fit_transform(data[col])
    label_encoders[col] = le  # Store the encoder for later use if needed

data[column_after_encode_std] = scaler.fit_transform(data[column_after_encode])


def create_train_test_group(
    data: pd.DataFrame,
    id_column: str = "id",
    test_size: float = 0.8,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Adds a 'group' column to the DataFrame, assigning 'train' or 'test' based on a train/test split of unique IDs.
    This function is optimized for speed using numpy and vectorized operations.

    Args:
        data (pd.DataFrame): The input DataFrame.
        id_column (str): The name of the column containing unique IDs. Defaults to "id".
        test_size (float): The proportion of unique IDs to include in the test set. Defaults to 0.1.
        random_state (int): The random state for the train/test split. Defaults to 42.

    Returns:
        pd.DataFrame: The DataFrame with the added 'group' column.
    """
    unique_ids = data[id_column].unique()
    train_ids, test_ids = train_test_split(
        unique_ids, test_size=test_size, random_state=random_state
    )

    # Convert test_ids to a set for faster membership checking
    test_ids_set = set(test_ids)

    # Use numpy's vectorized apply for faster group assignment
    data["group"] = np.where(data[id_column].isin(test_ids_set), "test", "train")

    return data


# Example usage (assuming 'data', 'column_after_std', and 'column_after_encode_std' are already defined):
col_rnn_origin = ["id"] + column_after_std + column_after_encode_std + ["mro"]
data_rnn_origin = data[
    col_rnn_origin
].copy()  # Create a copy to avoid modifying the original DataFrame

data_rnn_origin = create_train_test_group(data_rnn_origin)


rnn_features = [
    "hard_braking_std",
    "hard_acceleration_std",
    "speeding_sum_std",
    "day_mileage_std",
    "engn_size_std",
    "est_hh_incm_prmr_cd_std",
    "purchaser_age_at_tm_of_purch_std",
    "tavg_std",
    "random_avg_traffic_std",
    "gmqualty_model_encode_std",
    "umf_xref_finc_gbl_trim_encode_std",
    "input_indiv_gndr_prmr_cd_encode_std",
]

rnn_target = ["mro"]


def prepare_data(
    data_rnn_origin: pd.DataFrame,
    rnn_features: list,
    rnn_target: list,
    group: str = "train",
):
    """Converts DataFrame to Tensor and groups by ID (accelerated with CuPy, handles string IDs)."""

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
        # time_window = 10
        # current_idx = 0
        # selected_id = None
        # current_group_len = len(features_tensor)
    return processed_data


train_data = prepare_data(data_rnn_origin, rnn_features, rnn_target, "train")
# test_data = prepare_data(data_rnn_origin, rnn_features, rnn_target, "test")


class mroRnnDataset(Dataset):

    def __init__(
        self,
        data: dict,
        max_seq_length: int = 10,
    ):
        self.data = data
        self.max_seq_length = max_seq_length
        self.ids = list(data.keys())

    def __len__(self):
        count = 0
        for id in self.ids:
            features, targets = self.data[id]
            count += len(features)
        return count

    def __getitem__(self, idx):
        current_idx = 0
        pos = None

        for id in self.ids:
            features, targets = self.data[id]
            group_len = len(features)
            if idx < current_idx + group_len:
                selected_id = id
                pos = idx - current_idx
                break
            current_idx += group_len

        features = features[: pos + 1, :]
        targets = targets[: pos + 1]

        if len(features) > self.max_seq_length:
            features = features[-self.max_seq_length :]
            targets = targets[-self.max_seq_length :]

        return features, targets


train_data_set = mroRnnDataset(train_data, max_seq_length=10)


def collate_fn(batch):
    """
    use collate_fn to pad sequences to the same length within a batch
    """
    # batch is a list of (sequence, target) pairs
    sequences, targets = zip(*batch)

    # use pad_sequence to pad sequences to the same length
    padded_sequences = pad_sequence(
        sequences, batch_first=True, padding_value=0.0, padding_side="left"
    )
    padded_targets = pad_sequence(
        targets, batch_first=True, padding_value=0.0, padding_side="left"
    )
    lengths = torch.tensor(
        [len(seq) for seq in sequences]
    )  # get the original length of each sequence
    return padded_sequences, padded_targets, lengths


batch_size = 2048  # Example batch size

train_dataloader = DataLoader(
    train_data_set,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=16,
)


class RnnModel(nn.Module):

    def __init__(
        self,
        rnn_type: str,
        input_size,
        rnn_output_size,
        output_size,
    ):
        super(RnnModel, self).__init__()
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=rnn_output_size,
                num_layers=1,
                batch_first=True,
            )
        self.fc = nn.Sequential(nn.Linear(rnn_output_size, output_size), nn.Sigmoid())

    def forward(self, x, length: int):
        # x  (batch_size, seq_len, input_size)
        # use pack_padded_sequence and pad_packed_sequence to deal with different length of x input
        packed_x = pack_padded_sequence(
            x, length, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.rnn(packed_x)
        rnn_out, _ = pad_packed_sequence(packed_out, batch_first=True)
        model_out = self.fc(rnn_out)  # (batch_size, seq_len, output_size)
        return model_out


# Check if CUDA (NVIDIA GPU) is available
if torch.cuda.is_available():
    # Get the number of available CUDA devices
    num_gpus = torch.cuda.device_count()
    print(f"Number of CUDA devices available: {num_gpus}")

    # Select a specific GPU (e.g., GPU 0)
    device = torch.device("cuda:8")  # Use "cuda:1" for GPU 1, etc.
    print(f"Using device: {torch.cuda.get_device_name(device)}")

else:
    device = "cpu"
    print("CUDA is not available, using CPU")

print(f"Using {device} device")


input_feature_size = len(rnn_features)
rnn_output_size = 16
output_size = len(rnn_target)

model = RnnModel(
    rnn_type="LSTM",
    input_size=input_feature_size,
    rnn_output_size=rnn_output_size,
    output_size=output_size,
).to(device)

print(model)


criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


num_epochs = 1000
for epoch in range(num_epochs):
    running_loss = 0.0
    print(f"Epoch {epoch}")
    model.train()
    for train_inputs, train_targets, train_lengths in train_dataloader:

        print("Add new batch of data!")

        optimizer.zero_grad()
        train_inputs = train_inputs.to(device)
        train_targets = train_targets.to(device)
        # train_lengths = train_lengths.to(device)

        model_out = model(train_inputs, train_lengths)

        loss = criterion(model_out[:, -1, :], train_targets[:, -1, :])
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    average_loss = running_loss / len(train_dataloader)
    print(f"Average training loss: {average_loss}")
