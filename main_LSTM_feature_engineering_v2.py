import pandas as pd
import os
from datetime import datetime
import torch
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch import optim
import torch.nn as nn
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import create_train_test_group
# update mroRnnDataset with better performance
from utils import mroRnnDatasetNew as mroRnnDataset
from utils import collate_fn


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


sample_frac = 1
test_size = 0.1
max_seq_length = 50
batch_size = 4096
num_workers = 16
rnn_output_size = 16
learning_rate = 0.001
num_epochs = 1000


best_val_loss = float("inf")
best_epoch = float("inf")
counter = 0
patience = 10


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


# Example usage (assuming 'data', 'column_after_std', and 'column_after_encode_std' are already defined):
col_rnn_origin = ["id"] + column_after_std + column_after_encode_std + ["mro"]

data_rnn_origin = data[
    col_rnn_origin
].copy()  # Create a copy to avoid modifying the original DataFrame

data_rnn_origin = create_train_test_group(
    data_rnn_origin, sample_frac=sample_frac, test_size=test_size
)


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


train_data_set = mroRnnDataset(
    data_rnn_origin=data_rnn_origin,
    rnn_features=rnn_features,
    rnn_target=rnn_target,
    group="train",
    max_seq_length=max_seq_length,
)

test_data_set = mroRnnDataset(
    data_rnn_origin=data_rnn_origin,
    rnn_features=rnn_features,
    rnn_target=rnn_target,
    group="test",
    max_seq_length=max_seq_length,
)


# Example batch size
train_dataloader = DataLoader(
    train_data_set,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=num_workers,
)

test_dataloader = DataLoader(
    test_data_set,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=num_workers,
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
                num_layers=3,
                batch_first=True,
                bidirectional=True,
                dropout=0.5,
            )
        self.fc = nn.Sequential(
            nn.Linear(rnn_output_size * 2, 16),
            nn.ReLU(),
            nn.Linear(16, output_size),
        )
        # self.fc = nn.Linear(rnn_output_size, output_size)

    def forward(self, x, length: torch.Tensor):
        # x  (batch_size, seq_len, input_size)
        # use pack_padded_sequence and pad_packed_sequence to deal with different length of x input
        packed_x = pack_padded_sequence(
            x, length, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.rnn(packed_x)
        rnn_out, _ = pad_packed_sequence(packed_out, batch_first=True)
        model_out = self.fc(rnn_out)  # (batch_size, seq_len, output_size)
        return model_out


input_feature_size = len(rnn_features)
output_size = len(rnn_target)

model = RnnModel(
    rnn_type="LSTM",
    input_size=input_feature_size,
    rnn_output_size=rnn_output_size,
    output_size=output_size,
).to(device)

print(model)


pos_weight_value = (data_rnn_origin["mro"] == 0).sum() / (
    data_rnn_origin["mro"] == 1
).sum()
print(f"pos_weight: {pos_weight_value}")


# criterion = nn.BCELoss()
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value])).to(device)
# criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# for input, target, length in train_dataloader:
#     target: torch.Tensor
#     input: torch.Tensor
#     print(input.shape)
#     print(target.shape)
#     print(length)
#     break


log_dir = "Out"
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file_path = os.path.join(log_dir, f"training_log_{timestamp}.csv")

pt_model_dir = os.path.join(log_dir, "models")
os.makedirs(pt_model_dir, exist_ok=True)

fieldnames = [
    "Epoch",
    "Train Loss",
    "Train F1",
    "Train Accuracy",
    "Train Recall",
    "Train Precision",
    "Val Loss",
    "Val F1",
    "Val Accuracy",
    "Val Recall",
    "Val Precision",
]

log_df = pd.DataFrame(columns=fieldnames)
log_df.to_csv(log_file_path, index=False)


model_save_path = os.path.join(pt_model_dir, f"best_model_{timestamp}.pth")
pt_model_save_path = os.path.join(pt_model_dir, f"best_model_{timestamp}.pt")


from sklearn.metrics import (
    f1_score,
    accuracy_score,
    recall_score,
    precision_score,
)


for epoch in range(num_epochs):
    running_loss = 0.0
    print(f"Epoch {epoch}")
    model.train()
    all_train_mro_preds = []
    all_train_mro_targets = []
    for train_inputs, train_targets, train_lengths in train_dataloader:

        optimizer.zero_grad()
        train_inputs = train_inputs.to(device)
        train_targets = train_targets.to(device)
        # train_lengths = train_lengths.to(device)

        model_out = model(train_inputs, train_lengths)

        loss = criterion(model_out[:, -1, :], train_targets[:, -1, :])
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        mro_pred = torch.sigmoid(model_out[:, -1, :])
        mro_preds = (mro_pred > 0.5).int().cpu().numpy().flatten()
        mro_targets = train_targets[:, -1, :].cpu().numpy().flatten()

        all_train_mro_preds.extend(mro_preds)
        all_train_mro_targets.extend(mro_targets)

    average_loss = running_loss / len(train_dataloader)
    print(f"Average training loss: {average_loss}")

    train_f1 = f1_score(all_train_mro_targets, all_train_mro_preds)
    print(f"Training F1 Score: {train_f1}")
    train_accuracy = accuracy_score(all_train_mro_targets, all_train_mro_preds)
    print(f"Training Accuracy: {train_accuracy}")
    train_recall = recall_score(all_train_mro_targets, all_train_mro_preds)
    print(f"Training Recall: {train_recall}")
    train_precision = precision_score(all_train_mro_targets, all_train_mro_preds)
    print(f"Training Precision: {train_precision}")

    # Validation loop
    model.eval()
    val_loss = 0.0
    all_val_mro_preds = []
    all_val_mro_targets = []
    with torch.no_grad():
        for val_inputs, val_targets, test_lengths in test_dataloader:
            val_inputs = val_inputs.to(device)
            val_targets = val_targets.to(device)
            model_out = model(val_inputs, test_lengths)
            loss = criterion(model_out[:, -1, :], val_targets[:, -1, :])

            val_loss += loss.item()
            mro_pred = torch.sigmoid(model_out[:, -1, :])
            mro_preds = (mro_pred > 0.5).int().cpu().numpy().flatten()
            mro_targets = val_targets[:, -1, :].cpu().numpy().flatten()

            all_val_mro_preds.extend(mro_preds)
            all_val_mro_targets.extend(mro_targets)

        average_val_loss = val_loss / len(test_dataloader)
        print(f"Validation Loss: {average_val_loss}")
        val_f1 = f1_score(all_val_mro_targets, all_val_mro_preds)
        print(f"Validation F1 Score: {val_f1}")
        val_accuracy = accuracy_score(all_val_mro_targets, all_val_mro_preds)
        print(f"Validation Accuracy: {val_accuracy}")
        val_recall = recall_score(all_val_mro_targets, all_val_mro_preds)
        print(f"Validation Recall: {val_recall}")
        val_precision = precision_score(all_val_mro_targets, all_val_mro_preds)
        print(f"Validation Precision: {val_precision}")

    log_entry = {
        "Epoch": epoch,
        "Train Loss": average_loss,
        "Train F1": train_f1,
        "Train Accuracy": train_accuracy,
        "Train Recall": train_recall,
        "Train Precision": train_precision,
        "Val Loss": average_val_loss,
        "Val F1": val_f1,
        "Val Accuracy": val_accuracy,
        "Val Recall": val_recall,
        "Val Precision": val_precision,
    }
    entry_df = pd.DataFrame([log_entry])
    entry_df.to_csv(log_file_path, mode="a", header=False, index=False)

    if average_val_loss < best_val_loss:
        best_val_loss = average_val_loss
        counter = 0
        # save the model
        torch.save(model.state_dict(), model_save_path)
        torchscript_model = torch.jit.script(model)
        torchscript_model.save(pt_model_save_path)
        best_epoch = epoch

    else:
        counter += 1
        if counter >= patience:
            print("Early stopping!")
            print("Best Epoch is:", best_epoch)
            break
