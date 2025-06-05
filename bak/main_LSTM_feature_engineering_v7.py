import pandas as pd
import os
from datetime import datetime
import torch
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from torch import optim
import torch.nn as nn
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import create_train_test_group

# update mroRnnDataset with better performance
from utils import mroRnnDataset as mroRnnDataset
from utils import collate_fn
import torch.nn.functional as F
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    recall_score,
    precision_score,
    roc_auc_score,
)
import optuna

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
# ---------------------------------------------------------
# data preprocessing control parameter
sample_frac = 0.01
test_size = 0.1
# max_seq_length = 50
# batch_size = 4096
# num_workers = 16

# learning_rate = 0.001
# num_epochs = 1000


# best_val_loss = float("inf")
# best_val_auc = 0.0
# best_val_f1 = 0.0

# best_epoch = float("inf")
# counter = 0
# patience = 10

# time_window_leakage = 5

# ---------------------------------------------------------
# Parameters for RNN Model
rnn_type = "GRU"
rnn_output_size = 16
bidirectional = True
# pooling_method value =  None, 'max', 'avg'
# 'attention', 'multihead_attention'
pooling_method = "attention"
num_heads = 4
use_last_hidden = True
# ---------------------------------------------------------

# log_dir = "./Out"
# os.makedirs(log_dir, exist_ok=True)
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# log_file_path = os.path.join(log_dir, f"training_log_{timestamp}_test.csv")

# pt_model_dir = os.path.join(log_dir, "models")
# os.makedirs(pt_model_dir, exist_ok=True)


# model_save_path = os.path.join(pt_model_dir, f"best_model_{timestamp}.pth")
# pt_model_save_path = os.path.join(pt_model_dir, f"best_model_{timestamp}.pt")


file_name = "./Data/mro_daily_clean.csv"
# ---------------------------------------------------------


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

# ---------------------------------------------------------

# standardize data
scaler = StandardScaler()
data[column_after_std] = scaler.fit_transform(data[column_need_std])


encoder = OneHotEncoder(sparse_output=False)
encoded_categorical = encoder.fit_transform(data[column_need_encode])

category_counts = [
    len(encoder.categories_[i]) for i, _ in enumerate(column_need_encode)
]

onehot_feature_names = []
for col_idx, col in enumerate(column_need_encode):
    num_categories = category_counts[col_idx]
    onehot_feature_names.extend([f"{col}_onehot_{i}" for i in range(num_categories)])

encoded_df = pd.DataFrame(
    encoded_categorical, index=data.index, columns=onehot_feature_names
)
data = pd.concat([data, encoded_df], axis=1)


rnn_features = column_after_std + onehot_feature_names
col_rnn_origin = ["id"] + rnn_features + ["mro"]
data_rnn_origin = data[col_rnn_origin].copy()
data_rnn_origin = create_train_test_group(
    data_rnn_origin, sample_frac=sample_frac, test_size=test_size
)
rnn_target = ["mro"]
# ---------------------------------------------------------


def label_leakage_all_events(
    df: pd.DataFrame,
    id_col="id",
    target_col="mro",
    window_size=5,
):
    """
    For each time series of IDs, find all positions where MRO=1,
    mark the MRO of each time step before each MRO=1 as 1.
    """
    df = df.copy()

    def process_group(group: pd.DataFrame):
        # find all the index when MRO=1
        one_indices = group[group[target_col] == 1].index.tolist()
        if not one_indices:  # if no 1 in the seq, return origin seq
            return group

        for first_one_idx in one_indices:
            start_idx = max(group.index.min(), first_one_idx - window_size)
            group.loc[start_idx:first_one_idx, target_col] = 1
        return group

    # Step 1: Remove id_col from dataframe before groupby
    original_id = df[id_col]
    data_to_process = df.drop(columns=[id_col])

    # Step 2: Group and apply function on reduced dataframe
    processed_data = (
        data_to_process.groupby(original_id, group_keys=False)
        .apply(process_group)
        .reset_index(drop=True)
    )

    # Step 3: Re-add the id column back
    processed_data[id_col] = original_id.reset_index(drop=True)

    return processed_data


class RnnModel(nn.Module):

    def __init__(
        self,
        rnn_type: str,
        input_size,
        rnn_output_size,
        output_size,
        num_layers: int = 3,
        bidirectional: bool = False,
        pooling_method: str = "attention",  # value =  None, 'max', 'avg', 'attention', 'multihead_attention'
        num_heads=4,
        use_last_hidden: bool = True,
    ):
        super(RnnModel, self).__init__()
        # bidirectional = False
        num_directions = 2 if bidirectional else 1
        self.embed_dim = rnn_output_size * num_directions
        self.pooling_method = pooling_method
        self.use_last_hidden = use_last_hidden

        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=rnn_output_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=0.5,
            )
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=rnn_output_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=0.5,
            )
        else:
            raise ValueError(f"Unsupported rnn_type: {rnn_type}")

        if pooling_method == "multihead_attention":
            self.attn = nn.MultiheadAttention(
                embed_dim=self.embed_dim, num_heads=num_heads, batch_first=True
            )
        elif pooling_method == "attention":
            self.attn = nn.MultiheadAttention(
                embed_dim=self.embed_dim, num_heads=1, batch_first=True
            )
        elif pooling_method in ["max", "avg"]:
            self.pool = getattr(nn, f"Adaptive{pooling_method.capitalize()}Pool1d")(1)
        elif pooling_method is None:
            pass
        else:
            raise ValueError(f"Unsupported pooling_method: {pooling_method}")
        self.fc = nn.Sequential(
            nn.Linear(self.embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, output_size),
        )

    def forward(self, x, length: torch.Tensor):
        # x (batch_size, seq_len, input_size)
        # use pack_padded_sequence and pad_packed_sequence to deal with different length of x input
        packed_x = pack_padded_sequence(
            x, length, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.rnn(packed_x)
        rnn_out, _ = pad_packed_sequence(packed_out, batch_first=True)
        # pooled = self.pool(rnn_out.transpose(1, 2)).squeeze(2)
        # attended = self.attention(rnn_out)
        if self.pooling_method is None:
            if self.use_last_hidden:
                idx = (length - 1).to(x.device)
                batch_indices = torch.arange(x.size(0), device=x.device)
                pooled = rnn_out[batch_indices, idx]  # (B, H)
            else:
                # use the average to of all outputs
                pooled = rnn_out.mean(dim=1)  # (B, H)
        elif self.pooling_method in ["max", "avg"]:
            pooled = self.pool(rnn_out.transpose(1, 2)).squeeze(2)
        elif self.pooling_method in ["attention", "multihead_attention"]:
            # if self.pooling_method == "multihead_attention":
            #     attn_out, _ = self.attn(rnn_out, rnn_out, rnn_out)
            #     pooled = attn_out.mean(dim=1)
            # else:
            #     pooled = self.attn(rnn_out)
            attn_output, _ = self.attn(rnn_out, rnn_out, rnn_out)
            pooled = attn_output.mean(dim=1)
        else:
            raise ValueError("Invalid pooling_method")

        model_out = self.fc(pooled)
        return model_out


# ---------------------------------------------------------
log_dir = "./Out"
os.makedirs(log_dir, exist_ok=True)
global_log_file = os.path.join(log_dir, "best_experiments_log.csv")

global_log_columns = [
    "Timestamp",
    "Best Epoch",
    "Learning Rate",
    "RNN Type",
    "RNN Output Size",
    "RNN Layers",
    "Bidirectional",
    "Pooling Method",
    "Num Heads",
    "Time Window Leakage",
    "Train Loss",
    "Train F1",
    "Train Accuracy",
    "Train Recall",
    "Train Precision",
    "Train AUC",
    "Val Loss",
    "Val F1",
    "Val Accuracy",
    "Val Recall",
    "Val Precision",
    "Val AUC",
    "Log File Path",
    "Model Save Path",
]
if not os.path.exists(global_log_file):
    df_empty = pd.DataFrame(columns=global_log_columns)
    df_empty.to_csv(global_log_file, index=False)
# ---------------------------------------------------------


def train_model(trial: optuna.trial.Trial, data_rnn_origin, rnn_features, rnn_target):

    # ---------------------------------------------------------
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    # batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    rnn_output_size = trial.suggest_categorical("rnn_output_size", [16, 32])
    pooling_method = trial.suggest_categorical(
        "pooling_method", ["max", "avg", "attention", "multihead_attention", None]
    )
    time_window_leakage = trial.suggest_int("time_window_leakage", 5, 10)
    # num_heads = trial.suggest_int("num_heads", 2, 8)
    num_heads = trial.suggest_categorical("num_heads", [4, 8])
    rnn_type = trial.suggest_categorical("rnn_type", ["GRU", "LSTM"])
    bidirectional = trial.suggest_categorical("bidirectional", [True, False])
    num_layers = trial.suggest_int("num_layers", 1, 4)

    use_last_hidden = True
    max_seq_length = 50
    batch_size = 4096
    num_workers = 16
    num_epochs = 1000

    best_val_loss = float("inf")
    best_val_auc = 0.0
    best_val_f1 = 0.0

    best_epoch = float("inf")
    counter = 0
    patience = 10
    # ---------------------------------------------------------
    log_dir = "./Out"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_log_dir = os.path.join(log_dir, f"training_log_{timestamp}_test")
    os.makedirs(current_log_dir, exist_ok=True)
    log_file_path = os.path.join(current_log_dir, f"training_log_{timestamp}_test.csv")

    pt_model_dir = os.path.join(current_log_dir, "models")
    os.makedirs(pt_model_dir, exist_ok=True)

    model_save_path = os.path.join(pt_model_dir, f"best_model_{timestamp}.pth")
    # ---------------------------------------------------------
    data_rnn_origin = label_leakage_all_events(
        data_rnn_origin,
        id_col="id",
        target_col="mro",
        window_size=time_window_leakage,
    )
    # ---------------------------------------------------------

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
    # ---------------------------------------------------------

    input_feature_size = len(rnn_features)
    output_size = len(rnn_target)

    model = RnnModel(
        rnn_type=rnn_type,
        input_size=input_feature_size,
        rnn_output_size=rnn_output_size,
        output_size=output_size,
        num_layers=num_layers,
        bidirectional=bidirectional,
        # value =  None, 'max', 'avg', 'attention', 'multihead_attention'
        pooling_method=pooling_method,
        num_heads=num_heads,
        use_last_hidden=use_last_hidden,
    ).to(device)

    print(model)

    pos_weight_value = (data_rnn_origin["mro"] == 0).sum() / (
        data_rnn_origin["mro"] == 1
    ).sum()
    print(f"pos_weight: {pos_weight_value}")

    # criterion = nn.BCELoss()
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value])).to(
        device
    )

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    fieldnames = [
        "Epoch",
        "Train Loss",
        "Train F1",
        "Train Accuracy",
        "Train Recall",
        "Train Precision",
        "Train AUC",
        "Val Loss",
        "Val F1",
        "Val Accuracy",
        "Val Recall",
        "Val Precision",
        "Val AUC",
    ]

    log_df = pd.DataFrame(columns=fieldnames)
    log_df.to_csv(log_file_path, index=False)
    log_row = {
        "Timestamp": timestamp,
        "Best Epoch": None,
        # ---- Learning Control ----
        "Learning Rate": learning_rate,
        "RNN Type": rnn_type,
        "RNN Output Size": rnn_output_size,
        "RNN Layers": num_layers,
        "Bidirectional": bidirectional,
        "Pooling Method": pooling_method,
        "Num Heads": num_heads,
        "Time Window Leakage": time_window_leakage,
        # ---- Train ----
        "Train Loss": None,
        "Train F1": None,
        "Train Accuracy": None,
        "Train Recall": None,
        "Train Precision": None,
        "Train AUC": None,
        # ---- Test -----
        "Val Loss": None,
        "Val F1": None,
        "Val Accuracy": None,
        "Val Recall": None,
        "Val Precision": None,
        "Val AUC": None,
        # ---- Log File Path ----
        "Log File Path": log_file_path,
        "Model Save Path": model_save_path,
    }
    for epoch in range(num_epochs):
        running_loss = 0.0
        print(f"Epoch {epoch}")
        model.train()
        all_train_mro_preds = []
        all_train_mro_targets = []
        all_train_mro_scores = []
        for train_inputs, train_targets, train_lengths in train_dataloader:

            optimizer.zero_grad()
            train_inputs = train_inputs.to(device)
            train_targets = train_targets.to(device)
            # train_lengths = train_lengths.to(device)

            model_out = model(train_inputs, train_lengths)

            loss = criterion(model_out, train_targets[:, -1, :])
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            mro_pred = torch.sigmoid(model_out)
            mro_preds = (mro_pred > 0.5).int().cpu().numpy().flatten()
            mro_targets = train_targets[:, -1, :].cpu().numpy().flatten()

            all_train_mro_preds.extend(mro_preds)
            all_train_mro_targets.extend(mro_targets)
            all_train_mro_scores.extend(
                torch.sigmoid(model_out).detach().cpu().numpy().flatten()
            )

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
        train_auc = roc_auc_score(all_train_mro_targets, all_train_mro_scores)
        print(f"Training AUC: {train_auc}")

        # Validation loop
        model.eval()
        val_loss = 0.0
        all_val_mro_preds = []
        all_val_mro_targets = []
        all_val_mro_scores = []
        with torch.no_grad():
            for val_inputs, val_targets, test_lengths in test_dataloader:
                val_inputs = val_inputs.to(device)
                val_targets = val_targets.to(device)
                model_out = model(val_inputs, test_lengths)
                loss = criterion(model_out, val_targets[:, -1, :])

                val_loss += loss.item()
                mro_pred = torch.sigmoid(model_out)
                mro_preds = (mro_pred > 0.5).int().cpu().numpy().flatten()
                mro_targets = val_targets[:, -1, :].cpu().numpy().flatten()

                all_val_mro_preds.extend(mro_preds)
                all_val_mro_targets.extend(mro_targets)
                all_val_mro_scores.extend(
                    torch.sigmoid(model_out).detach().cpu().numpy().flatten()
                )
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
            val_auc = roc_auc_score(all_val_mro_targets, all_val_mro_scores)
            print(f"Validation AUC: {val_auc}")

        log_entry = {
            "Epoch": epoch,
            "Train Loss": average_loss,
            "Train F1": train_f1,
            "Train Accuracy": train_accuracy,
            "Train Recall": train_recall,
            "Train Precision": train_precision,
            "Train AUC": train_auc,
            "Val Loss": average_val_loss,
            "Val F1": val_f1,
            "Val Accuracy": val_accuracy,
            "Val Recall": val_recall,
            "Val Precision": val_precision,
            "Val AUC": val_auc,
        }
        entry_df = pd.DataFrame([log_entry])
        entry_df.to_csv(log_file_path, mode="a", header=False, index=False)

        if best_val_f1 < val_f1:
            best_val_f1 = val_f1
            counter = 0
            # save the model
            torch.save(model.state_dict(), model_save_path)
            # torchscript_model = torch.jit.script(model)
            # torchscript_model.save(pt_model_save_path)
            best_epoch = epoch
            log_row = {
                "Timestamp": timestamp,
                "Best Epoch": epoch,
                # ---- Learning Control ----
                "Learning Rate": learning_rate,
                "RNN Type": rnn_type,
                "RNN Output Size": rnn_output_size,
                "RNN Layers": num_layers,
                "Bidirectional": bidirectional,
                "Pooling Method": pooling_method,
                "Num Heads": num_heads,
                "Time Window Leakage": time_window_leakage,
                # ---- Train ----
                "Train Loss": average_loss,
                "Train F1": train_f1,
                "Train Accuracy": train_accuracy,
                "Train Recall": train_recall,
                "Train Precision": train_precision,
                "Train AUC": train_auc,
                # ---- Test -----
                "Val Loss": average_val_loss,
                "Val F1": val_f1,
                "Val Accuracy": val_accuracy,
                "Val Recall": val_recall,
                "Val Precision": val_precision,
                "Val AUC": val_auc,
                # ---- Log File Path ----
                "Log File Path": log_file_path,
                "Model Save Path": model_save_path,
            }

        else:
            counter += 1
            if counter >= patience:
                print("Early stopping!")
                print("Best Epoch is:", best_epoch)
                df_log = pd.DataFrame([log_row])
                df_log.to_csv(
                    global_log_file,
                    mode="a",
                    header=not os.path.exists(global_log_file),
                    index=False,
                )
                break

    return best_val_f1


study = optuna.create_study(
    direction="maximize",
    study_name="mro_lstm_mix",
    storage="sqlite:///mro_lstm.db",
)
study.optimize(
    lambda trial: train_model(trial, data_rnn_origin, rnn_features, rnn_target),
    n_trials=1000,
)
