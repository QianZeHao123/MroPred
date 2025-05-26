import pandas as pd
import os
import time

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    recall_score,
    precision_score,
    roc_auc_score,
)

import torch
from torch.utils.data import DataLoader
from torch import optim


from utils import create_train_test_group
from utils import collate_fn

from model import FocalLoss
from model import RnnModel
from model import mroRnnDataset

import ray.train.torch
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5,6,7,8"


# ---------------------------------------------------------
# data preprocessing control parameter
sample_frac = 1
test_size = 0.1
valid_size = 0.1
max_seq_length = 8
batch_size = 4096
num_workers = 16

learning_rate = 0.0005
num_epochs = 1000

# ---------------------------------------------------------
# Parameters for RNN Model
rnn_type = "LSTM"
rnn_output_size = 128
bidirectional = True
# pooling_method value =  None, 'max', 'avg', 'attention', 'multihead_attention'
num_layers = 2
pooling_method = None
num_heads = None
use_last_hidden = True
agg_fun = ["mean", "sum", "max", "min", "std", "skew"]

# ---------------------------------------------------------
file_name = "./Data/mro_daily_clean.csv"
data = pd.read_csv(file_name, index_col=0, engine="pyarrow")

# ---------------------------------------------------------
# log file
result_csv_file = "./Out/ray_train_log.csv"
# if the file exist, delete it
os.makedirs(os.path.dirname(result_csv_file), exist_ok=True)

result_csv_file = os.path.abspath(result_csv_file)

if os.path.exists(result_csv_file):
    os.remove(result_csv_file)
    print(f"Deleted existing CSV file: {result_csv_file}")
# ---------------------------------------------------------
# model output
output_dir = "./Out"
os.makedirs(output_dir, exist_ok=True)
current_time = time.strftime("%Y%m%d_%H%M%S")
run_dir = os.path.join(output_dir, current_time)
os.makedirs(run_dir, exist_ok=True)
run_dir = os.path.abspath(run_dir)

# ---------------------------------------------------------
# early stop control parameter
best_val_loss = float("inf")
early_stop_patience = 20
early_stop_counter = 0
best_epoch = 0
# ---------------------------------------------------------
continuous_variable = [
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

category_variable = [
    "gmqualty_model",
    "umf_xref_finc_gbl_trim",
    "input_indiv_gndr_prmr_cd",
]

driver_navigation = [
    "id",
    "yr_nbr",
    "mth_nbr",
    "week_nbr",
]

mro = ["mro"]


data = data[driver_navigation + continuous_variable + category_variable + mro]


data = data.groupby(["id", "yr_nbr", "week_nbr"]).agg(
    {
        "mth_nbr": "first",
        "mro": "max",
        "hard_braking": agg_fun,
        "hard_acceleration": agg_fun,
        "speeding_sum": agg_fun,
        "day_mileage": agg_fun,
        "est_hh_incm_prmr_cd": "first",
        "purchaser_age_at_tm_of_purch": "first",
        "input_indiv_gndr_prmr_cd": "first",
        "gmqualty_model": "first",
        "umf_xref_finc_gbl_trim": "first",
        "engn_size": "first",
        "tavg": agg_fun,
        "random_avg_traffic": agg_fun,
    }
)

data.reset_index(inplace=True)


def flatten_columns(df: pd.DataFrame):
    def clean_col(col):
        if isinstance(col, tuple):
            col_name, agg_func = col
            agg_func = agg_func.strip()
            if col_name in mro and agg_func == "max":
                return "mro"
            if agg_func in ("first", ""):
                return col_name
            return f"{col_name}_{agg_func}"
        else:
            return col

    df.columns = [clean_col(col) for col in df.columns]
    return df


data = flatten_columns(data)
data.fillna(0, inplace=True)
data = data.drop(["yr_nbr", "week_nbr", "mth_nbr"], axis=1)


col_need_std = [
    item
    for item in data.columns.values.tolist()
    if item not in (mro + ["id"] + category_variable)
]

col_need_encode = category_variable


scaler = StandardScaler()
data[col_need_std] = scaler.fit_transform(data[col_need_std])


encoder = OneHotEncoder(sparse_output=False)
encoded_categorical = encoder.fit_transform(data[col_need_encode])

category_counts = [len(encoder.categories_[i]) for i, _ in enumerate(col_need_encode)]

onehot_feature_names = []
for col_idx, col in enumerate(col_need_encode):
    num_categories = category_counts[col_idx]
    onehot_feature_names.extend([f"{col}_onehot_{i}" for i in range(num_categories)])

encoded_df = pd.DataFrame(
    encoded_categorical, index=data.index, columns=onehot_feature_names
)
data = pd.concat([data, encoded_df], axis=1)


rnn_features = col_need_std + onehot_feature_names
# col_rnn_origin = ["id"] + rnn_features + ["mro"]
col_rnn_origin = ["id"] + rnn_features + mro
data_rnn_origin = data[col_rnn_origin].copy()
data_rnn_origin = create_train_test_group(
    data_rnn_origin,
    sample_frac=sample_frac,
    test_size=test_size,
    valid_size=valid_size,
    random_state=42,
)


# rnn_target = ["mro"]
rnn_target = mro


# ---------------------------------------------------------
train_data_set = mroRnnDataset(
    data_rnn_origin=data_rnn_origin,
    rnn_features=rnn_features,
    rnn_target=rnn_target,
    group="train",
    max_seq_length=max_seq_length,
)

val_data_set = mroRnnDataset(
    data_rnn_origin=data_rnn_origin,
    rnn_features=rnn_features,
    rnn_target=rnn_target,
    group="valid",
    max_seq_length=max_seq_length,
)

test_data_set = mroRnnDataset(
    data_rnn_origin=data_rnn_origin,
    rnn_features=rnn_features,
    rnn_target=rnn_target,
    group="test",
    max_seq_length=max_seq_length,
)


input_feature_size = len(rnn_features)
output_size = len(rnn_target)


alpha = 1 - data_rnn_origin["mro"].eq(1).mean()
print(f"Alpha value for Focal Loss: {alpha}")
gamma = 4
print(f"Gamma value for Focal Loss: {gamma}")


def train_func():
    global best_val_loss, early_stop_counter, best_epoch
    # Model, Loss, Optimizer
    model = RnnModel(
        rnn_type=rnn_type,
        input_size=input_feature_size,
        rnn_output_size=rnn_output_size,
        output_size=output_size,
        bidirectional=bidirectional,
        num_layers=num_layers,
        pooling_method=pooling_method,
        num_heads=num_heads,
        use_last_hidden=True,
    )
    # [1] Prepare model.
    model = ray.train.torch.prepare_model(model)
    # model.to("cuda")  # This is done by `prepare_model`
    criterion = FocalLoss(alpha=alpha, gamma=gamma)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Data
    train_dataloader = DataLoader(
        train_data_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    val_dataloader = DataLoader(
        val_data_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    # [2] Prepare dataloader.
    train_loader = ray.train.torch.prepare_data_loader(train_dataloader)
    val_loader = ray.train.torch.prepare_data_loader(val_dataloader)

    # Training
    for epoch in range(num_epochs):
        if ray.train.get_context().get_world_size() > 1:
            train_loader.sampler.set_epoch(epoch)

        train_running_loss = 0.0
        # print(f"Epoch {epoch}")
        model.train()
        all_train_mro_preds = []
        all_train_mro_targets = []
        all_train_mro_scores = []
        for train_inputs, train_targets, train_lengths in train_loader:

            optimizer.zero_grad()
            train_inputs = train_inputs[:, :-1, :]
            train_targets = train_targets
            train_lengths = train_lengths

            # model_out = model(train_inputs, train_lengths)
            model_out = model(train_inputs, train_lengths)
            # the loss using Focal Loss with mean
            # which means the average loss of this batch
            loss = criterion(model_out, train_targets[:, -1, :])
            loss.backward()
            optimizer.step()
            # running loss equals to single loss * (train length / batch_szie)
            train_running_loss += loss.item()

            mro_pred = torch.sigmoid(model_out)
            mro_preds = (mro_pred > 0.5).int().cpu().numpy().flatten()
            mro_targets = train_targets[:, -1, :].cpu().numpy().flatten()

            all_train_mro_preds.extend(mro_preds)
            all_train_mro_targets.extend(mro_targets)
            all_train_mro_scores.extend(
                torch.sigmoid(model_out).detach().cpu().numpy().flatten()
            )

        train_average_loss = train_running_loss / (len(train_loader) / batch_size)
        train_f1 = f1_score(all_train_mro_targets, all_train_mro_preds)
        train_accuracy = accuracy_score(all_train_mro_targets, all_train_mro_preds)
        train_recall = recall_score(all_train_mro_targets, all_train_mro_preds)
        train_precision = precision_score(all_train_mro_targets, all_train_mro_preds)
        train_auc = roc_auc_score(all_train_mro_targets, all_train_mro_scores)
        # ------------------------------------------------
        model.eval()
        val_running_loss = 0.0
        all_val_mro_preds = []
        all_val_mro_targets = []
        all_val_mro_scores = []

        with torch.no_grad():
            for val_inputs, val_targets, val_lengths in val_loader:
                val_inputs = val_inputs[:, :-1, :]
                val_targets = val_targets

                model_out = model(val_inputs, val_lengths)
                loss = criterion(model_out, val_targets[:, -1, :])
                val_running_loss += loss.item()

                mro_pred = torch.sigmoid(model_out)
                mro_preds = (mro_pred > 0.5).int().cpu().numpy().flatten()
                mro_targets = val_targets[:, -1, :].cpu().numpy().flatten()
                all_val_mro_preds.extend(mro_preds)
                all_val_mro_targets.extend(mro_targets)
                all_val_mro_scores.extend(
                    torch.sigmoid(model_out).cpu().numpy().flatten()
                )

        val_average_loss = val_running_loss / (len(val_loader) / batch_size)
        val_f1 = f1_score(all_val_mro_targets, all_val_mro_preds)
        val_accuracy = accuracy_score(all_val_mro_targets, all_val_mro_preds)
        val_recall = recall_score(all_val_mro_targets, all_val_mro_preds)
        val_precision = precision_score(all_val_mro_targets, all_val_mro_preds)
        val_auc = roc_auc_score(all_val_mro_targets, all_val_mro_scores)
        # ------------------------------------------------
        # [3] Report metrics and checkpoint.
        metrics = {
            "epoch": epoch,
            "train_average_loss": train_average_loss,
            "val_average_loss": val_average_loss,
            "train_f1": train_f1,
            "train_accuracy": train_accuracy,
            "train_recall": train_recall,
            "train_precision": train_precision,
            "train_auc": train_auc,
            "val_f1": val_f1,
            "val_accuracy": val_accuracy,
            "val_recall": val_recall,
            "val_precision": val_precision,
            "val_auc": val_auc,
        }

        # with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
        #     torch.save(
        #         model.state_dict(), os.path.join(temp_checkpoint_dir, "model.pt")
        #     )
        #     ray.train.report(
        #         metrics,
        #         checkpoint=ray.train.Checkpoint.from_directory(temp_checkpoint_dir),
        #     )
        ray.train.report(
            metrics,
            checkpoint=ray.train.Checkpoint.from_directory(run_dir),
        )
        if ray.train.get_context().get_world_rank() == 0:
            df = pd.DataFrame([metrics])
            # Append to CSV
            write_header = not os.path.exists(result_csv_file)
            df.to_csv(result_csv_file, mode="a", header=write_header, index=False)

            print(metrics)

        if val_average_loss < best_val_loss:
            best_val_loss = val_average_loss
            best_epoch = epoch
            early_stop_counter = 0
            torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))
        else:
            early_stop_counter += 1

        if early_stop_counter >= early_stop_patience:
            print(f"Early stopping at epoch {epoch} due to no improvement in val loss.")
            break


# [4] Configure scaling and resource requirements.
scaling_config = ray.train.ScalingConfig(num_workers=4, use_gpu=True)

# [5] Launch distributed training job.
trainer = ray.train.torch.TorchTrainer(
    train_func,
    scaling_config=scaling_config,
)
result = trainer.fit()