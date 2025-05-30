# control parameter: data preparation
csv_file_name = "./Data/mro_daily_clean.csv"
target_mro = ["mro"]
maintain_repair_mro = "full"

add_mro_prev = False
add_purchase_time = True
add_driver_behavior = True
agg_weeks = 1
agg_fun = ["mean", "sum", "max", "min", "std", "skew"]


# control parameter: sample_frac, test_size, valid_size
sample_frac = 1.0
test_size = 0.1
valid_size = 0.1


# control parameter: length of LSTM seqence input
seq_length = 8


# output folder and file name
out_folder = "./output/lstm/our_model_without_prev_mro"
result_csv_file_name = "ray_train_log.csv"


# control parameter: Focal Loss
gamma = 4

# control parameter: early stop
best_val_loss = float("inf")
early_stop_patience = 20
early_stop_counter = 0
best_epoch = 0

# control parameter:
rnn_type = "LSTM"
rnn_output_size = 128
bidirectional = True
num_layers = 2
pooling_method = None
num_heads = None
use_last_hidden = True

learning_rate = 0.0005
batch_size = 4096

num_workers = 4
num_epochs = 1000
# ------------------------------------------------

import os
import shutil

# set GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3, 4, 5, 6, 7"

if os.path.exists(out_folder):
    shutil.rmtree(out_folder)
    print(f"Deleted existing directory: {out_folder}")

os.makedirs(out_folder, exist_ok=True)
out_folder = os.path.abspath(out_folder)
print("Create a new output folder")
print(f"It is located in {out_folder}")

result_csv_file = os.path.join(out_folder, result_csv_file_name)
result_csv_file = os.path.abspath(result_csv_file)
print(f"The model training folder is {result_csv_file}")


from model import preprocess_data

prep_data = preprocess_data(
    file_name=csv_file_name,
    target_mro=target_mro,
    maintain_repair_mro=maintain_repair_mro,
    add_mro_prev=add_mro_prev,
    add_purchase_time=add_purchase_time,
    add_driver_behavior=add_driver_behavior,
    agg_weeks=agg_weeks,
    agg_fun=agg_fun,
)

data = prep_data["data"]
rnn_features = prep_data["rnn_features"]
rnn_target = prep_data["rnn_target"]
col_rnn_origin = ["id"] + rnn_features + rnn_target
data_rnn_origin = data[col_rnn_origin].copy()


from utils import create_train_test_group

data_rnn_origin = create_train_test_group(
    data_rnn_origin,
    sample_frac=sample_frac,
    test_size=test_size,
    valid_size=valid_size,
    random_state=42,
)


from model import mroRnnDataset


train_data_set = mroRnnDataset(
    data_rnn_origin=data_rnn_origin,
    rnn_features=rnn_features,
    rnn_target=rnn_target,
    group="train",
    seq_length=seq_length,
)

val_data_set = mroRnnDataset(
    data_rnn_origin=data_rnn_origin,
    rnn_features=rnn_features,
    rnn_target=rnn_target,
    group="valid",
    seq_length=seq_length,
)


input_feature_size = len(rnn_features)
output_size = len(rnn_target)

alpha = 1 - data_rnn_origin["target_mro"].eq(1).mean()
print(f"Alpha value for Focal Loss: {alpha}")
gamma = gamma
print(f"Gamma value for Focal Loss: {gamma}")


from model import FocalLoss
from model import RnnModel
from utils import collate_fn

import torch
from torch import optim
from torch.utils.data import DataLoader


import ray.train.torch

from sklearn.metrics import (
    f1_score,
    accuracy_score,
    recall_score,
    precision_score,
    roc_auc_score,
)

import pandas as pd


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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=3)

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
        scheduler.step(val_average_loss)
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

        ray.train.report(
            metrics,
            checkpoint=ray.train.Checkpoint.from_directory(out_folder),
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
            torch.save(model.state_dict(), os.path.join(out_folder, "model.pt"))
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
