import pandas as pd
import os
from datetime import datetime
import torch
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from torch import optim

from sklearn.metrics import (
    f1_score,
    accuracy_score,
    recall_score,
    precision_score,
    roc_auc_score,
)


from utils import create_train_test_group
from utils import mroRnnDataset as mroRnnDataset
from utils import collate_fn
import torch.nn.functional as F


from model import FocalLoss
from model import RnnModel

# Check if CUDA (NVIDIA GPU) is available
if torch.cuda.is_available():
    # Get the number of available CUDA devices
    num_gpus = torch.cuda.device_count()
    print(f"Number of CUDA devices available: {num_gpus}")
    # Select a specific GPU (e.g., GPU 0)
    device = torch.device("cuda:6")  # Use "cuda:1" for GPU 1, etc.
    print(f"Using device: {torch.cuda.get_device_name(device)}")
else:
    device = "cpu"
    print("CUDA is not available, using CPU")
print(f"Using {device} device")


# ---------------------------------------------------------
# data preprocessing control parameter
sample_frac = 1
test_size = 0.1
max_seq_length = 8
batch_size = 4096
num_workers = 16

learning_rate = 0.0005
num_epochs = 1000


best_val_loss = float("inf")
best_val_auc = 0.0
best_val_f1 = 0.0

best_epoch = float("inf")
counter = 0
patience = 20


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

log_dir = "./Out"
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
timestamp = 'weekly_level_without_user_behavior'
log_file_path = os.path.join(log_dir, f"training_log_{timestamp}.csv")

pt_model_dir = os.path.join(log_dir, "models")
os.makedirs(pt_model_dir, exist_ok=True)


model_save_path = os.path.join(pt_model_dir, f"best_model_{timestamp}.pth")
# pt_model_save_path = os.path.join(pt_model_dir, f"best_model_{timestamp}.pt")


file_name = "./Data/mro_daily_clean.csv"
# ---------------------------------------------------------
data = pd.read_csv(file_name, index_col=0, engine="pyarrow")


continuous_variable = [
    # "hard_braking",
    # "hard_acceleration",
    # "speeding_sum",
    # "day_mileage",
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
        # "hard_braking": ["mean", "max", "min", "std", "skew"],
        # "hard_acceleration": ["mean", "max", "min", "std", "skew"],
        # "speeding_sum": ["mean", "max", "min", "std", "skew"],
        # "day_mileage": ["mean", "max", "min", "std", "skew"],
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
    data_rnn_origin, sample_frac=sample_frac, test_size=test_size
)
# rnn_target = ["mro"]
rnn_target = mro
# ---------------------------------------------------------


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
    bidirectional=bidirectional,
    num_layers=num_layers,
    # value =  None, 'max', 'avg', 'attention', 'multihead_attention'
    pooling_method=pooling_method,
    num_heads=num_heads,
    use_last_hidden=True,
).to(device)

print(model)


# ---------------------------------------------------------
alpha = 1 - data_rnn_origin["mro"].eq(1).mean()
print(f"Alpha value for Focal Loss: {alpha}")
gamma = 6
print(f"Gamma value for Focal Loss: {gamma}")
criterion = FocalLoss(alpha=alpha, gamma=gamma).to(device)


# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
# add learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=3)

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
        # the loss using Focal Loss with mean
        # which means the average loss of this batch
        loss = criterion(model_out, train_targets[:, -1, :])
        loss.backward()
        optimizer.step()
        # running loss equals to single loss * (train length / batch_szie)
        running_loss += loss.item()

        mro_pred = torch.sigmoid(model_out)
        mro_preds = (mro_pred > 0.5).int().cpu().numpy().flatten()
        mro_targets = train_targets[:, -1, :].cpu().numpy().flatten()

        all_train_mro_preds.extend(mro_preds)
        all_train_mro_targets.extend(mro_targets)
        all_train_mro_scores.extend(
            torch.sigmoid(model_out).detach().cpu().numpy().flatten()
        )

    average_loss = running_loss / (len(train_dataloader) / batch_size)
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
        for val_inputs, val_targets, val_lengths in test_dataloader:
            val_inputs = val_inputs.to(device)
            val_targets = val_targets.to(device)
            model_out = model(val_inputs, val_lengths)
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

        average_val_loss = val_loss / (len(test_dataloader) / batch_size)
        print(f"Validation Loss: {average_val_loss}")
        # update learning rate with scheduler
        scheduler.step(average_val_loss)

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

    if best_val_loss > average_val_loss:
        best_val_loss = average_val_loss
        counter = 0
        # save the model
        torch.save(model.state_dict(), model_save_path)
        # torchscript_model = torch.jit.script(model)
        # torchscript_model.save(pt_model_save_path)
        best_epoch = epoch
        best_log_entry = log_entry

    else:
        counter += 1
        if counter >= patience:
            print("Early stopping!")
            print("Best Epoch is:", best_epoch)
            break
