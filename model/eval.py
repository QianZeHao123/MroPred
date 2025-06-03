from .data_prep import preprocess_data
from .model import RnnModel
from utils import create_train_test_group
from model import mroRnnDataset
import torch
from collections import OrderedDict
from model import FocalLoss
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    recall_score,
    precision_score,
    roc_auc_score,
)
from utils import collate_fn
import pandas as pd
import numpy as np
import os


def model_eval(
    device,
    experiment_name: str,
    csv_file_name: str,
    target_mro: list,
    maintain_repair_mro: str,
    add_mro_prev: bool,
    add_purchase_time: bool,
    add_driver_behavior: bool,
    agg_weeks: int,
    agg_fun: list,
    sample_frac: float,
    test_size: float,
    valid_size: float,
    seq_length: int,
    model_path: str,
    # control parameter: Focal Loss
    gamma: int,
    # control parameter:
    rnn_type: str,
    rnn_output_size: int,
    bidirectional: bool,
    num_layers: int,
    pooling_method: str,
    num_heads: int,
    batch_size: int,
    # save the results
    result_file="./output/lstm/evaluation_results.csv",
    purchase_year: int = None,
):
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
    original_data = pd.read_csv(csv_file_name, index_col=0, engine="pyarrow")
    # add purchase_year id select here:
    if purchase_year is not None and "purchase_yr_nbr" in original_data.columns:
        selected_ids = original_data.loc[
            original_data["purchase_yr_nbr"] == purchase_year, "id"
        ].unique()
        print(
            f"Filtering by purchase_year={purchase_year}, selected {len(selected_ids)} ids."
        )
        data = data[data["id"].isin(selected_ids)]
    else:
        selected_ids = data["id"].unique()
        print(f"No Purchase Year Selected, selected {len(selected_ids)} ids.")

    rnn_features = prep_data["rnn_features"]
    rnn_target = prep_data["rnn_target"]
    col_rnn_origin = ["id"] + rnn_features + rnn_target
    data_rnn_origin = data[col_rnn_origin].copy()

    data_rnn_origin = create_train_test_group(
        data_rnn_origin,
        sample_frac=sample_frac,
        test_size=test_size,
        valid_size=valid_size,
        random_state=42,
    )

    test_data_set = mroRnnDataset(
        data_rnn_origin=data_rnn_origin,
        rnn_features=rnn_features,
        rnn_target=rnn_target,
        group="test",
        seq_length=seq_length,
    )

    input_feature_size = len(rnn_features)
    output_size = len(rnn_target)

    alpha = 1 - data_rnn_origin["target_mro"].eq(1).mean()
    print(f"Alpha value for Focal Loss: {alpha}")
    gamma = gamma
    print(f"Gamma value for Focal Loss: {gamma}")

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

    # model_path = './output/lstm/benchmark_old_agg/model.pt'
    state_dict = torch.load(model_path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k  # remove 'module.'
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model.eval()
    model.to(device)

    criterion = FocalLoss(alpha=alpha, gamma=gamma).to(device)
    test_loader = DataLoader(
        test_data_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=16,
    )

    all_test_mro_preds = []
    all_test_mro_targets = []
    all_test_mro_scores = []
    test_running_loss = 0.0

    with torch.no_grad():
        for test_inputs, test_targets, test_lengths in test_loader:
            test_inputs = test_inputs[:, :-1, :].to(device)
            test_targets = test_targets.to(device)
            test_lengths = test_lengths.to("cpu")

            # forward
            model_out = model(test_inputs, test_lengths)

            # calculate the loss
            loss = criterion(model_out, test_targets[:, -1, :])
            test_running_loss += loss.item()

            # get the predicted result
            mro_pred = torch.sigmoid(model_out)
            # mro_preds = (mro_pred > 0.5).int().cpu().numpy().flatten()
            mro_targets = test_targets[:, -1, :].cpu().numpy().flatten()

            # all_test_mro_preds.extend(mro_preds)
            all_test_mro_targets.extend(mro_targets)
            all_test_mro_scores.extend(mro_pred.cpu().numpy().flatten())

    best_threshold = 0.5
    best_f1 = 0.0
    thresholds = np.arange(0.1, 0.91, 0.01)

    for threshold in thresholds:
        preds = (np.array(all_test_mro_scores) > threshold).astype(int)
        f1 = f1_score(all_test_mro_targets, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print(f"Best Threshold: {best_threshold:.2f}, Best F1 Score: {best_f1:.4f}")

    test_average_loss = test_running_loss / (len(test_loader) / batch_size)
    # test_f1 = f1_score(all_test_mro_targets, all_test_mro_preds)
    # test_accuracy = accuracy_score(all_test_mro_targets, all_test_mro_preds)
    # test_recall = recall_score(all_test_mro_targets, all_test_mro_preds)
    # test_precision = precision_score(all_test_mro_targets, all_test_mro_preds)
    # test_auc = roc_auc_score(all_test_mro_targets, all_test_mro_scores)
    final_preds = (np.array(all_test_mro_scores) > best_threshold).astype(int)
    test_f1 = f1_score(all_test_mro_targets, final_preds)
    test_accuracy = accuracy_score(all_test_mro_targets, final_preds)
    test_recall = recall_score(all_test_mro_targets, final_preds)
    test_precision = precision_score(all_test_mro_targets, final_preds)
    test_auc = roc_auc_score(all_test_mro_targets, all_test_mro_scores)

    metrics = {
        "experiment_name": experiment_name,
        "test_average_loss": test_average_loss,
        "test_f1": test_f1,
        "test_accuracy": test_accuracy,
        "test_recall": test_recall,
        "test_precision": test_precision,
        "test_auc": test_auc,
        "best_threshold": best_threshold,
    }

    pd.DataFrame([metrics]).to_csv(
        result_file,
        index=False,
        mode="a",
        header=not os.path.exists(result_file),
    )
    return metrics
