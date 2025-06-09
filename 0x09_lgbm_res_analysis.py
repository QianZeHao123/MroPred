import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.metrics import (
    f1_score,
    accuracy_score,
    recall_score,
    precision_score,
)

import os


lgbm_res_data_path = "./output/lgbm/lgbm_combined_results.csv"
lgbm_res_data = pd.read_csv(lgbm_res_data_path)

lgbm_perf_file_path = "./output/lgbm/lgbm_perf_results.csv"


def prepare_data(data: pd.DataFrame):
    """Load and split the dataset into train, validation, and test sets."""

    train_dataset = data[data["group"] == "train"]
    valid_dataset = data[data["group"] == "valid"]
    test_dataset = data[data["group"] == "test"]

    train_dataset = train_dataset.drop(["group", "id"], axis=1)
    valid_dataset = valid_dataset.drop(["group", "id"], axis=1)
    test_dataset = test_dataset.drop(["group", "id"], axis=1)

    return train_dataset, valid_dataset, test_dataset


for index, trial in lgbm_res_data.iterrows():
    # print(trial['Trial Name'])
    print("Start to analysis trial:", trial["Trial Name"])
    trial_name = trial["Trial Name"]
    test_auc = trial["Test AUC"]
    # ----------------------------------------
    # load the LightGBM model
    trial_model_path = trial["Best Model Path"]
    trail_model = lgb.Booster(model_file=trial_model_path)
    # ----------------------------------------
    # load the data
    trial_data = pd.read_parquet(trial["Data Path"])
    train_dataset, valid_dataset, test_dataset = prepare_data(trial_data)
    train_X = train_dataset.drop(columns=["target_mro"], axis=1)

    train_y = train_dataset["target_mro"]
    train_pred = trail_model.predict(train_X)

    # ----------------------------------------
    # get best threshold from training set
    best_threshold = 0.5
    best_f1 = 0.0
    thresholds = np.arange(0.02, 0.98, 0.01)

    for threshold in thresholds:
        preds = (np.array(train_pred) > threshold).astype(int)
        f1 = f1_score(train_y, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print(f"Best Threshold: {best_threshold:.2f}, Best F1 Score: {best_f1:.4f}")

    # ----------------------------------------
    # apply best threshold to test set
    test_X = test_dataset.drop(columns=["target_mro"], axis=1)
    test_y = test_dataset["target_mro"]

    test_pred = trail_model.predict(test_X)
    test_res = (np.array(test_pred) > best_threshold).astype(int)

    test_f1 = f1_score(test_y, test_res)
    test_accuracy = accuracy_score(test_y, test_res)
    test_recall = recall_score(test_y, test_res)
    test_precision = precision_score(test_y, test_res)

    results_df = pd.DataFrame(
        {
            "Trial Name": [trial_name],
            "Best Threshold": [best_threshold],
            "Best Test F1": [test_f1],
            "Best Test Accuracy": [test_accuracy],
            "Best Test Recall": [test_recall],
            "Best Test Precision": [test_precision],
            "Test AUC Score": [test_auc],
        }
    )
    
    results_df.to_csv(
        lgbm_perf_file_path,
        mode="a",
        index=False,
        header=not os.path.isfile(lgbm_perf_file_path),
    )
