import pandas as pd
import lightgbm as lgb

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    roc_auc_score,
)
import numpy as np

import os


lstm_maintenance_repair_file_folder = "./output/lstm"
os.makedirs(lstm_maintenance_repair_file_folder, exist_ok=True)
lstm_maintenance_repair_file_name = "lstm_maintenance_repair.csv"
lstm_maintenance_repair_file_path = os.path.join(
    lstm_maintenance_repair_file_folder, lstm_maintenance_repair_file_name
)

lstm_maintenance_repair_file_path = os.path.abspath(lstm_maintenance_repair_file_path)


gbdt_db0_model = "/home/user14/Cyber/MroPred/output/lgbm/benchmark/lgbm_tuning_results_gbdt_db0_mp1_pt1_as_weekly_tw8/lgbm_tuning_experiment/train_lgbm_fe064_00001_1_learning_rate=0.0358,max_depth=19,num_leaves=368_2025-06-09_02-32-54/checkpoint_000000/model.txt"

gbdt_db1_model = "/home/user14/Cyber/MroPred/output/lgbm/our_model/lgbm_tuning_results_gbdt_db1_mp1_pt1_as_weekly_tw8/lgbm_tuning_experiment/train_lgbm_89478_00002_2_learning_rate=0.0099,max_depth=8,num_leaves=959_2025-06-09_10-22-05/checkpoint_000000/model.txt"

rf_db0_model = "/home/user14/Cyber/MroPred/output/lgbm/benchmark/lgbm_tuning_results_rf_db0_mp1_pt1_as_weekly_tw8/lgbm_tuning_experiment/train_lgbm_1a76b_00000_0_bagging_fraction=0.7738,bagging_freq=7,feature_fraction=0.7924,max_depth=18,num_leaves=425_2025-06-09_09-50-21/checkpoint_000000/model.txt"

rf_db1_model = "/home/user14/Cyber/MroPred/output/lgbm/our_model/lgbm_tuning_results_rf_db1_mp1_pt1_as_weekly_tw8/lgbm_tuning_experiment/train_lgbm_88c5d_00000_0_bagging_fraction=0.6658,bagging_freq=4,feature_fraction=0.7009,max_depth=15,num_leaves=577_2025-06-09_02-01-00/checkpoint_000000/model.txt"

dart_db0_model = "/home/user14/Cyber/MroPred/output/lgbm/benchmark/lgbm_tuning_results_dart_db0_mp1_pt1_as_weekly_tw8/lgbm_tuning_experiment/train_lgbm_c0510_00007_7_learning_rate=0.0211,max_depth=17,num_leaves=314_2025-06-09_10-02-09/checkpoint_000000/model.txt"

dart_db1_model = "/home/user14/Cyber/MroPred/output/lgbm/our_model/lgbm_tuning_results_dart_db1_mp1_pt1_as_weekly_tw8/lgbm_tuning_experiment/train_lgbm_2388e_00014_14_learning_rate=0.0588,max_depth=17,num_leaves=144_2025-06-09_02-19-38/checkpoint_000000/model.txt"


benchmark_data_maintenance = (
    "./Data/maintenance_repair_comparison/benchmark_data_maintenance.gzip"
)

benchmark_data_repair = (
    "./Data/maintenance_repair_comparison/benchmark_data_repair.gzip"
)

our_model_data_maintenance = (
    "./Data/maintenance_repair_comparison/our_model_data_maintenance.gzip"
)

our_model_data_repair = (
    "./Data/maintenance_repair_comparison/our_model_data_repair.gzip"
)


# best_model_path = dart_db0_model
# data_path = benchmark_data_maintenance
# trial_name: str = "DART-benchmark-maintenance"

# best_model_path = rf_db0_model
# data_path = benchmark_data_maintenance
# trial_name: str = "RF-benchmark-maintenance"

# best_model_path = gbdt_db0_model
# data_path = benchmark_data_maintenance
# trial_name: str = "GBDT-benchmark-maintenance"

# best_model_path = dart_db0_model
# data_path = benchmark_data_repair
# trial_name: str = "DART-benchmark-repair"

# best_model_path = rf_db0_model
# data_path = benchmark_data_repair
# trial_name: str = "RF-benchmark-repair"

# best_model_path = gbdt_db0_model
# data_path = benchmark_data_repair
# trial_name: str = "GBDT-benchmark-repair"

# ----------------------------------------

# best_model_path = dart_db1_model
# data_path = our_model_data_maintenance
# trial_name: str = "DART-our_model-maintenance"

# best_model_path = rf_db1_model
# data_path = our_model_data_maintenance
# trial_name: str = "RF-our_model-maintenance"

# best_model_path = gbdt_db1_model
# data_path = our_model_data_maintenance
# trial_name: str = "GBDT-our_model-maintenance"

# ----------------------------------------

# best_model_path = dart_db1_model
# data_path = our_model_data_repair
# trial_name: str = "DART-our_model-repair"

# best_model_path = rf_db1_model
# data_path = our_model_data_repair
# trial_name: str = "RF-our_model-repair"

best_model_path = gbdt_db1_model
data_path = our_model_data_repair
trial_name: str = "GBDT-our_model-repair"


booster = lgb.Booster(model_file=best_model_path)
data_lgbm = pd.read_parquet(data_path)


def prepare_data(data: pd.DataFrame):
    """Load and split the dataset into train, validation, and test sets."""

    train_dataset = data[data["group"] == "train"]
    valid_dataset = data[data["group"] == "valid"]
    test_dataset = data[data["group"] == "test"]

    train_dataset = train_dataset.drop(["group", "id"], axis=1)
    valid_dataset = valid_dataset.drop(["group", "id"], axis=1)
    test_dataset = test_dataset.drop(["group", "id"], axis=1)

    return train_dataset, valid_dataset, test_dataset


train_dataset, valid_dataset, test_dataset = prepare_data(data_lgbm)

train_X = train_dataset.drop(columns=["target_mro"], axis=1)

train_y = train_dataset["target_mro"]
train_pred = booster.predict(train_X)

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

test_pred = booster.predict(test_X)
test_res = (np.array(test_pred) > best_threshold).astype(int)

test_f1 = f1_score(test_y, test_res)
test_accuracy = accuracy_score(test_y, test_res)
test_recall = recall_score(test_y, test_res)
test_precision = precision_score(test_y, test_res)

test_auc = roc_auc_score(test_y, test_pred)


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
    lstm_maintenance_repair_file_path,
    mode="a",
    index=False,
    header=not os.path.isfile(lstm_maintenance_repair_file_path),
)
