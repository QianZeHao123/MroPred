from model import preprocess_data_lgbm as preprocess_data
import os
import pandas as pd
from utils import create_train_test_group
import lightgbm as lgb


from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.lightgbm import TuneReportCheckpointCallback

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    roc_auc_score,
)

import shutil


"""
The following part is the important parameter for the LightGBM training
"""
# ------------------------------------------
# our_model_benchmark = "our_model"
our_model_benchmark = "benchmark"
# ------------------------------------------
# trail_name = "Our Model GBDT"
# trail_name = "Our Model RF"
# trail_name = "Our Model DART"

# trail_name = "Benchmark GBDT"
# trail_name = "Benchmark DART"
# trail_name = "Benchmark RF"

# trail_name = "Our Model GBDT -- 4 week"
# trail_name = "Our Model RF -- 4 week"
# trail_name = "Our Model DART -- 4 week"

# trail_name = "Benchmark GBDT -- 4 week"
# trail_name = "Benchmark DART -- 4 week"
# trail_name = "Benchmark RF -- 4 week"

# trail_name = "Our Model GBDT -- 12 week"
# trail_name = "Our Model RF -- 12 week"
# trail_name = "Our Model DART -- 12 week"

# trail_name = "Benchmark GBDT -- 12 week"
# trail_name = "Benchmark DART -- 12 week"
# trail_name = "Benchmark RF -- 12 week"

# trail_name = "Our Model GBDT -- 1 month"
# trail_name = "Our Model RF -- 1 month"
# trail_name = "Our Model DART -- 1 month"

# trail_name = "Benchmark GBDT -- 1 month"
# trail_name = "Benchmark DART -- 1 month"
# trail_name = "Benchmark RF -- 1 month"

# trail_name = "Our Model GBDT -- 2 month"
# trail_name = "Our Model RF -- 2 month"
# trail_name = "Our Model DART -- 2 month"

# trail_name = "Benchmark GBDT -- 2 month"
# trail_name = "Benchmark DART -- 2 month"
# trail_name = "Benchmark RF -- 2 month"

# trail_name = "Our Model GBDT -- 3 month"
# trail_name = "Our Model RF -- 3 month"
# trail_name = "Our Model DART -- 3 month"

# trail_name = "Benchmark GBDT -- 3 month"
# trail_name = "Benchmark DART -- 3 month"
trail_name = "Benchmark RF -- 3 month"
# ------------------------------------------
# main difference between benchmark and our model
# add_driver_behavior = True
add_driver_behavior = False
# ------------------------------------------
# time_window = 8
# time_window = 4
# time_window = 12

# time_window = 1
# time_window = 2
time_window = 3

# agg_scale = "weekly"
agg_scale = "monthly"

# ------------------------------------------
# boosting could be "gbdt", "rf" (random forest) and "dart"
# boosting: str = "gbdt"
boosting: str = "rf"
# boosting: str = "dart"

# ------------------------------------------
csv_file_name = "./Data/mro_daily_clean.csv"
target_mro = ["mro"]

maintain_repair_mro = "full"

add_mro_prev = True
add_purchase_time = True

agg_fun = ["mean", "sum", "max", "min", "std", "skew"]

# ------------------------------------------
# LightGBM Parameters
metric: list = ["binary_logloss", "binary_error", "auc", "average_precision"]
learning_rate: float = 0.05
num_leaves: int = 64
max_depth: int = 8
is_unbalance: bool = True


# data record folder
data_lgbm_file_name = f"data_lgbm_db{int(add_driver_behavior)}_mp{int(add_mro_prev)}_pt{int(add_purchase_time)}_as_{agg_scale}_tw{time_window}.gzip"
data_lgbm_path = os.path.join(f"./Data/{our_model_benchmark}", data_lgbm_file_name)
data_lgbm_path = os.path.abspath(data_lgbm_path)

# ------------------------------------------
# model record folder
tune_result_storage_path = f"./output/lgbm/{our_model_benchmark}/lgbm_tuning_results_{boosting}_db{int(add_driver_behavior)}_mp{int(add_mro_prev)}_pt{int(add_purchase_time)}_as_{agg_scale}_tw{time_window}"

# ------------------------------------------
result_combine_folder = "./output/lgbm"
os.makedirs(result_combine_folder, exist_ok=True)
result_combine_path = os.path.join(result_combine_folder, "lgbm_combined_results.csv")

# ------------------------------------------
# train control and scaling control parameters
num_workers = 4
num_boost_round = 1000
early_stopping_round = 10


"""
End of the Parameter Config
"""


if os.path.isfile(data_lgbm_path):
    print(f"Data file {data_lgbm_path} exists.")
    data_lgbm = pd.read_parquet(data_lgbm_path)
else:
    print(f"{data_lgbm_path} does not exist.")
    # control parameter: data preparation
    data = preprocess_data(
        file_name=csv_file_name,
        target_mro=target_mro,
        maintain_repair_mro=maintain_repair_mro,
        add_mro_prev=add_mro_prev,
        add_purchase_time=add_purchase_time,
        add_driver_behavior=add_driver_behavior,
        agg_scale=agg_scale,
        agg_fun=agg_fun,
        time_window=time_window,
    )

    data_lgbm = create_train_test_group(
        data=data,
        sample_frac=1.0,
        test_size=0.1,
        valid_size=0.1,
        random_state=42,
    )

    data_lgbm.to_parquet(data_lgbm_path, compression="gzip", engine="pyarrow")


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


def train_lgbm(config, train_dataset: pd.DataFrame, valid_dataset: pd.DataFrame):
    # train_dataset, valid_dataset, _ = prepare_data(data_lgbm)
    # train set
    train_set = lgb.Dataset(
        train_dataset.drop(["target_mro"], axis=1), label=train_dataset["target_mro"]
    )
    # valid set
    valid_X = valid_dataset.drop(["target_mro"], axis=1)
    valid_y = valid_dataset["target_mro"]
    valid_set = lgb.Dataset(valid_X, label=valid_y)

    gbm = lgb.train(
        config,
        train_set,
        valid_sets=[valid_set],
        valid_names=["eval"],
        callbacks=[
            TuneReportCheckpointCallback(
                {
                    "binary_error": "eval-binary_error",
                    "auc": "eval-auc",
                    # "binary_logloss": "eval-binary_logloss",
                }
            )
        ],
    )


if os.path.exists(tune_result_storage_path):
    shutil.rmtree(tune_result_storage_path)
    print(f"Already exists {tune_result_storage_path}, remove it and create a new one.")

os.makedirs(tune_result_storage_path, exist_ok=True)
tune_result_storage_path = os.path.abspath(tune_result_storage_path)


if __name__ == "__main__":
    if boosting == "gbdt":
        config = {
            "objective": "binary",
            "metric": ["binary_logloss", "binary_error", "auc", "average_precision"],
            "verbose": 1,
            "is_unbalance": True,
            # "max_depth": 8,
            "max_depth": tune.randint(4, 20),
            "boosting_type": boosting,
            "device_type": "cpu",
            "num_leaves": tune.randint(10, 1000),
            "learning_rate": tune.loguniform(1e-8, 1e-1),
            # "bagging_fraction": tune.uniform(0.01, 0.99),
            # "bagging_freq": tune.uniform(0.01, 0.99),
            # "learning_rate": 0.05,
        }
    elif boosting == "dart":
        config = {
            "objective": "binary",
            "metric": ["binary_logloss", "binary_error", "auc", "average_precision"],
            "verbose": 1,
            "is_unbalance": True,
            # "max_depth": 8,
            "max_depth": tune.randint(4, 20),
            "boosting_type": boosting,
            "device_type": "cpu",
            "num_leaves": tune.randint(10, 1000),
            "learning_rate": tune.loguniform(1e-8, 1e-1),
            # "bagging_fraction": tune.uniform(0.01, 0.99),
            # "bagging_freq": tune.randint(1, 10),
            # "feature_fraction": tune.uniform(0.5, 1.0),
        }
    elif boosting == "rf":
        config = {
            "objective": "binary",
            "metric": ["binary_logloss", "binary_error", "auc", "average_precision"],
            "verbose": 1,
            "is_unbalance": True,
            # "max_depth": 8,
            "max_depth": tune.randint(4, 20),
            "boosting_type": boosting,
            "device_type": "cpu",
            "num_leaves": tune.randint(10, 1000),
            "learning_rate": 1,
            "bagging_fraction": tune.uniform(0.01, 0.99),
            "bagging_freq": tune.randint(1, 10),
            "feature_fraction": tune.uniform(0.5, 1.0),
        }
    else:
        config = {
            "objective": "binary",
            "metric": ["binary_logloss", "binary_error", "auc", "average_precision"],
            "verbose": 1,
            "is_unbalance": True,
            # "max_depth": 8,
            "max_depth": tune.randint(4, 20),
            "boosting_type": "gbdt",
            "device_type": "cpu",
            "num_leaves": tune.randint(10, 1000),
            "learning_rate": tune.loguniform(1e-8, 1e-1),
        }

    tuner = tune.Tuner(
        # train_lgbm,
        # tune.with_parameters(
        #     train_lgbm, train_dataset=train_dataset, valid_dataset=valid_dataset
        # ),
        tune.with_resources(
            tune.with_parameters(
                train_lgbm, train_dataset=train_dataset, valid_dataset=valid_dataset
            ),
            {"cpu": 16},
        ),
        tune_config=tune.TuneConfig(
            # metric="binary_error",
            # mode="min",
            metric="auc",
            mode="max",
            scheduler=ASHAScheduler(),
            num_samples=5,
        ),
        run_config=tune.RunConfig(
            name="lgbm_tuning_experiment",
            storage_path=tune_result_storage_path,
        ),
        param_space=config,
    )
    results = tuner.fit()
    # print(f"Best hyperparameters found were: {results.get_best_result().config}")


best_model_path = os.path.join(results.get_best_result().checkpoint.path, "model.txt")
booster = lgb.Booster(model_file=best_model_path)


def get_X_y(df):
    X = df.drop("target_mro", axis=1)
    y = df["target_mro"]
    return X, y


X_train, y_train = get_X_y(train_dataset)
X_valid, y_valid = get_X_y(valid_dataset)
X_test, y_test = get_X_y(test_dataset)


def predict_and_eval(booster, X, y_true: pd.DataFrame, dataset_name="dataset"):
    y_prob = booster.predict(X)
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)

    print(f"\nEvaluation on {dataset_name}:")
    print(f"Accuracy:  {acc:.5f}")
    print(f"Precision: {precision:.5f}")
    print(f"Recall:    {recall:.5f}")
    print(f"F1 Score:  {f1:.5f}")
    print(f"AUC:       {auc:.5f}")

    result_df = pd.DataFrame(
        {"y_true": y_true.values, "y_prob": y_prob, "y_pred": y_pred}
    )

    # return acc, precision, recall, f1, auc, result_df
    return {
        "auc": auc,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "result_df": result_df,
    }


train_results = predict_and_eval(booster, X_train, y_train, "Train Set")
valid_results = predict_and_eval(booster, X_valid, y_valid, "Validation Set")
test_results = predict_and_eval(booster, X_test, y_test, "Test Set")

results_df = pd.DataFrame(
    {
        "Trial Name": [trail_name],
        "Data Path": [data_lgbm_path],
        "Best Model Path": [best_model_path],
        # ------------------------------------------
        "csv_file_name": [csv_file_name],
        "target_mro": [target_mro],
        "maintain_repair_mro": [maintain_repair_mro],
        "add_mro_prev": [add_mro_prev],
        "add_purchase_time": [add_purchase_time],
        "add_driver_behavior": [add_driver_behavior],
        "agg_scale": [agg_scale],
        "agg_fun": [agg_fun],
        "time_window": [time_window],
        # ------------------------------------------
        # metrics record
        "Train Accuracy": [train_results["accuracy"]],
        "Train Precision": [train_results["precision"]],
        "Train Recall": [train_results["recall"]],
        "Train F1 Score": [train_results["f1_score"]],
        "Train AUC": [train_results["auc"]],
        "Validation Accuracy": [valid_results["accuracy"]],
        "Validation Precision": [valid_results["precision"]],
        "Validation Recall": [valid_results["recall"]],
        "Validation F1 Score": [valid_results["f1_score"]],
        "Validation AUC": [valid_results["auc"]],
        "Test Accuracy": [test_results["accuracy"]],
        "Test Precision": [test_results["precision"]],
        "Test Recall": [test_results["recall"]],
        "Test F1 Score": [test_results["f1_score"]],
        "Test AUC": [test_results["auc"]],
    }
)


results_df.to_csv(
    result_combine_path,
    mode="a",
    index=False,
    header=not os.path.isfile(result_combine_path),
)

print(f"Results appended to: {result_combine_path}")
