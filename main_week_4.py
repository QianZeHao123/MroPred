import optuna
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import pandas as pd
import lightgbm as lgb
from constants import (
    driver_behavior,
    vehicle_attributes,
    driver_attributes,
    driver_navigation,
    gis_attributes,
    record_day,
    target_mro,
)
from utils import evaluate_model
from rich.console import Console
from rich.markdown import Markdown

console = Console()
console.print(Markdown("# MRO Model Training"))
console.print(Markdown("## Weekly Level Aggregation"))
# Load the data
file_name = "./Data/mro_daily_clean.csv"
data = pd.read_csv(file_name, index_col=0, engine="pyarrow")
console.print("Finished loading data with PyArrow engine", style="bold green")

selected_columns = (
    driver_navigation
    + driver_behavior
    + vehicle_attributes
    + driver_attributes
    + gis_attributes
    + record_day
    + target_mro
)

data = data[selected_columns]
data["purchase_time"] = (
    data["purchase_yr_nbr"].astype(int).astype(str)
    + "_"
    + data["purchase_mth_nbr"].astype(int).astype(str)
)
data = data[data["record_days"] >= 120]
data = data.drop(["purchase_yr_nbr", "purchase_mth_nbr"], axis=1)


weekly_level_mean = data.groupby(["id", "yr_nbr", "week_nbr"]).agg(
    {
        "mth_nbr": "first",
        # if exists mro = 1, then mro = 1
        "mro": "max",
        "hard_braking": "mean",
        "hard_acceleration": "mean",
        "speeding_sum": "mean",
        "day_mileage": "mean",
        "est_hh_incm_prmr_cd": "first",
        "purchaser_age_at_tm_of_purch": "first",
        "input_indiv_gndr_prmr_cd": "first",
        "gmqualty_model": "first",
        "umf_xref_finc_gbl_trim": "first",
        "engn_size": "first",
        "purchase_time": "first",
        "tavg": "mean",
        "record_days": "first",
        "random_avg_traffic": "mean",
    }
)

weekly_level_mean.reset_index(inplace=True)


# set the length of historical time 4, 8, 16
time_window = 4
for i in range(1, time_window + 1):
    weekly_level_mean[f"hard_braking_{i}"] = weekly_level_mean.groupby("id")[
        "hard_braking"
    ].transform(lambda x: x.shift(i))
    weekly_level_mean[f"hard_acceleration_{i}"] = weekly_level_mean.groupby("id")[
        "hard_acceleration"
    ].transform(lambda x: x.shift(i))
    weekly_level_mean[f"speeding_sum_{i}"] = weekly_level_mean.groupby("id")[
        "speeding_sum"
    ].transform(lambda x: x.shift(i))
    weekly_level_mean[f"day_mileage_{i}"] = weekly_level_mean.groupby("id")[
        "day_mileage"
    ].transform(lambda x: x.shift(i))
    weekly_level_mean[f"tavg_{i}"] = weekly_level_mean.groupby("id")["tavg"].transform(
        lambda x: x.shift(i)
    )
    weekly_level_mean[f"random_avg_traffic_{i}"] = weekly_level_mean.groupby("id")[
        "random_avg_traffic"
    ].transform(lambda x: x.shift(i))

weekly_level_mean["mro_prev"] = weekly_level_mean.groupby("id")["mro"].transform(
    lambda x: x.shift(1)
)
console.print(
    "Finished creating time window features",
    style="bold green",
)

column_full = weekly_level_mean.columns.values.tolist()
features_remove = [
    "id",
    "yr_nbr",
    "week_nbr",
    "mth_nbr",
    "mro",
    "record_days",
    "gmqualty_model",
    "hard_braking",
    "hard_acceleration",
    "speeding_sum",
    "day_mileage",
    "tavg",
    "random_avg_traffic",
]
features = [item for item in column_full if item not in features_remove]


unique_ids = weekly_level_mean["id"].unique()

train_ids, val_ids = train_test_split(unique_ids, test_size=0.1, random_state=42)

train_data = weekly_level_mean[weekly_level_mean["id"].isin(train_ids)]
val_data = weekly_level_mean[weekly_level_mean["id"].isin(val_ids)]


X_train = train_data[features]
y_train = train_data[target_mro]
X_valid = val_data[features]
y_valid = val_data[target_mro]


# make sure all object type columns are converted to category
for col in X_train.select_dtypes(include=["object"]).columns:
    X_train[col] = X_train[col].astype("category")
    X_valid[col] = X_valid[col].astype("category")
console.print(
    "Finished converting object type columns to category",
    style="bold green",
)


def objective(trial):
    param = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        # "max_depth": -1,
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "device_type": "gpu",
        # "gpu_platform_id": 0,
        # "gpu_device_id": 2,
        "verbose": -1,
    }

    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_valid = lgb.Dataset(X_valid, label=y_valid)

    model = lgb.train(
        param,
        lgb_train,
        valid_sets=[lgb_valid],
        callbacks=[lgb.early_stopping(stopping_rounds=10)],
        num_boost_round=1000,
    )

    # preds = model.predict(X_valid)
    # auc = roc_auc_score(y_valid, preds)
    y_pred = model.predict(X_valid)
    metrics = evaluate_model(y_valid, y_pred)
    console.print(Markdown("---"))
    console.print("The following metrics are calculated based on the best threshold")
    console.print("Best Threshold:", metrics["Best Threshold"])
    console.print("Precision:", metrics["precision"])
    console.print("Recall:", metrics["recall"])
    console.print("F1 Score:", metrics["f1"])
    console.print("Accuracy:", metrics["accuracy"])
    console.print("AUC:", metrics["auc"])
    console.print("Average Precision:", metrics["average_precision"])
    console.print("Confusion Matrix:\n", metrics["confusion_matrix"])

    auc = metrics["auc"]
    return auc


study = optuna.create_study(
    direction="maximize",
    study_name="mro_lgbm_tuning_4",
    storage="sqlite:///mro.db",
)
study.optimize(objective, n_trials=1000)
