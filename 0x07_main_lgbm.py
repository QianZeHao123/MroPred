from model import preprocess_data_lgbm as preprocess_data
import os
import pandas as pd
import ray
from ray.train import CheckpointConfig, RunConfig, ScalingConfig
from ray.train.lightgbm import LightGBMTrainer
from utils import create_train_test_group
import lightgbm as lgb

"""
The following part is the important parameter for the LightGBM training
"""
csv_file_name = "./Data/mro_daily_clean.csv"
target_mro = ["mro"]

maintain_repair_mro = "full"

add_mro_prev = True
add_purchase_time = True
add_driver_behavior = True
agg_weeks = 1
agg_fun = ["mean", "sum", "max", "min", "std", "skew"]
# time window could be 4, 8, 12
time_window = 8

# ------------------------------------------
# LightGBM Parameters
metric: list = ["binary_logloss", "binary_error", "auc", "average_precision"]
learning_rate: float = 0.05
num_leaves: int = 64
max_depth: int = 8
is_unbalance: bool = True
# boosting could be "gbdt", "rf" (random forest) and "dart"
boosting: str = "gbdt"


# ------------------------------------------
# data record folder
data_lgbm_file_name = f"data_lgbm_db{int(add_driver_behavior)}_mp{int(add_mro_prev)}_pt{int(add_purchase_time)}_aw{agg_weeks}_tw{time_window}.gzip"
data_lgbm_path = os.path.join("./Data", data_lgbm_file_name)
data_lgbm_path = os.path.abspath(data_lgbm_path)

# ------------------------------------------
# model record folder
model_name = f"model_lgbm_{boosting}_db{int(add_driver_behavior)}_mp{int(add_mro_prev)}_pt{int(add_purchase_time)}_aw{agg_weeks}_tw{time_window}.txt"
model_output_dir = "./output/lgbm"
os.makedirs(model_output_dir, exist_ok=True)
model_path = os.path.join(model_output_dir, model_name)


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
        agg_weeks=agg_weeks,
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
ray_train_dataset = ray.data.from_pandas(train_dataset)
ray_valid_dataset = ray.data.from_pandas(valid_dataset)

# ------------------------------------------
# Configure checkpointing to save progress during training
run_config = RunConfig(
    checkpoint_config=CheckpointConfig(
        # Checkpoint every 10 iterations.
        checkpoint_frequency=1,
        # Only keep the latest checkpoint and delete the others.
        num_to_keep=50,
    )
)
# Set up the XGBoost trainer with the specified configuration
trainer = LightGBMTrainer(
    # see "How to scale out training?" for more details
    scaling_config=ScalingConfig(
        # Number of workers to use for data parallelism.
        num_workers=num_workers,
        # Whether to use GPU acceleration. Set to True to schedule GPU workers.
        use_gpu=False,
        # resources_per_worker={"CPU": 2, "GPU": 1},
    ),
    label_column="target_mro",
    num_boost_round=num_boost_round,
    # XGBoost specific params (see the `xgboost.train` API reference)
    params={
        "objective": "binary",
        "metric": metric,
        "learning_rate": learning_rate,
        "num_leaves": num_leaves,
        "max_depth": max_depth,
        "is_unbalance": is_unbalance,
        "boosting": boosting,
        # "early_stopping_round": early_stopping_round,
        # "device_type": "gpu",
    },

    datasets={"train": ray_train_dataset, "valid": ray_valid_dataset},
    # store the preprocessor in the checkpoint for inference later
    run_config=run_config,
)
# ------------------------------------------
# start training
result = trainer.fit()


# ------------------------------------------
# save the model
booster = trainer.get_model(result.checkpoint)
booster.save_model(model_path)
