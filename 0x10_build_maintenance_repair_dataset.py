from model import preprocess_data_lgbm as preprocess_data
from utils import create_train_test_group
import os


maintenance_repair_folder = "./Data/maintenance_repair_comparison"
os.makedirs(maintenance_repair_folder, exist_ok=True)


csv_file_name = "./Data/mro_daily_clean.csv"
target_mro = ["mro"]
add_mro_prev = True
add_purchase_time = True
agg_fun = ["mean", "sum", "max", "min", "std", "skew"]
time_window = 8
agg_scale = "weekly"


benchmark_data_lgbm_file_maintenance_path = os.path.join(
    maintenance_repair_folder, "benchmark_data_maintenance.gzip"
)

benchmark_data_lgbm_file_repair_path = os.path.join(
    maintenance_repair_folder, "benchmark_data_repair.gzip"
)

our_model_data_lgbm_file_maintenance_path = os.path.join(
    maintenance_repair_folder, "our_model_data_maintenance.gzip"
)

our_model_data_lgbm_file_repair_path = os.path.join(
    maintenance_repair_folder, "our_model_data_repair.gzip"
)


# -------------------------------------------------
# benchmark data maintenance
benchmark_data_maintenance = preprocess_data(
    file_name=csv_file_name,
    target_mro=target_mro,
    maintain_repair_mro="maintenance",
    add_mro_prev=add_mro_prev,
    add_purchase_time=add_purchase_time,
    add_driver_behavior=False,
    agg_scale=agg_scale,
    agg_fun=agg_fun,
    time_window=time_window,
)

benchmark_data_lgbm_maintenance = create_train_test_group(
    data=benchmark_data_maintenance,
    sample_frac=1.0,
    test_size=0.1,
    valid_size=0.1,
    random_state=42,
)

benchmark_data_lgbm_maintenance.to_parquet(
    benchmark_data_lgbm_file_maintenance_path, compression="gzip", engine="pyarrow"
)
# -------------------------------------------------
# benchmark_data_repair
benchmark_data_repair = preprocess_data(
    file_name=csv_file_name,
    target_mro=target_mro,
    maintain_repair_mro="repair",
    add_mro_prev=add_mro_prev,
    add_purchase_time=add_purchase_time,
    add_driver_behavior=False,
    agg_scale=agg_scale,
    agg_fun=agg_fun,
    time_window=time_window,
)

benchmark_data_lgbm_repair = create_train_test_group(
    data=benchmark_data_repair,
    sample_frac=1.0,
    test_size=0.1,
    valid_size=0.1,
    random_state=42,
)

benchmark_data_lgbm_repair.to_parquet(
    benchmark_data_lgbm_file_repair_path, compression="gzip", engine="pyarrow"
)
# -------------------------------------------------
# our_model_data_maintenance
our_model_data_maintenance = preprocess_data(
    file_name=csv_file_name,
    target_mro=target_mro,
    maintain_repair_mro="maintenance",
    add_mro_prev=add_mro_prev,
    add_purchase_time=add_purchase_time,
    add_driver_behavior=True,
    agg_scale=agg_scale,
    agg_fun=agg_fun,
    time_window=time_window,
)

our_model_data_lgbm_maintenance = create_train_test_group(
    data=our_model_data_maintenance,
    sample_frac=1.0,
    test_size=0.1,
    valid_size=0.1,
    random_state=42,
)

our_model_data_lgbm_maintenance.to_parquet(
    our_model_data_lgbm_file_maintenance_path, compression="gzip", engine="pyarrow"
)
# -------------------------------------------------
# our_model_data_repair
our_model_data_repair = preprocess_data(
    file_name=csv_file_name,
    target_mro=target_mro,
    maintain_repair_mro="repair",
    add_mro_prev=add_mro_prev,
    add_purchase_time=add_purchase_time,
    add_driver_behavior=True,
    agg_scale=agg_scale,
    agg_fun=agg_fun,
    time_window=time_window,
)

our_model_data_lgbm_repair = create_train_test_group(
    data=our_model_data_repair,
    sample_frac=1.0,
    test_size=0.1,
    valid_size=0.1,
    random_state=42,
)

our_model_data_lgbm_repair.to_parquet(
    our_model_data_lgbm_file_repair_path, compression="gzip", engine="pyarrow"
)
