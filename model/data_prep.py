import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np

def preprocess_data(
    file_name: str,
    target_mro: list,
    maintain_repair_mro: str,
    add_mro_prev: bool,
    add_purchase_time: bool,
    add_driver_behavior: bool,
    agg_weeks: int,
    agg_fun: list,
):
    data = pd.read_csv(file_name, index_col=0, engine="pyarrow")
    print("Load the dataset", file_name, "successfully.")
    # --------------------------------------------------
    # target mro selection
    mro_detail = [
        "battery_dummy",
        "brake_dummy",
        "tire_dummy",
        "lof_dummy",
        "wiper_dummy",
        "filter_dummy",
        "others",
    ]
    if target_mro == ["mro"]:
        data["target_mro"] = data["mro"]
    elif isinstance(target_mro, list) and all(col in mro_detail for col in target_mro):
        data["target_mro"] = data[target_mro].max(axis=1)
    else:
        print("Target MRO is defined with error")
        print("Use the mro as default mro")
        target_mro = ["mro"]
        data["target_mro"] = data["mro"]
    print("The MRO choosen is:", target_mro)
    # --------------------------------------------------
    # choose full mro, maintanance or repair
    if maintain_repair_mro == "maintenance":
        data["maintenance"] = np.where(
            (data["mro"] == 1) & (data["service_days"] <= 3), 1, 0
        )
        data["target_mro"] = data["maintenance"]
        print("Target MRO is maintenance.")
    elif maintain_repair_mro == "repair":
        data["repair"] = np.where(
            (data["mro"] == 1) & (data["service_days"] > 3), 1, 0
        )
        data["target_mro"] = data["repair"]
        print("Target MRO is Repair.")
    else:
        print("No need to know the maintenance or repair.")
        print("Use the Target MRO.")

    # add previous mro
    if add_mro_prev:
        data.sort_values(by=["id", "yr_nbr", "week_nbr"], inplace=True)
        data["mro_prev"] = data.groupby("id")["mro"].shift(1)
        mro_prev = ["mro_prev"]
    else:
        mro_prev = []
    print("Add Previous MRO:", add_mro_prev)
    # --------------------------------------------------
    # dealing with purchase time
    if add_purchase_time:
        data["purchase_month"] = data["purchase_mth_nbr"].astype(int)
        # devide into 2 bins: 1-6 is the first half, 7-12 is the second half
        data["purchase_half_year"] = pd.cut(
            data["purchase_month"],
            bins=[0, 6, 12],
            labels=["first_half", "second_half"],
        )

        data["purchase_time"] = (
            data["purchase_yr_nbr"].astype(int).astype(str)
            + "_"
            + data["purchase_half_year"].astype(str)
        )

        purchase_time = ["purchase_time"]
    else:
        purchase_time = []
    print("Add Purchase Time:", add_purchase_time)
    # --------------------------------------------------
    # weekly aggregation
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
    ] + purchase_time

    driver_navigation = [
        "id",
        "yr_nbr",
        "mth_nbr",
        "week_nbr",
    ]

    data = data[
        driver_navigation
        + continuous_variable
        + category_variable
        + mro_prev
        + ["target_mro"]
    ]

    agg_rules = {
        # "mth_nbr": "first",
        "target_mro": "max",
        "est_hh_incm_prmr_cd": "first",
        "purchaser_age_at_tm_of_purch": "first",
        "input_indiv_gndr_prmr_cd": "first",
        "gmqualty_model": "first",
        "umf_xref_finc_gbl_trim": "first",
        "engn_size": "first",
        "tavg": agg_fun,
        "random_avg_traffic": agg_fun,
    }
    # --------------------------------------------------
    if add_driver_behavior:
        agg_rules["hard_braking"] = agg_fun
        agg_rules["hard_acceleration"] = agg_fun
        agg_rules["speeding_sum"] = agg_fun
        agg_rules["day_mileage"] = agg_fun
    print("Add behavior of Driver:", add_driver_behavior)
    # --------------------------------------------------
    if add_mro_prev:
        agg_rules["mro_prev"] = "max"
    if add_purchase_time:
        agg_rules["purchase_time"] = "first"
    # --------------------------------------------------
    # week aggregate
    data["group_week"] = (data["week_nbr"] - 1) // agg_weeks
    print("Aggregate the data into", agg_weeks, "week.")
    data = data.groupby(["id", "yr_nbr", "group_week"]).agg(agg_rules)
    # --------------------------------------------------
    data.reset_index(inplace=True)

    def flatten_columns(df: pd.DataFrame):
        def clean_col(col):
            if isinstance(col, tuple):
                col_name, agg_func = col
                agg_func = agg_func.strip()
                if col_name in (["target_mro"] + mro_prev) and agg_func == "max":
                    return col_name
                if agg_func in ("first", ""):
                    return col_name
                return f"{col_name}_{agg_func}"
            else:
                return col

        df.columns = [clean_col(col) for col in df.columns]
        return df

    data = flatten_columns(data)
    data.fillna(0, inplace=True)
    data = data.drop(["yr_nbr", "group_week"], axis=1)
    # --------------------------------------------------
    # Standardization
    col_need_std = [
        item
        for item in data.columns.values.tolist()
        if item not in (["target_mro"] + mro_prev + ["id"] + category_variable)
    ]

    col_need_encode = category_variable

    scaler = StandardScaler()
    data[col_need_std] = scaler.fit_transform(data[col_need_std])

    encoder = OneHotEncoder(sparse_output=False)
    encoded_categorical = encoder.fit_transform(data[col_need_encode])

    category_counts = [
        len(encoder.categories_[i]) for i, _ in enumerate(col_need_encode)
    ]

    onehot_feature_names = []
    for col_idx, col in enumerate(col_need_encode):
        num_categories = category_counts[col_idx]
        onehot_feature_names.extend(
            [f"{col}_onehot_{i}" for i in range(num_categories)]
        )

    encoded_df = pd.DataFrame(
        encoded_categorical, index=data.index, columns=onehot_feature_names
    )
    data = pd.concat([data, encoded_df], axis=1)
    data = data.drop(columns=col_need_encode)
    print("Finish the process of data standardization")

    rnn_features = col_need_std + onehot_feature_names + mro_prev
    print("The RNN features are:", rnn_features)

    rnn_target = ["target_mro"]
    print("The RNN target is:", rnn_target)

    return {
        "data": data,
        "rnn_features": rnn_features,
        "rnn_target": rnn_target,
    }