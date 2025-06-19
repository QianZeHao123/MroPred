import pandas as pd
import numpy as np


def preprocess_data_lgbm(
    file_name: str,
    target_mro: list,
    maintain_repair_mro: str,
    add_mro_prev: bool,
    add_purchase_time: bool,
    add_driver_behavior: bool,
    agg_scale: str,
    agg_fun: list,
    time_window: int = 8,
):
    """
    file_name: path to the csv file
    target_mro: a list = ['mro'] or items in mro_detail
    maintain_repair_mro: a string representing the maintenance or repair or all the mro
    add_mro_prev: if this is True -> add a new column pre_mro = mro(t-1)
    add_purchase_time: if this is True -> add a column Year_upper / Year_lower
    add_driver_behavior: if this is True -> add columns for driver behaviors
    agg_scale: this is a string either 'weekly' or 'monthly'
    agg_fun: a list of aggregate function
    time_window: number of previous time periods
    """
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
        data["repair"] = np.where((data["mro"] == 1) & (data["service_days"] > 3), 1, 0)
        data["target_mro"] = data["repair"]
        print("Target MRO is Repair.")
    else:
        print("No need to know the maintenance or repair.")
        print("Use the Target MRO.")
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
        # + mro_prev
        + ["target_mro"]
    ]

    agg_rules = {
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

        driver_behavior_list = [
            "hard_braking",
            "hard_acceleration",
            "speeding_sum",
            "day_mileage",
        ]
    else:
        driver_behavior_list = []
    print("Add behavior of Driver:", add_driver_behavior)
    # --------------------------------------------------
    # if add_mro_prev:
    #     agg_rules["mro_prev"] = "max"
    if add_purchase_time:
        agg_rules["purchase_time"] = "first"
    # --------------------------------------------------
    # group aggregated by 'week' or 'month'
    if agg_scale == "weekly":
        agg_scale_column = "week_nbr"
    elif agg_scale == "monthly":
        agg_scale_column = "mth_nbr"
    else:
        print("Not choose the aggregation scale, use weekly as default.")
        agg_scale_column = "week_nbr"

    data = data.groupby(["id", "yr_nbr", agg_scale_column]).agg(agg_rules)
    # --------------------------------------------------
    data.reset_index(inplace=True)

    def flatten_columns(df: pd.DataFrame):
        def clean_col(col):
            if isinstance(col, tuple):
                col_name, agg_func = col
                agg_func = agg_func.strip()
                # if col_name in (["target_mro"] + mro_prev) and agg_func == "max":
                #     return col_name
                if col_name in (["target_mro"]) and agg_func == "max":
                    return col_name
                if agg_func in ("first", ""):
                    return col_name
                return f"{col_name}_{agg_func}"
            else:
                return col

        df.columns = [clean_col(col) for col in df.columns]
        return df

    data = flatten_columns(data)

    time_series_features = ["tavg", "random_avg_traffic"] + driver_behavior_list
    time_series_columns = [
        f"{feature}_{agg}" for feature in time_series_features for agg in agg_fun
    ]
    # --------------------------------------------------
    # build the time series
    shift_cols = data[["id"] + time_series_columns]
    for i in range(1, time_window + 1):
        shifted = (
            shift_cols.groupby("id")[time_series_columns].shift(i).add_suffix(f"_{i}")
        )
        data = pd.concat([data, shifted], axis=1)

    # add previous mro
    if add_mro_prev:
        data["mro_prev"] = data.groupby("id")["target_mro"].transform(
            lambda x: x.shift(1)
        )
    print("Add Previous MRO:", add_mro_prev)

    data.fillna(0, inplace=True)

    data = data.drop(["yr_nbr", agg_scale_column], axis=1)
    data = data.drop(time_series_columns, axis=1)

    for col in data.select_dtypes(include="object").columns:
        data[col] = data[col].astype("category")
    return data
