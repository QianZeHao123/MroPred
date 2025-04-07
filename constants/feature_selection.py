driver_behavior = [
    "hard_braking",
    "hard_acceleration",
    "speeding_sum",
    "day_mileage",
]

# Vehicle and driver attributes
vehicle_attributes = [
    "gmqualty_model",
    "umf_xref_finc_gbl_trim",
    "engn_size",
    "purchase_yr_nbr",
    "purchase_mth_nbr",
]

# driver attributes
driver_attributes = [
    "est_hh_incm_prmr_cd",
    "purchaser_age_at_tm_of_purch",
    "input_indiv_gndr_prmr_cd",
]

# driver nevigation
driver_navigation = [
    "id",
    "yr_nbr",
    "mth_nbr",
    "week_nbr",
]

# gis attributes
gis_attributes = [
    "tavg",
    "random_avg_traffic",
]

# record day, use this as a filter to make sure we have at least 16 weeks of records
record_day = ["record_days"]
