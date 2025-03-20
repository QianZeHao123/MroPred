# MroPred
A research project in CityUHK. Mro Prediction with LightGBM.

## Environment Setup

```bash
conda create -n mro python=3.11.11
```


data_preprocess.ipynb: add GIS information to the original data, preprocess it, and aggregate into weekly level
weekly_filter_new_2.csv: weekly level data, remove brake, tire, filter
LSTM.ipynb: use 'weekly_filter_new_2.csv'

in the classification folder:
preprocess_for_classification.ipynb: reshape 'weekly_filter_new_2.csv' -> '8weeks_new.csv'
8weeks_new.csv: dataset used for classification algorithm
2month_new.csv: aggregate data into monthly level
mro_prediction_lightGBM.ipynb: script for LightGBM

in the classification/imbalance folder: paper + code