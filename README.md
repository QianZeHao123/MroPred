# MroPred
A research project in CityUHK. Mro Prediction with LightGBM.

## Environment Setup

```bash
conda create -n mro python=3.11.11
```

## Export the Tmux Log

```shell
tmux capture-pane -t MRO4month -pS -10000 > ./log/mro_output.log
```

1. data_preprocess.ipynb: add GIS information to the original data, preprocess it, and aggregate into weekly level
2. weekly_filter_new_2.csv: weekly level data, remove brake, tire, filter
3. LSTM.ipynb: use 'weekly_filter_new_2.csv'

in the classification folder:
preprocess_for_classification.ipynb: reshape 'weekly_filter_new_2.csv' -> '8weeks_new.csv'
8weeks_new.csv: dataset used for classification algorithm
2month_new.csv: aggregate data into monthly level
mro_prediction_lightGBM.ipynb: script for LightGBM

in the classification/imbalance folder: paper + code

```shell
tmux new -t MRO_LSTM
```

```shell
tmux a -t MRO_LSTM
```

```shell
source .bashrc
conda activate mro
cd /home/user14/Cyber/MroPred
ls -la
python 6_LSTM_feature_engineering.py
```

```shell
tmux a -t MRO_LSTM_adv
```

```shell
source .bashrc
conda activate mro
cd /home/user14/Cyber/MroPred
ls -la
python main_LSTM_feature_engineering_v3.py
```