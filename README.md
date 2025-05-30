# MroPred
A research project in CityUHK. Mro Prediction with LightGBM, LSTM.

1. data_preprocess.ipynb: add GIS information to the original data, preprocess it, and aggregate into weekly level
2. weekly_filter_new_2.csv: weekly level data, remove brake, tire, filter
3. LSTM.ipynb: use 'weekly_filter_new_2.csv'

in the classification folder:
preprocess_for_classification.ipynb: reshape 'weekly_filter_new_2.csv' -> '8weeks_new.csv'
8weeks_new.csv: dataset used for classification algorithm
2month_new.csv: aggregate data into monthly level
mro_prediction_lightGBM.ipynb: script for LightGBM

in the classification/imbalance folder: paper + code


## Environment Setup

```bash
conda create -n mro python=3.11.11
pip install -r requirements.txt
conda activate mro
```

## Run the code with Tmux

### Run the Batch Train

```shell
tmux new -t <tmux session name>
```

```shell
tmux a -t <tmux session name>
```

```shell
source .bashrc
conda activate mro
cd /home/user14/Cyber/MroPred
python <python script>.py
```

### Export the Tmux Log

```shell
tmux capture-pane -t <tmux session name> -pS -100000 > ./log/<log file name>.log
```

## Show Optuna Result

```shell
optuna-dashboard sqlite:///mro_lstm.db
```
