import torch
import torch.nn.functional as F
import torchmetrics
from models.ANN import ANN
from models.LSTM import StatefulLSTM, StatelessLSTM
from models.Transformer import PatchTST

config = {
  'dataset_setting':{
    "main_csv": "/home/dataset/complete.csv",
    "time_axis": "일시",
    "target": "PM-2.5"
  },
  "window_params":{
    "tst_size": 200,
    "lookback_size": 9,
    "forecast_size": 4
  },
  
  'model': ANN, # or RandomForestRegressor
  'ann_model_params': {
    'd_hidden': 512,
    'activation': "relu",
    'use_dropout': False,
  },
  
  
  'train_params': {
    'data_loader_params': {
      'batch_size': 32,
      'shuffle': True,
    },
    'loss': F.mse_loss,
    'optim': torch.optim.AdamW,
    'optim_params': {
      'lr': 0.0001,
    },
    'metric': torchmetrics.MeanSquaredError(squared=False),
    'device': 'cuda',
    'epochs': 10,
  },
  
  
   
  'cv_params':{
    'n_split': 5,
  },
  
    
   'preprocess' : {
      "features":  ['lane_count', 'road_rating', 'maximum_speed_limit',
                  'weight_restricted', 'month', 'rough_road_name', 
                  'line_number', 'start_latitude_enc', 'end_latitude_enc','end_turn_restricted','start_turn_restricted', 'weight_restricted_enc',
                  "base_hour", "peak_season", 'multi_linked', 'connect_code', 'peak_hour'],
      "train-csv": "/home/data/train_last.csv",
      "test-csv" : "/home/data/test_last.csv",
      "output-train-feas-csv" : "./data/trn_X.csv",
      "output-test-feas-csv" : "./data/tst_X.csv", 
      "output-train-target-csv" : "./data/trn_y.csv", 
      "output-test-target-csv" : "./data/tst_y.csv", 
      "encoding-columns": ['start_turn_restricted', 'end_turn_restricted'],
      "scale-columns" : [], 
      "target-col" : "target",
      "scaler" : "None"
  },
  'wandb':{
      'use_wandb': False,
      'wandb_runname': "test",
      },
  
  'files': {
    'X_csv': './data/trn_X.csv',
    'y_csv': './data/trn_y.csv',
    'X_test_csv': './data/tst_X.csv',
    'output': './model.pth',
    'output_csv': './results/five_fold.csv',
    'submission_csv': './submission.csv',
  },
}