import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-darkgrid')

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import MinMaxScaler
from models.ANN import ANN
from tqdm.auto import trange


class TimeSeriesDataset(torch.utils.data.Dataset):
    '''
    TODO(영준)
    멀티 채널이 입력으로 들어갈 떄, y가 목표 컬럼만 나올 수 있도록
    '''
    def __init__(self, ts:np.array, lookback_size:int, forecast_size:int):
        self.lookback_size = lookback_size
        self.forecast_size = forecast_size
        self.data = ts

    def __len__(self):
        return len(self.data) - self.lookback_size - self.forecast_size + 1

    def __getitem__(self, i):
        idx = (i+self.lookback_size)
        look_back = self.data[i:idx]
        forecast = self.data[idx:idx+self.forecast_size]

        return look_back, forecast

def mape(y_pred, y_true):
    return (np.abs(y_pred - y_true)/y_true).mean() * 100

def mae(y_pred, y_true):
    return np.abs(y_pred - y_true).mean()


def main(cfg):
    ################ 1. Dataset Load  ################
    dataset_setting = cfg.get("dataset_setting")
    main_csv = dataset_setting.get("main_csv")
    time_axis = dataset_setting.get("time_axis")
    target = dataset_setting.get("target")
    
    data = pd.read_csv(main_csv)
    # data_only_pm25 = pd.DataFrame(data.loc[:, ["PM-2.5", "일시"]])
    data[time_axis] = pd.to_datetime(data[time_axis])
    data.index = data[time_axis]
    del data[time_axis]
    data = data.iloc[50:, :] # TODO: 나중에 없애고 파일 자체를 2018.4 ~ 부터 저장되어있도록 바꾸기
    
    m_data = data.copy()
    m_data = m_data.dropna()
    target_column = data.columns.get_loc(target)
    ##################################################
    
    ############### 2. Preprocessing  ################
    # hyperparameter
    window_params = cfg.get("window_params")
    tst_size = window_params.get("tst_size")
    lookback_size = window_params.get("lookback_size")
    forecast_size = window_params.get("forecast_size")
    
    train_params = cfg.get("train_params")
    epochs = train_params.get("epochs")
    data_loader_params = train_params.get("data_loader_params")

    # 결측치 처리는 완전히 되었다고 가정
    
    # scaling
    scaler = MinMaxScaler()
    trn_scaled = scaler.fit_transform(m_data[:-tst_size].to_numpy(dtype=np.float32))
    tst_scaled = scaler.transform(m_data[-tst_size-lookback_size:].to_numpy(dtype=np.float32))

    trn_ds = TimeSeriesDataset(trn_scaled, lookback_size, forecast_size)
    tst_ds = TimeSeriesDataset(tst_scaled, lookback_size, forecast_size)

    trn_dl = torch.utils.data.DataLoader(trn_ds, **data_loader_params)
    tst_dl = torch.utils.data.DataLoader(tst_ds, batch_size=tst_size, shuffle=False)
    ##################################################
    
    ########## 3. Train Hyperparams setting ##########
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # hyperparameter
    c_in = 22 #TODO: 자동으로 설정될 수 있도록
    ''' 
    TODO(영준): 자동으로 설정될 수 있도록
    만약에 입력 포멧이 단일채널이면(c_in=1) shape = (batch_size, lookback_size)
    멀티채널이면(c_in>1) shape = (batch_size, c_in, lookback_size)
    if ...:
        ddd
    else:
        ddd
    '''
    
    # model stting
    model = cfg.get("model")
    model_params = cfg.get("ann_model_params")
    model_params["d_in"] = lookback_size
    model_params["d_out"] = forecast_size
    model_params["c_in"] = c_in
    model = model(**model_params)
    model.to(device)
    
    # optimzer / loss setting
    Optim = train_params.get('optim')
    optim_params = train_params.get('optim_params')
    optim = Optim(model.parameters(), **optim_params)
    
    loss_func = train_params.get("loss")
    
    pbar = trange(epochs)
    ##################################################
    
    ################### 4. Train #####################
    for i in pbar:
        model.train()
        trn_loss = .0
        for x, y in trn_dl:
            '''
            TODO(영준)
            멀티 채널이 입력으로 들어갈 떄, y가 목표 컬럼만 나올 수 있도록 Dataset 수정
            Before:
                x, y = x.flatten(1).to(device), y[:,:,target_column].to(device)
            After:
                x, y = x.flatten(1).to(device), y.to(device)
            '''
            x, y = x.flatten(1).to(device), y[:,:,target_column].to(device)   # (32, 18), (32, 4)
            p = model(x)
            optim.zero_grad()
            loss = loss_func(p, y)
            loss.backward()
            optim.step()
            trn_loss += loss.item()*len(y)
        trn_loss = trn_loss/len(trn_ds)

        model.eval()
        with torch.inference_mode():
            x, y = next(iter(tst_dl))
            x, y = x.flatten(1).to(device), y[:,:,target_column].to(device)
            p = model(x)
            tst_loss = F.mse_loss(p,y)
        pbar.set_postfix({'loss':trn_loss, 'tst_loss':tst_loss.item()})
    ##################################################
    
    ################# 5. Evaluation ##################
    model.eval()
    with torch.inference_mode():
        x, y = next(iter(tst_dl))
        x, y = x.flatten(1).to(device), y[:,:,target_column].to(device)
        p = model(x)

        y = y.cpu()/scaler.scale_[0] + scaler.min_[0]
        p = p.cpu()/scaler.scale_[0] + scaler.min_[0]

        y = np.concatenate([y[:,0], y[-1,1:]])
        p = np.concatenate([p[:,0], p[-1,1:]])
    ##################################################
    
    #################### 6. Plot #####################
    plt.title(f"Neural Network, MAPE:{mape(p,y):.4f}, MAE:{mae(p,y):.4f}")
    plt.plot(range(tst_size), y, label="True")
    plt.plot(range(tst_size), p, label="Prediction")
    plt.legend()
    plt.savefig("figs/graph.jpg", format="jpeg")
    ##################################################
    

def get_args_parser(add_help=True):
  import argparse
  
  parser = argparse.ArgumentParser(description="Pytorch K-fold Cross Validation", add_help=add_help)
  parser.add_argument("-c", "--config", default="./config.py", type=str, help="configuration file")

  return parser

if __name__ == "__main__":
  args = get_args_parser().parse_args()
  exec(open(args.config).read())
  
  main(config)