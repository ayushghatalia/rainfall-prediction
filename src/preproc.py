import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Code to do a year block based data split into train test and validation
# Here you have to select the entire year by random rather than a pure random split of data
# 70% of the data (years) will be used for training, 10% for validation and 20% for testing
# Eg if we have data for 10 years, 7 years for training (random, not in order), 2 for testing and 1 for validation
# You can either use random year generator as its straightforward in our case. Else you may use random shuffle of years and then split the years
def data_split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=(2/3), random_state=42)

    return X_train, X_test, X_val, y_train, y_test, y_val


def normalise_with_standard_scaler(data): # dimensions of input array should be num_images, features, img_height, img_width
    temp = np.moveaxis(data, 1, -1)
    temp_compressed = temp.reshape(-1, data.shape[1])

    scaler = StandardScaler()
    scaler.fit(temp_compressed)
    print(scaler.mean_)
    print(scaler.scale_)
    temp_compressed = scaler.transform(temp_compressed)

    data_normalised = temp_compressed.reshape(temp.shape)
    data_normalised = np.moveaxis(data_normalised, 3, 1)
    return data_normalised


def prep_lstm_dataset(dataX, datay, time_step):
    'Create LSTM dataset from the given data'
    dew = dataX[:, 0, :, :].reshape(dataX.shape[0]//122, -1, 122)
    temp = dataX[:, 1, :, :].reshape(dataX.shape[0]//122, -1, 122)
    rad = dataX[:, 2, :, :].reshape(dataX.shape[0]//122, -1, 122)
    u_wind = dataX[:, 3, :, :].reshape(dataX.shape[0]//122, -1, 122)
    v_wind = dataX[:, 4, :, :].reshape(dataX.shape[0]//122, -1, 122)
    press = dataX[:, 5, :, :].reshape(dataX.shape[0]//122, -1, 122)
    rain = datay[:, 0, :, :].reshape(datay.shape[0]//122, -1, 122)
    print(dew.shape)
    print(rain.shape)

    X, y = [], []
    for i in range(rain.shape[0]):
        for j in range(rain.shape[2]-time_step):
            feature = [dew[i, :, j:j+time_step], temp[i, :, j:j+time_step], rad[i, :, j:j+time_step], u_wind[i, :, j:j+time_step], v_wind[i, :, j:j+time_step], press[i, :, j:j+time_step]]
            target = [rain[i, :, j+1:j+time_step+1]]
            X.append(feature)
            y.append(target)

    X = torch.tensor(X)
    print(X.shape)
    X = X.swapaxes(1,3)
    X = X.swapaxes(1,2)
    X = X.reshape(X.shape[0]*X.shape[1], X.shape[2], X.shape[3])
    y = torch.tensor(y)
    print(y.shape)
    y = y.swapaxes(1,3)
    y = y.swapaxes(1,2)
    y = y.reshape(y.shape[0]*y.shape[1], y.shape[2], y.shape[3])

    return X, y

def prep_unet_dataset(data):
    'Create U-Net dataset from the given data'
    inp = data[:, :6, :, :]
    inp = inp.reshape(30, inp.shape[0]//30, inp.shape[1], inp.shape[2], inp.shape[3])
    out = data[:, 6:, :, :]
    out = out.reshape(30, out.shape[0]//30, out.shape[1], out.shape[2], out.shape[3])
    train_in, test_in, val_in, train_out, test_out, val_out = data_split(inp, out)

    train_in = train_in.reshape(-1, inp.shape[2], inp.shape[3], inp.shape[4])
    test_in = test_in.reshape(-1, inp.shape[2], inp.shape[3], inp.shape[4])
    val_in = val_in.reshape(-1, inp.shape[2], inp.shape[3], inp.shape[4])
    train_out = train_out.reshape(-1, out.shape[2], out.shape[3], out.shape[4])
    test_out = test_out.reshape(-1, out.shape[2], out.shape[3], out.shape[4])
    val_out = val_out.reshape(-1, out.shape[2], out.shape[3], out.shape[4])

    return train_in, test_in, val_in, train_out, test_out, val_out

class RainfallDataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, in_params, rain):
            'Initialization'
            self.in_params = in_params
            self.rain = rain

    def __len__(self):
            'Denotes the total number of samples'
            return len(self.in_params)

    def __getitem__(self, idx):
            'Generates one sample of data'
            X = self.in_params[idx]
            y = self.rain[idx]
            return X, y