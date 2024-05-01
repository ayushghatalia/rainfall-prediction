from dataset import *
from preproc import *
from lstm import LSTM
from unet import UNet
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Loading the data
im_np = np.load('/content/drive/MyDrive/PS1/Aditya/PS1/Numpy/im_np_combined_normalized.npy')
X = im_np[:, :6, :, :].reshape(30, 122, 6, 64, 128)
y = im_np[:, 6:, :, :].reshape(30, 122, 1, 64, 128)
# X = np.load('/content/drive/MyDrive/PS1/Aditya/PS1/Numpy/X_lstm.npy')
# y = np.load('/content/drive/MyDrive/PS1/Aditya/PS1/Numpy/y_lstm.npy')
X_train, X_test, X_val, y_train, y_test, y_val = data_split(torch.tensor(X), torch.tensor(y))
X_train = X_train.float()
X_test = X_test.float()
X_val = X_val.float()
y_train = y_train.float()
y_test = y_test.float()
y_val = y_val.float()
train_dataloader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
val_dataloader = DataLoader(TensorDataset(X_val, y_val), batch_size=64, shuffle=True)
test_dataloader = DataLoader(TensorDataset(X_test, y_test), batch_size=64, shuffle=True)

#Hyperparameters
MAX_EPOCH = 200
lr = 0.01
factor = 0.5
patience = 5
wt_decay = 1e-5

#Loading the model
model = UNet().to(device) #Change this to LSTM() for LSTM model
criterion = nn.MSELoss().to(device)
optimizer = torch.optim.Rprop(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=factor, patience=patience)

def epoch_train():
    running_train_loss = 0.0
    for batch, data in enumerate(train_dataloader):
        X, y = data
        X = X.to(device)
        y = y.to(device)
        out = model(X)
        loss = criterion(out, y)
        running_train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    avg_train_loss = running_train_loss/(batch+1)

    return avg_train_loss

def epoch_val():
    running_val_loss = 0.0
    curr_pred = []
    with torch.no_grad():
        for batch, data in enumerate(val_dataloader):
            X, y = data
            X = X.to(device)
            y = y.to(device)
            val_out = model(X)
            curr_pred.append(val_out)
            val_loss = criterion(val_out, y)
            running_val_loss += val_loss.item()
    avg_val_loss = running_val_loss/(batch+1)

    return avg_val_loss, curr_pred

train_losses = []
val_losses = []
best_pred = []
min_val_loss = 9999.9
for epoch in tqdm(range(MAX_EPOCH)):
    print('Epoch {}: '.format(epoch+1))

    model.train(True)
    avg_train_loss = epoch_train()

    model.eval()
    avg_val_loss, curr_pred = epoch_val()

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    print('Train loss = {}, Val loss = {}'.format(avg_train_loss, avg_val_loss))

    if(avg_val_loss<min_val_loss):
        min_val_loss=avg_val_loss
        best_pred = curr_pred
        torch.save(model.state_dict(), "/content/drive/MyDrive/PS1/Aditya/PS1/Models/model_final1.pth" )
        torch.save(optimizer.state_dict(), "/content/drive/MyDrive/PS1/Aditya/PS1/Models/opt_final1.pth")
    scheduler.step(avg_val_loss)