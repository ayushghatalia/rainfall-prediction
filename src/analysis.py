import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from unet import UNet
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def plot_loss(train_losses, val_losses, lr, patience, factor, MAX_EPOCH):
    plt.plot(range(0, len(train_losses)), train_losses[0:], color='b', label='Training')
    plt.plot(range(0, len(val_losses)), val_losses[0:], color='g', label='Validation')
    plt.title('Epochs: {}, LR: {}, Patience: {}, Factor: {}'.format(MAX_EPOCH, lr, patience, factor))
    plt.suptitle('Optimizer: RProp, Dropout: No')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def plot_hist(train_losses, val_losses):
    unnorm = np.load('/content/drive/MyDrive/PS1/Aditya/PS1/Numpy/im_np_combined.npy')
    rainfall = unnorm[:,6,:,:]
    rain_new = rainfall.flatten()
    scaler = StandardScaler()
    scaler.fit(rain_new.reshape(-1,1))
    std_dev = scaler.scale_
    train_loss_trf = np.sqrt(np.array(train_losses).reshape(-1,1))*std_dev
    val_loss_trf = np.sqrt(np.array(val_losses).reshape(-1,1))*std_dev
    train_loss_trf_mm = train_loss_trf*1000
    val_loss_trf_mm = val_loss_trf*1000

    plt.hist(val_loss_trf_mm, bins = 25)
    plt.xlabel('Loss(mm)')
    plt.ylabel('Frequency')
    plt.locator_params(axis='x', nbins = 10)
    plt.locator_params(axis='y', nbins = 15)
    plt.suptitle('Histogram of Losses')
    plt.title('Min Loss = {}'.format(np.amin(val_loss_trf_mm)))
    plt.show()

def model_predict(model, test_data):
    model.eval()
    preds = torch.from_numpy(np.array([])).to(device)
    with torch.no_grad():
        for batch, data in enumerate(test_data):
            X, y = data
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            preds = torch.cat((preds, pred), 0)
    return preds

def plot_actual_rain(rainfall, scaler):
    rainfall_unnorm = (rainfall.squeeze().cpu().numpy())*scaler.scale_ + scaler.mean_
    plt.title('Actual Rainfall')
    plt.imshow(np.mean(rainfall_unnorm, axis=0))
    plt.colorbar()
    plt.show()

def plot_pixelwise_pred(preds, scaler, test_dataloader):
    model = UNet().to(device)
    model.load_state_dict(torch.load("/content/drive/MyDrive/PS1/Aditya/PS1/Models/model.pth"))
    preds = model_predict(model, test_dataloader)
    preds = preds.reshape(-1, 8192, 4, 1) #708, 8192, 4, 1
    preds = preds.swapaxes(1,2)
    preds = preds.swapaxes(1,3)
    preds = preds.reshape(708, 1, 64, 128, 4)[:, :, :, :, -1]
    day_pred = np.squeeze(preds.cpu().numpy())*scaler.scale_ + scaler.mean_
    plt.title('Predictions')
    plt.imshow(np.mean(day_pred, axis=0), cmap='viridis')
    plt.colorbar()
    plt.show()

def plot_pixelwise_RMSE(targets, test_dataloader):
    model = UNet().to(device)
    model.load_state_dict(torch.load("/content/drive/MyDrive/PS1/Aditya/PS1/Models/model.pth"))
    preds = model_predict(model, test_dataloader)
    preds = preds.reshape(-1, 8192, 4, 1) #708, 8192, 4, 1
    preds = preds.swapaxes(1,2)
    preds = preds.swapaxes(1,3)
    preds = preds.reshape(708, 1, 64, 128, 4)[:, :, :, :, -1]
    day_pred = preds.squeeze().cpu().numpy()
    targets = targets.squeeze().cpu().numpy()
    print(day_pred.shape)
    print(targets.shape)
    loss = np.sqrt(np.mean((day_pred-targets)**2, axis=0))
    print(loss.shape)
    plt.title('RMSE Loss')
    plt.imshow(loss, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.show()

def plot_nse(preds, targets, scaler):
    day_pred = np.squeeze(preds.cpu().numpy())*scaler.scale_ + scaler.mean_
    targets_unnorm = (targets.squeeze().cpu().numpy())*scaler.scale_ + scaler.mean_
    nse = (1-(np.sum((targets_unnorm-day_pred)**2, axis=0)/np.sum((targets_unnorm-np.mean(targets_unnorm))**2, axis=0)))
    plt.title('Nash Sutcliffe Efficiency')
    plt.imshow(nse, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.show()

def plot_r2(test_dataloader, targets, scaler):
    model = UNet().to(device)
    model.load_state_dict(torch.load("/content/drive/MyDrive/PS1/Aditya/PS1/Models/model.pth"))
    preds = model_predict(model, test_dataloader)
    preds = (preds.cpu().numpy().flatten())*scaler.scale_ + scaler.mean_
    targets_unnorm = targets.flatten().cpu().numpy()*scaler.scale_ + scaler.mean_

    # Calculate R-squared
    r2 = r2_score(targets_unnorm, preds)

    # Plot the R-squared value
    plt.figure()
    plt.scatter(targets_unnorm, preds)
    plt.plot([targets_unnorm.min(), targets_unnorm.max()], [targets_unnorm.min(), targets_unnorm.max()], 'k--', lw=2)
    plt.xlabel('Actual Labels')
    plt.ylabel('Predicted Labels')
    plt.title('R-squared: {:.4f}'.format(r2))
    plt.show()