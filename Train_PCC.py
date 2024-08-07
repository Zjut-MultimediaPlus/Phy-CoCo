import os
import torch.optim
from torch import nn
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import Set_seed
from PCC_Dataset import Dataset
import PCC_Config as config
from PCC_Net import *

def initalize_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.1)
def save_model(model, model_output_dir, epoch):
    save_model_file = os.path.join(model_output_dir, "epoch_{}.pth".format(epoch))
    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)
    torch.save(model, save_model_file)

def train_model(model, loss_func, dataset, optimizer, params, epoch):
    if epoch == 1:
        initalize_weights(model)

    model.train()

    batch_loss = 0
    batch_wind_loss = 0
    batch_rmw_loss = 0

    item = 0
    batch_num = 0

    wind_error = 0
    rmw_error = 0

    for batch, data in enumerate(tqdm(dataset)):

        btemp = data["btemp"]
        btemp = btemp.to(config.device)

        pre = data['pre']
        pre = pre.to(torch.float32)
        pre = pre.to(config.device)
        pre = pre.reshape(pre.shape[0], 1)

        t = data['occur_t']
        t = t.to(torch.float32)
        t = t.to(config.device)
        t = t.reshape(t.shape[0], 1)

        lat = data["lat"]
        lat = lat.reshape(lat.shape[0], 1)
        lat = lat.to(torch.float32)
        lat = lat.to(config.device)

        lon = data["lon"]
        lon = lon.reshape(lon.shape[0], 1)
        lon = lon.to(torch.float32)
        lon = lon.to(config.device)

        wind_label = data["wind"]
        wind_label = wind_label.to(config.device)
        rmw_label = data["RMW"]
        rmw_label = rmw_label.to(config.device)

        optimizer.zero_grad()

        rmw, wind = model(btemp, pre, lat, lon, t)

        print("wind_label={}".format(wind_label))
        print("MSW={}".format(wind))

        print("rmw_label={}".format(rmw_label))
        print("RMW={}".format(rmw))

        wind_loss = loss_func(wind.float(), wind_label.float())
        rmw_loss = loss_func(rmw.float(), rmw_label.float())

        total_loss = params[0] * wind_loss + params[1] * rmw_loss

        total_loss.backward()
        optimizer.step()

        print("Train Epoch = {} MSW Loss = {}".format(epoch, wind_loss.data.item()))
        print("Train Epoch = {} RMW Loss = {}".format(epoch, rmw_loss.data.item()))

        print("Train Epoch = {} Loss = {}".format(epoch, total_loss.data.item()))
        batch_loss += total_loss.data.item()
        batch_wind_loss += wind_loss.data.item()
        batch_rmw_loss += rmw_loss.data.item()

        wind_label_re = wind_label.cpu().detach().numpy() * (170 - 19) + 19
        wind_re = wind.cpu().detach().numpy() * (170 - 19) + 19
        wind_error = wind_error + np.sum(np.abs(wind_re - wind_label_re))

        rmw_label_re = rmw_label.cpu().detach().numpy() * (200 - 5) + 5
        rmw_re = rmw.cpu().detach().numpy() * (200 - 5) + 5
        rmw_error = rmw_error + np.sum(np.abs(rmw_re - rmw_label_re))

        batch_num += 1
        item += len(rmw_re)

    print("Train Epoch = {} mean loss = {} ".format(epoch, batch_loss / batch_num))
    print("Train Epoch = {} mean msw error = {} ".format(epoch, wind_error / item))
    print("Train Epoch = {} mean rmw error = {} ".format(epoch, rmw_error / item))

    return batch_loss / batch_num, wind_error / item, rmw_error / item

def valid_model(model, loss_func, dataset, epoch):

    model.eval()

    batch_loss = 0

    batch_num = 0
    item = 0

    wind_error = 0
    rmw_error = 0

    with torch.no_grad():
        for batch, data in enumerate(tqdm(dataset)):
            btemp = data["btemp"]
            btemp = btemp.to(config.device)

            pre = data['pre']
            pre = pre.to(torch.float32)
            pre = pre.to(config.device)
            pre = pre.reshape(pre.shape[0], 1)

            t = data['occur_t']
            t = t.to(torch.float32)
            t = t.to(config.device)
            t = t.reshape(t.shape[0], 1)

            lat = data["lat"]
            lat = lat.reshape(lat.shape[0], 1)
            lat = lat.to(torch.float32)
            lat = lat.to(config.device)

            lon = data["lon"]
            lon = lon.reshape(lon.shape[0], 1)
            lon = lon.to(torch.float32)
            lon = lon.to(config.device)

            wind_label = data["wind"]
            wind_label = wind_label.to(config.device)

            rmw_label = data["RMW"]
            rmw_label = rmw_label.to(config.device)

            rmw, wind = model(btemp, pre, lat, lon, t)

            print("wind_label={}".format(wind_label))
            print("Intensity={}".format(wind))


            print("rmw_label={}".format(rmw_label))
            print("RMW={}".format(rmw))

            wind_loss = loss_func(wind.float(), wind_label.float())
            rmw_loss = loss_func(rmw.float(), rmw_label.float())

            total_loss = rmw_loss + wind_loss

            print("Valid Epoch = {} MSW Loss = {}".format(epoch, wind_loss.data.item()))
            print("Valid Epoch = {} RMW Loss = {}".format(epoch, rmw_loss.data.item()))

            print("Valid Epoch = {} Loss = {}".format(epoch, total_loss.data.item()))
            batch_loss += total_loss.data.item()

            # 标签和估计结果反归一化后比较误差
            wind_label_re = wind_label.cpu().detach().numpy() * (170 - 19) + 19
            wind_re = wind.cpu().detach().numpy() * (170 - 19) + 19
            wind_error = wind_error + np.sum(np.abs(wind_re - wind_label_re))

            rmw_label_re = rmw_label.cpu().detach().numpy() * (200 - 5) + 5
            rmw_re = rmw.cpu().detach().numpy() * (200 - 5) + 5
            rmw_error = rmw_error + np.sum(np.abs(rmw_re - rmw_label_re))

            batch_num += 1
            item += len(rmw_re)

    print("Valid Epoch = {} mean loss = {} ".format(epoch, batch_loss / batch_num))
    print("Valid Epoch = {} mean msw error = {} ".format(epoch, wind_error / item))
    print("Valid Epoch = {} mean rmw error = {} ".format(epoch, rmw_error / item))

    return batch_loss / batch_num,  wind_error / item, rmw_error / item

def train(model, Estim_Loss, optimizer, params, epochs):
    train_transform = None
    valid_transform = None
    train_dataset = Dataset(config.train_npy_path, train_transform, config.data_format)
    valid_dataset = Dataset(config.valid_npy_path, valid_transform, config.data_format)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size,
        shuffle=True, num_workers=config.num_workers,
    )

    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config.batch_size,
        shuffle=True, num_workers=config.num_workers
    )

    '''
    训练模型
    '''
    start_epoch = 0
    train_loss_array = np.zeros(epochs + 1)
    valid_loss_array = np.zeros(epochs + 1)
    # 强度
    train_wind_error_array = np.zeros(epochs + 1)
    valid_wind_error_array = np.zeros(epochs + 1)
    # 风圈RMW
    train_rmw_error_array = np.zeros(epochs + 1)
    valid_rmw_error_array = np.zeros(epochs + 1)

    for epoch in range(start_epoch + 1, epochs + 1):
        train_epoch_loss, train_wind_error, train_rmw_error = \
            train_model(model, Estim_Loss, train_dataloader, optimizer, params, epoch)
        valid_epoch_loss, valid_wind_error, valid_rmw_error = \
            valid_model(model, Estim_Loss, valid_dataloader, epoch)

        train_loss_array[epoch] = train_epoch_loss
        valid_loss_array[epoch] = valid_epoch_loss

        train_wind_error_array[epoch] = train_wind_error
        valid_wind_error_array[epoch] = valid_wind_error

        train_rmw_error_array[epoch] = train_rmw_error
        valid_rmw_error_array[epoch] = valid_rmw_error

        # 模型保存
        if epoch % config.save_model_iter == 0:
            save_model(model, config.model_output_dir, epoch)

    # 绘制loss
    fig_loss, ax_loss = plt.subplots(figsize=(12, 8))
    train_loss, = plt.plot(np.arange(1, epochs + 1), train_loss_array[1:], 'r')
    val_loss, = plt.plot(np.arange(1, epochs + 1), valid_loss_array[1:], 'g')
    plt.xlabel('epochs')
    plt.ylabel('Estimation loss')
    plt.title("train/valid loss vs epoch")
    # 添加图例
    ax_loss.legend(handles=[train_loss, val_loss], labels=['train_epoch_loss', 'val_epoch_loss'],
                         loc='best')
    fig_loss.savefig(config.save_fig_dir + 'loss.png')
    plt.close(fig_loss)

    fig_int_error, ax_int_error = plt.subplots(figsize=(12, 8))
    train_wind_error, = plt.plot(np.arange(1, epochs + 1),
                                                      train_wind_error_array[1:], 'r')
    valid_wind_error, = plt.plot(np.arange(1, epochs + 1),
                                       valid_wind_error_array[1:], 'g')
    plt.xlabel('epochs')
    plt.ylabel('Intensity Error')
    plt.title("train/valid error vs epoch")
    # 添加图例
    ax_int_error.legend(handles=[train_wind_error,
                              valid_wind_error],
                     labels=['train_error', 'valid_error'],
                     loc='best')
    fig_int_error.savefig(config.save_fig_dir + 'int_error.png')
    plt.close(fig_int_error)

    # 绘制RMW误差
    fig_rmw_error, ax_rmw_error = plt.subplots(figsize=(12, 8))
    train_rmw_error, = plt.plot(np.arange(1, epochs + 1),
                                 train_rmw_error_array[1:], 'r')
    valid_rmw_error, = plt.plot(np.arange(1, epochs + 1),
                                 valid_rmw_error_array[1:], 'g')
    plt.xlabel('epochs')
    plt.ylabel('RMW Error')
    plt.title("train/valid error vs epoch")
    # 添加图例
    ax_rmw_error.legend(handles=[train_rmw_error,
                                 valid_rmw_error],
                        labels=['train_error', 'valid_error'],
                        loc='best')
    fig_rmw_error.savefig(config.save_fig_dir + 'rmw_error.png')
    plt.close(fig_rmw_error)

if __name__ == '__main__':
    Set_seed.setup_seed(5)

    model = PCCNet().to(config.device)

    params = [1, 1]
    Estim_Loss = nn.L1Loss()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.0005, momentum=0.4)

    train(model, Estim_Loss, optimizer, params, config.epochs)
