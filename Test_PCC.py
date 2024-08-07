import math
import time
import torch
import numpy as np
from tqdm import tqdm
from PCC_Dataset import Dataset
import PCC_Config as config
transform_test = None

def test_model(model, dataset):

    model.eval()

    item = 0

    batch_num = 0

    wind_error = 0
    wind_RMSE_sum = 0

    rmw_error = 0
    rmw_RMSE_sum = 0

    wind_output_list = []
    wind_label_list = []
    RMW_output_list = []
    RMW_label_list = []

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

            batch_num += 1

            print("wind_label={}".format(wind_label))
            print("Intensity={}".format(wind))

            print("rmw_label={}".format(rmw_label))
            print("RMW={}".format(rmw))

            # 反归一化
            wind_label_re = wind_label.cpu().detach().numpy() * (170 - 19) + 19
            wind_re = wind.cpu().detach().numpy() * (170 - 19) + 19
            wind_error = wind_error + np.sum(np.abs(wind_re - wind_label_re))
            wind_RMSE_sum = wind_RMSE_sum + np.sum((wind_re - wind_label_re) * (wind_re - wind_label_re))

            rmw_label_re = rmw_label.cpu().detach().numpy() * (200 - 5) + 5
            rmw_re = rmw.cpu().detach().numpy() * (200 - 5) + 5
            rmw_error = rmw_error + np.sum(np.abs(rmw_re - rmw_label_re))
            rmw_RMSE_sum = rmw_RMSE_sum + np.sum((rmw_re - rmw_label_re) * (rmw_re - rmw_label_re))

            item += rmw_label.shape[0]

            for i in range(0, len(rmw_re)):
                wind_output_list.append(wind_re[i])
                wind_label_list.append(wind_label_re[i])
                RMW_output_list.append(rmw_re[i])
                RMW_label_list.append(rmw_label_re[i])

    return wind_error, wind_RMSE_sum, wind_label_list, wind_output_list, \
           rmw_error, rmw_RMSE_sum, RMW_label_list, RMW_output_list

if __name__ == '__main__':
    model = torch.load(config.predict_model, map_location='cuda:0').to(config.device)
    test_transform = None
    testset = Dataset(config.predict_npy_path, test_transform,
                      config.data_format)

    test_dataloader = torch.utils.data.DataLoader(
        testset, batch_size=config.batch_size,
        shuffle=True, num_workers=config.num_workers
    )

    ''' statistic  '''
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))

    torch.cuda.synchronize()
    start = time.time()

    wind_error, wind_RMSE_sum, wind_label_list, wind_output_list, \
    rmw_error, rmw_RMSE_sum, RMW_label_list, RMW_output_list = test_model(model, test_dataloader)

    torch.cuda.synchronize()
    end = time.time()
    print('infer_time:', end - start)

    test_num = len(RMW_label_list)

    test_wind_error = wind_error / test_num
    wind_RMSE = math.sqrt(wind_RMSE_sum / test_num)
    print("MAE of MSW={}".format(test_wind_error))
    print("RMSE of MSW={}".format(wind_RMSE))

    test_RMW_error = rmw_error / test_num
    RMW_RMSE = math.sqrt(rmw_RMSE_sum / test_num)
    print("MAE of RMW={}".format(test_RMW_error))
    print("RMSE of RMW={}".format(RMW_RMSE))
