import glob
from matplotlib import image
import math
import numpy as np
import torch
import os
from tqdm import tqdm
import PCC_Config as config

def default_loader(path):
    raw_data = np.load(path, allow_pickle=True)
    tensor_data = torch.from_numpy(raw_data)
    return tensor_data

class Dataset():
    def __init__(self, data_btemp_path, data_transforms=None, data_format='npy'):

        self.data_transforms = data_transforms
        self.data_format = data_format
        self.npy_btemp_paths = data_btemp_path

        self.npys_btemp = []

        self.winds = []
        self.RMWs = []
        self.R34s = []
        self.pressures = []

        self.pres = []
        self.lats = []
        self.lons = []
        self.ts = []
        self.categories = []

        for npy_file in os.listdir(data_btemp_path):
            filename = npy_file.split("_")
            lat = float(filename[0])
            lon = float(filename[1])
            t = float(filename[2])
            pre = float(filename[3])
            pressure = float(filename[4])
            wind = float(filename[5])
            RMW = float(filename[6])
            R34 = float(filename[7])
            category = int(filename[8])

            self.npys_btemp.append(data_btemp_path + npy_file)

            self.winds.append(wind)
            self.RMWs.append(RMW)
            self.R34s.append(R34)
            self.pressures.append(pressure)

            self.pres.append(pre)
            self.lats.append(lat)
            self.lons.append(lon)
            self.ts.append(t)
            self.categories.append(category)

        # 标签归一化
        for i in range(len(self.winds)):
            self.winds[i] = (self.winds[i] - 19) / (170 - 19)
            self.RMWs[i] = (self.RMWs[i] - 5) / (200 - 5)
            self.lats[i] = (self.lats[i] - (-32.0377)) / (44.9 - (-32.0377))
            self.lons[i] = (self.lons[i] - 86.27) / (193.7 - 86.27)

    def __len__(self):
        return len(self.npys_btemp)

    def __getitem__(self, index):
        # 8, 156, 156
        btemp_file_path = self.npys_btemp[index]
        npy_btemp = default_loader(btemp_file_path)

        if self.data_transforms is not None:
            npy_btemp = self.data_transforms(npy_btemp)

        wind = self.winds[index]
        RMW = self.RMWs[index]
        R34 = self.R34s[index]
        pressure = self.pressures[index]

        pre = self.pres[index]
        lat = self.lats[index]
        lon = self.lons[index]
        t = self.ts[index]
        category = self.categories[index]

        sample = {'lat': lat, 'lon': lon, 'occur_t': t,
                  'pre': pre, 'category': category,
                  'btemp': npy_btemp,
                  'RMW': RMW, 'wind': wind, 'pressure': pressure, 'R34': R34}

        return sample


if __name__ == '__main__':

    print("trainset中的标签最值：")
    train_dataset = Dataset(config.train_npy_path, None, config.data_format)
    print("valset中的标签最值：")
    valid_dataset = Dataset(config.valid_npy_path, None, config.data_format)
    print("testset中的标签最值：")
    test_dataset = Dataset(config.predict_npy_path, None, config.data_format)
    for batch, data in enumerate(tqdm(test_dataset)):
        t = data['occur_t']
        btemp156 = data['btemp']
