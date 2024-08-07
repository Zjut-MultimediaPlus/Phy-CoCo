import torch.fft
import torch.nn.functional as F
import math
import torch
from torch import nn
from collections import OrderedDict
import PCC_Config as config

class Shared_Network156_chres(nn.Module):
    def __init__(self):
        super().__init__()
        self.Lat_FC = nn.Sequential(OrderedDict(
            [('Lat_FC1', nn.Linear(1, 5)),
             ('Tanh', nn.Tanh()),
             ('Lat_FC2', nn.Linear(5, 25)),
             ('Tanh', nn.Tanh()),
             ('Lat_FC3', nn.Linear(25, 58)),
             ('Tanh', nn.Tanh()),
             ('Lat_FC4', nn.Linear(58, 156)),
             ('Tanh', nn.Tanh())]))
        self.Lon_FC = nn.Sequential(OrderedDict(
            [('Lon_FC1', nn.Linear(1, 5)),
             ('Tanh', nn.Tanh()),
             ('Lon_FC2', nn.Linear(5, 25)),
             ('Tanh', nn.Tanh()),
             ('Lon_FC3', nn.Linear(25, 58)),
             ('Tanh', nn.Tanh()),
             ('Lon_FC4', nn.Linear(58, 156)),
             ('Tanh', nn.Tanh())]))
        self.conv1 = nn.Conv2d(9, 16, kernel_size=7, stride=3, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.conv4 = nn.Conv2d(96, 192, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # self.pre2branch_w1 = nn.Sequential(OrderedDict(
        #     [('branch1_linear1', nn.Linear(1, 2)),
        #      ('Tanh', nn.Tanh()),
        #      ('branch1_linear2', nn.Linear(2, 4)),
        #      ('Tanh', nn.Tanh())]))
        # self.pre2branch_w2 = nn.Sequential(OrderedDict(
        #     [('branch2_linear1', nn.Linear(1, 2)),
        #      ('Tanh', nn.Tanh()),
        #      ('branch2_linear2', nn.Linear(2, 4)),
        #      ('Tanh', nn.Tanh())]))

        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()

    def forward(self, x, lat, lon):
        # bs, 156
        lat = self.Lat_FC(lat)
        # print(lat)
        # bs, 156, 1
        lat = lat.unsqueeze(dim=2)
        # bs, 156
        lon = self.Lon_FC(lon)
        # print(lon)
        # bs, 1, 156
        lon = lon.unsqueeze(dim=1)
        # bs, 156, 156
        geo_aux = torch.bmm(lat, lon)
        # bs, 1, 156, 156
        geo_aux = geo_aux.unsqueeze(dim=1)
        # bs, 8, 156, 156
        # x = x * geo_aux

        # bs, 1, 1, 1
        # tw = t.unsqueeze(dim=1).unsqueeze(dim=2)
        # t0w = t0.unsqueeze(dim=1).unsqueeze(dim=2)
        # x[:, :4] = t0w * x[:, :4]
        # x[:, 4:] = tw * x[:, 4:]

        # bs, 9, 156, 156
        x = torch.cat((x, geo_aux), dim=1)

        x = self.conv1(x)  # (16, 50, 50)
        x = self.relu(x)
        x = self.pool1(x)  # (16, 24, 24)

        x = self.conv2(x)  # (32, 20, 20)
        x = self.relu(x)
        x = self.pool2(x)  # (32, 9, 9)
        res_x1 = x

        x = self.conv3(x)  # (64, 9, 9)
        x = self.relu(x)
        x = torch.cat((res_x1, x), dim=1)       # (96, 9, 9)
        x = self.pool3(x)  # (96, 4, 4)
        res_x2 = x

        x = self.conv4(x)  # (192, 4, 4)
        x = self.relu(x)
        x = torch.cat((res_x2, x), dim=1)       # (96 + 192=288, 4, 4)
        x = self.pool4(x)  # (288, 2, 2)
        # x = self.flatten(x)

        # pre2w1 = self.pre2branch_w1(pre)
        # pre2w1 = pre2w1.unsqueeze(dim=1)
        # pre2w1 = pre2w1.reshape(pre2w1.shape[0], 1, 2, 2)
        #
        # pre2w2 = self.pre2branch_w2(pre)
        # pre2w2 = pre2w2.unsqueeze(dim=1)
        # pre2w2 = pre2w2.reshape(pre2w2.shape[0], 1, 2, 2)
        #
        # wind_share_info = pre2w1 * x
        # wind_share_info = wind_share_info.reshape(wind_share_info.shape[0], 288, 2, 2)
        # size_share_info = pre2w2 * x
        # size_share_info = size_share_info.reshape(size_share_info.shape[0], 288, 2, 2)
        #
        # return wind_share_info, size_share_info
        return x

class Size_Cross_Atten(nn.Module):
    # 初始化
    def __init__(self, dim, num_heads=2, qkv_bias=False, atten_drop_ratio=0., proj_drop_ratio=0.):
        super(Size_Cross_Atten, self).__init__()

        # 多头注意力的数量
        self.num_heads = num_heads
        # 将生成的qkv均分成num_heads个。得到每个head的qkv对应的通道数。
        head_dim = dim // num_heads
        # 公式中的分母
        self.scale = head_dim ** -0.5

        # 通过一个全连接层计算qkv
        self.k = nn.Linear(in_features=dim, out_features=dim, bias=qkv_bias)
        self.qv = nn.Linear(in_features=dim, out_features=dim * 2, bias=qkv_bias)

        # dropout层
        self.atten_drop = nn.Dropout(atten_drop_ratio)

        # 再qkv计算完之后通过一个全连接提取特征
        self.proj = nn.Linear(in_features=dim, out_features=dim)
        # dropout层
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    # 前向传播
    def forward(self, cinfo, clen):
        # features: bs, 78, 4
        # t:   bs, 4
        B, N, C = cinfo.shape
        # b, 1, 4
        # promot = self.t2prompt(t).unsqueeze(dim=1)

        # 将输入特征图经过全连接层生成q [b,78,4]==>[b,78,4]
        k = self.k(clen)
        # 将输入特征图经过全连接层生成kv [b,78,4]==>[b,78,4*2]
        qv = self.qv(cinfo)

        # 维度调整 [b,78,4*2]==>[b, 78, 2, 2, 4//2]
        qv = qv.reshape(B, N, 2, self.num_heads, C // self.num_heads)
        # [b, 78, 1, 2, 4 // 2]
        k = k.reshape(B, N, 1, self.num_heads, C // self.num_heads)

        # 维度重排==> [2, B, 2, 78, 4//2]
        qv = qv.permute(2, 0, 3, 1, 4)
        k = k.permute(2, 0, 3, 1, 4)
        # 切片提取q、k、v的值，单个的shape=[B, 2, 64, 78//2]
        q, v = qv[0], qv[1]
        k = k[0]
        # 针对每个head计算 ==> [B, 2, 78, 2]
        atten = (q @ k.transpose(-2, -1)) * self.scale  # @ 代表在多维tensor的最后两个维度矩阵相乘
        # 对计算结果的每一行经过softmax
        atten = atten.softmax(dim=-1)
        # dropout层
        atten = self.atten_drop(atten)

        # softmax后的结果和v加权 ==> [B, 2, 78, 4//2]
        x = atten @ v
        # 通道重排 ==> [B, 78, 2, 4//2]
        x = x.transpose(1, 2)
        # 维度调整 ==> [B, 78, 4]
        x = x.reshape(B, N, C)

        # 通过全连接层融合特征 ==> [B, 78, 4]
        x = self.proj(x)
        # dropout层
        x = self.proj_drop(x)

        # x = promot * x

        return x

# Temporal Center Expand Pooling module
class Size_TCP(nn.Module):
    def __init__(self):
        super().__init__()

        self.GMP = nn.AdaptiveMaxPool2d((2, 2))
        self.GAP = nn.AdaptiveAvgPool2d((2, 2))

        self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2d = nn.Conv2d(78 * 2, 78, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()

        self.attention = Size_Cross_Atten(4)

        self.size_conv1ds = nn.Sequential(
            nn.Conv1d(78, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )
        self.pre2w = nn.Sequential(
            nn.Linear(1, 2),
            nn.Tanh(),
            nn.Linear(2, 4),
            nn.Tanh()
        )

    def forward(self, cloud_top, pre):
        # b, 1, 156, 156
        cloud_top = cloud_top.unsqueeze(dim=1)
        b, c, h, w = cloud_top.shape[0], cloud_top.shape[1], cloud_top.shape[2], cloud_top.shape[3]
        half = int(h / 2)
        pool_avg = []
        pool_max = []
        center_x, center_y = half, half
        for i in range(1, center_x + 1):
            index_x0, index_x1 = center_x - i, center_x + i
            index_y0, index_y1 = center_y - i, center_y + i
            pool_input = cloud_top[:, :, index_x0:index_x1 + 1, index_y0:index_y1 + 1]
            # list_len = 78 , every element in list : 1, 2, 2
            pool_avg.append(self.GAP(pool_input))
            pool_max.append(self.GMP(pool_input))

        # b, 78, 2, 2
        pool_avg = torch.cat(pool_avg, dim=1)
        pool_max = torch.cat(pool_max, dim=1)

        # b, 2 * 78, 2, 2
        pool_x = torch.cat([pool_avg, pool_max], dim=1)
        # b, 78, 2, 2
        pool_x = self.conv2d(pool_x)
        pool_x = self.relu(pool_x)
        # b, 78, 4
        cinfo = pool_x.reshape(b, half, 4)
        # 78
        clen = torch.tensor(range(1, 78+1)).to(config.device)
        # 转为nmi
        clen = clen * 2.7
        clen = (clen - 2.7) / (78 * 2.7 - 2.7)
        # b, 78, 4
        clen = clen.unsqueeze(dim=0).unsqueeze(dim=-1).expand(b, -1, 4)

        ''' q:cinfo; k:cinfo; v:clen '''
        # b, 78, 4
        size_tcp_info = self.attention(clen, cinfo)
        # b, 64, 4
        size_tcp_info = self.size_conv1ds(size_tcp_info)
        # b, 64, 2, 2
        size_tcp_info = size_tcp_info.reshape(b, 64, 2, 2)

        pre2w = self.pre2w(pre)
        pre2w = pre2w.unsqueeze(dim=1)
        pre2w = pre2w.reshape(pre2w.shape[0], 1, 2, 2)
        size_tcp_info = size_tcp_info * pre2w

        return size_tcp_info

class Wind_Cross_Atten(nn.Module):
    # 初始化
    def __init__(self, dim, num_heads=2, qkv_bias=False, atten_drop_ratio=0., proj_drop_ratio=0.):
        super(Wind_Cross_Atten, self).__init__()

        # 多头注意力的数量
        self.num_heads = num_heads
        # 将生成的qkv均分成num_heads个。得到每个head的qkv对应的通道数。
        head_dim = dim // num_heads
        # 公式中的分母
        self.scale = head_dim ** -0.5

        # 通过一个全连接层计算qkv
        self.k = nn.Linear(in_features=dim, out_features=dim, bias=qkv_bias)
        self.qv = nn.Linear(in_features=dim, out_features=dim * 2, bias=qkv_bias)

        # dropout层
        self.atten_drop = nn.Dropout(atten_drop_ratio)

        # 再qkv计算完之后通过一个全连接提取特征
        self.proj = nn.Linear(in_features=dim, out_features=dim)
        # dropout层
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    # 前向传播
    def forward(self, cinfo, clen):
        # features: bs, 78, 4
        # t:   bs, 4
        B, N, C = cinfo.shape
        # b, 1, 4
        # promot = self.t2prompt(t).unsqueeze(dim=1)

        # 将输入特征图经过全连接层生成q [b,78,4]==>[b,78,4]
        k = self.k(clen)
        # 将输入特征图经过全连接层生成kv [b,78,4]==>[b,78,4*2]
        qv = self.qv(cinfo)

        # 维度调整 [b,78,4*2]==>[b, 78, 2, 2, 4//2]
        qv = qv.reshape(B, N, 2, self.num_heads, C // self.num_heads)
        # [b, 78, 1, 2, 4 // 2]
        k = k.reshape(B, N, 1, self.num_heads, C // self.num_heads)

        # 维度重排==> [2, B, 2, 78, 4//2]
        qv = qv.permute(2, 0, 3, 1, 4)
        k = k.permute(2, 0, 3, 1, 4)
        # 切片提取q、k、v的值，单个的shape=[B, 2, 64, 78//2]
        q, v = qv[0], qv[1]
        k = k[0]
        # 针对每个head计算 ==> [B, 2, 78, 2]
        atten = (q @ k.transpose(-2, -1)) * self.scale  # @ 代表在多维tensor的最后两个维度矩阵相乘
        # 对计算结果的每一行经过softmax
        atten = atten.softmax(dim=-1)
        # dropout层
        atten = self.atten_drop(atten)

        # softmax后的结果和v加权 ==> [B, 2, 78, 4//2]
        x = atten @ v
        # 通道重排 ==> [B, 78, 2, 4//2]
        x = x.transpose(1, 2)
        # 维度调整 ==> [B, 78, 4]
        x = x.reshape(B, N, C)

        # 通过全连接层融合特征 ==> [B, 78, 4]
        x = self.proj(x)
        # dropout层
        x = self.proj_drop(x)

        # x = promot * x

        return x

# Temporal Center Expand Pooling module
class Wind_TCP(nn.Module):
    def __init__(self):
        super().__init__()

        self.GMP = nn.AdaptiveMaxPool2d((2, 2))
        self.GAP = nn.AdaptiveAvgPool2d((2, 2))

        self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2d = nn.Conv2d(78 * 2, 78, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()

        self.attention = Wind_Cross_Atten(4)

        self.wind_conv1ds = nn.Sequential(
            nn.Conv1d(78, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )
        self.pre2w = nn.Sequential(
            nn.Linear(1, 2),
            nn.Tanh(),
            nn.Linear(2, 4),
            nn.Tanh()
        )

    def forward(self, cloud_top, pre):
        # b, 1, 156, 156
        cloud_top = cloud_top.unsqueeze(dim=1)
        b, c, h, w = cloud_top.shape[0], cloud_top.shape[1], cloud_top.shape[2], cloud_top.shape[3]
        half = int(h / 2)
        pool_avg = []
        pool_max = []
        center_x, center_y = half, half
        for i in range(1, center_x + 1):
            index_x0, index_x1 = center_x - i, center_x + i
            index_y0, index_y1 = center_y - i, center_y + i
            pool_input = cloud_top[:, :, index_x0:index_x1 + 1, index_y0:index_y1 + 1]
            # list_len = 78 , every element in list : 1, 2, 2
            pool_avg.append(self.GAP(pool_input))
            pool_max.append(self.GMP(pool_input))

        # b, 78, 2, 2
        pool_avg = torch.cat(pool_avg, dim=1)
        pool_max = torch.cat(pool_max, dim=1)

        # b, 2 * 78, 2, 2
        pool_x = torch.cat([pool_avg, pool_max], dim=1)
        # b, 78, 2, 2
        pool_x = self.conv2d(pool_x)
        pool_x = self.relu(pool_x)
        # b, 78, 4
        cinfo = pool_x.reshape(b, half, 4)
        # 78
        clen = torch.tensor(range(1, 78+1)).to(config.device)
        # 转为nmi
        clen = clen * 2.7
        clen = (clen - 2.7) / (78 * 2.7 - 2.7)
        # b, 78, 4
        clen = clen.unsqueeze(dim=0).unsqueeze(dim=-1).expand(b, -1, 4)

        ''' q:cinfo; k:cinfo; v:clen '''
        # b, 78, 4
        wind_tcp_info = self.attention(cinfo, clen)
        # b, 64, 4
        wind_tcp_info = self.wind_conv1ds(wind_tcp_info)
        # b, 64, 2, 2
        wind_tcp_info = wind_tcp_info.reshape(b, 64, 2, 2)

        pre2w = self.pre2w(pre)
        pre2w = pre2w.unsqueeze(dim=1)
        pre2w = pre2w.reshape(pre2w.shape[0], 1, 2, 2)
        wind_tcp_info = wind_tcp_info * pre2w

        return wind_tcp_info

class PeRCNNuv_RTV(nn.Module):
    def __init__(self, channels):
        super(PeRCNNuv_RTV, self).__init__()
        self.channels = channels
        # self.c1 = nn.Conv2d(in_channels=channels, out_channels=channels // 2, kernel_size=1)
        self.real_c1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)
        self.real_c2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)
        self.real_c3 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)
        # self.real_conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)
        self.real_conv = nn.Conv1d(in_channels=4, out_channels=4, kernel_size=1)

        self.imag_c1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)
        self.imag_c2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)
        self.imag_c3 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)
        # self.imag_conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)
        self.imag_conv = nn.Conv1d(in_channels=4, out_channels=4, kernel_size=1)

        self.relu = nn.ReLU()

        # self.w1 = nn.Parameter(torch.tensor(0.1), requires_grad=True)
        # self.w2 = nn.Parameter(torch.tensor(0.05), requires_grad=True)
        # self.bias = nn.Parameter(torch.ones((1, 2, 2)), requires_grad=True)

    def forward(self, x):
        # 通过傅立叶变换，将x映射到频率和相位两个分量
        B, C, H, W = x.shape            # B, 64, 2, 2

        # r_x = x[0].cpu().sum(dim=0).detach().numpy()
        # plt.imsave("/data/yht/r_x.png", r_x)

        x = x.reshape(B, H, W, C)       # B, 2, 2, 64
        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")  # B, 2, 2, 64   (2/2+1=8)

        x_real = x.real                             # B, 2, 2, 64
        x_real = x_real.reshape(B, C, H, W)         # B, 64, 2, 2
        x_imag = x.imag                             # B, 2, 2, 64
        x_imag = x_imag.reshape(B, C, H, W)         # B, 64, 2, 2

        # rx_real = x_real.cpu().sum(dim=1).detach().numpy()[0]
        # plt.imsave("/data/yht/rx_real.png", rx_real)
        # rx_image = x_imag.cpu().sum(dim=1).detach().numpy()[0]
        # plt.imsave("/data/yht/rx_image.png", rx_image)

        x1_real = self.real_c1(x_real)
        x1_real = self.relu(x1_real)
        x2_real = self.real_c2(x_real)
        x2_real = self.relu(x2_real)
        x3_real = self.real_c3(x_real)
        x3_real = self.relu(x3_real)
        # b, 353 or 64, 2, 2
        x_pole_real = x1_real * x2_real * x3_real

        x_pole_real = x_pole_real.reshape(B, C, 4)
        x_pole_real = x_pole_real.transpose(1, 2)

        x_pole_real = self.real_conv(x_pole_real)
        x_pole_real = self.relu(x_pole_real)

        x_pole_real = x_pole_real.transpose(1, 2)
        x_pole_real = x_pole_real.reshape(B, C, 2, 2)

        # rx_real = x_real.cpu().sum(dim=1).detach().numpy()[0]
        # plt.imsave("/data/yht/rx_real.png", rx_real)

        x_pole_real = x_pole_real.reshape(B, H, W, C)       # B, 2, 2, 64

        x1_imag = self.imag_c1(x_imag)
        x1_imag = self.relu(x1_imag)
        x2_imag = self.imag_c2(x_imag)
        x2_imag = self.relu(x2_imag)
        x3_imag = self.imag_c3(x_imag)
        x3_imag = self.relu(x3_imag)
        # b, 353 or 64, 2, 2
        x_pole_imag = x1_imag * x2_imag * x3_imag

        x_pole_imag = x_pole_imag.reshape(B, C, 4)
        x_pole_imag = x_pole_imag.transpose(1, 2)

        x_pole_imag = self.imag_conv(x_pole_imag)
        x_pole_imag = self.relu(x_pole_imag)

        x_pole_imag = x_pole_imag.transpose(1, 2)
        x_pole_imag = x_pole_imag.reshape(B, C, 2, 2)

        # rx_image = x_imag.cpu().sum(dim=1).detach().numpy()[0]
        # plt.imsave("/data/yht/rx_image.png", rx_image)

        x_pole_imag = x_pole_imag.reshape(B, H, W, C)  # B, 2, 2, 64

        x_ploe = torch.stack([x_pole_real, x_pole_imag], dim=-1)             # B, 2, 2, 64, 2
        x_ploe = F.softshrink(x_ploe, lambd=0.01)
        x_ploe = torch.view_as_complex(x_ploe)                                    # 将输入x变成复数张量： B, 2, 2, 64
        x_ploe = torch.fft.irfft2(x_ploe, s=(H, W), dim=(1, 2), norm="ortho")     # B, 2, 2, 64
        x_ploe = x_ploe.reshape(B, C, H, W)                                       # B, 64, 2, 2

        # x_ploe = self.w1 * x_res + x_ploe

        # bias = nn.Parameter(torch.tensor((np.random.rand(bs, self.channels, 2, 2)), dtype=torch.float32), requires_grad=True).to(config.device)

        return x_ploe

class PeRCNNuv_VTR(nn.Module):
    def __init__(self, channels):
        super(PeRCNNuv_VTR, self).__init__()
        self.channels = channels
        # self.c1 = nn.Conv2d(in_channels=channels, out_channels=channels // 2, kernel_size=1)
        self.real_c1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)
        self.real_c2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)
        self.real_c3 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)
        # self.real_conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)
        self.real_conv = nn.Conv1d(in_channels=4, out_channels=4, kernel_size=1)

        self.imag_c1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)
        self.imag_c2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)
        self.imag_c3 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)
        # self.imag_conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)
        self.imag_conv = nn.Conv1d(in_channels=4, out_channels=4, kernel_size=1)

        self.relu = nn.ReLU()

        # self.w1 = nn.Parameter(torch.tensor(0.1), requires_grad=True)
        # self.w2 = nn.Parameter(torch.tensor(0.05), requires_grad=True)
        # self.bias = nn.Parameter(torch.ones((1, 2, 2)), requires_grad=True)

    def forward(self, x):
        # 通过傅立叶变换，将x映射到频率和相位两个分量
        B, C, H, W = x.shape            # B, 64, 2, 2
        # v_x = x[0].cpu().sum(dim=0).detach().numpy()
        # plt.imsave("/data/yht/v_x.png", v_x)

        x = x.reshape(B, H, W, C)       # B, 2, 2, 64
        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")  # B, 2, 2, 64   (2/2+1=8)

        x_real = x.real                             # B, 2, 2, 64
        x_real = x_real.reshape(B, C, H, W)         # B, 64, 2, 2
        x_imag = x.imag                             # B, 2, 2, 64
        x_imag = x_imag.reshape(B, C, H, W)         # B, 64, 2, 2

        # vx_real = x_real.cpu().sum(dim=1).detach().numpy()[0]
        # plt.imsave("/data/yht/vx_real.png", vx_real)
        # vx_image = x_imag.cpu().sum(dim=1).detach().numpy()[0]
        # plt.imsave("/data/yht/vx_image.png", vx_image)

        x1_real = self.real_c1(x_real)
        x1_real = self.relu(x1_real)
        x2_real = self.real_c2(x_real)
        x2_real = self.relu(x2_real)
        x3_real = self.real_c3(x_real)
        x3_real = self.relu(x3_real)
        # b, 353 or 64, 2, 2
        x_pole_real = x1_real * x2_real * x3_real

        x_pole_real = x_pole_real.reshape(B, C, 4)
        x_pole_real = x_pole_real.transpose(1, 2)

        x_pole_real = self.real_conv(x_pole_real)
        x_pole_real = self.relu(x_pole_real)

        x_pole_real = x_pole_real.transpose(1, 2)
        x_pole_real = x_pole_real.reshape(B, C, 2, 2)
        x_pole_real = x_pole_real.reshape(B, H, W, C)       # B, 2, 2, 64

        x1_imag = self.imag_c1(x_imag)
        x1_imag = self.relu(x1_imag)
        x2_imag = self.imag_c2(x_imag)
        x2_imag = self.relu(x2_imag)
        x3_imag = self.imag_c3(x_imag)
        x3_imag = self.relu(x3_imag)
        # b, 353 or 64, 2, 2
        x_pole_imag = x1_imag * x2_imag * x3_imag

        x_pole_imag = x_pole_imag.reshape(B, C, 4)
        x_pole_imag = x_pole_imag.transpose(1, 2)

        x_pole_imag = self.imag_conv(x_pole_imag)
        x_pole_imag = self.relu(x_pole_imag)

        x_pole_imag = x_pole_imag.transpose(1, 2)
        x_pole_imag = x_pole_imag.reshape(B, C, 2, 2)
        x_pole_imag = x_pole_imag.reshape(B, H, W, C)  # B, 2, 2, 64

        x_ploe = torch.stack([x_pole_real, x_pole_imag], dim=-1)             # B, 2, 2, 64, 2
        x_ploe = F.softshrink(x_ploe, lambd=0.01)
        x_ploe = torch.view_as_complex(x_ploe)                                    # 将输入x变成复数张量： B, 2, 2, 64
        x_ploe = torch.fft.irfft2(x_ploe, s=(H, W), dim=(1, 2), norm="ortho")     # B, 2, 2, 64
        x_ploe = x_ploe.reshape(B, C, H, W)                                       # B, 64, 2, 2

        # x_ploe = self.w1 * x_res + x_ploe

        # bias = nn.Parameter(torch.tensor((np.random.rand(bs, self.channels, 2, 2)), dtype=torch.float32), requires_grad=True).to(config.device)

        return x_ploe

class Vmax_GF(nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv = nn.Sequential(
        #     nn.Conv2d(64*2, 64, kernel_size=1, stride=1, padding=0),
        #     nn.ReLU()
        # )
        self.linear_1 = nn.Linear(256 * 2, 256)
        self.linear_2 = nn.Linear(256, 256)
        self.flatten = nn.Flatten()

    def forward(self, x1, x2):
        # x = [x1, x2]
        # x = torch.cat(x, dim=1)
        # vfusion = self.conv(x)

        bs, c, h, w = x1.shape
        # bs, l
        x1 = self.flatten(x1)
        x2 = self.flatten(x2)
        x = torch.cat([x1, x2], dim=1)
        fusion = torch.tanh(self.linear_1(x))
        fusion = self.linear_2(fusion)
        vfusion = fusion.reshape(bs, c, h, w)

        return vfusion

class RMW_GF(nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv = nn.Sequential(
        #     nn.Conv2d(64 * 2, 64, kernel_size=1, stride=1, padding=0),
        #     nn.ReLU()
        # )
        self.linear_1 = nn.Linear(256 * 2, 256)
        self.linear_2 = nn.Linear(256, 256)
        self.flatten = nn.Flatten()

    def forward(self, x1, x2):
        # x = [x1, x2]
        # x = torch.cat(x, dim=1)
        # rfusion = self.conv(x)

        bs, c, h, w = x1.shape
        # bs, l
        x1 = self.flatten(x1)
        x2 = self.flatten(x2)
        x = torch.cat([x1, x2], dim=1)
        fusion = torch.tanh(self.linear_1(x))
        fusion = self.linear_2(fusion)
        rfusion = fusion.reshape(bs, c, h, w)

        return rfusion

class PCCNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.ShareNet = Shared_Network156_chres()

        self.wind_tcp = Wind_TCP()
        self.VTR = PeRCNNuv_VTR(64)
        self.size_tcp = Size_TCP()
        self.RTV = PeRCNNuv_RTV(64)

        self.fusion_v = Vmax_GF()
        self.fusion_r = RMW_GF()

        self.t_coding = nn.Sequential(OrderedDict(
            [('t_linear1', nn.Linear(1, 2)),
             ('Tanh', nn.Tanh()),
             ('t_linear2', nn.Linear(2, 4)),
             ('Tanh', nn.Tanh())]))

        self.output_intensity = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1412, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1))
        self.output_RMW = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1412, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1))

    def forward(self, x, pre, lat, lon, t):
        # 128, 2, 2 (hard share)
        share_feature = self.ShareNet(x, lat, lon)

        # 64, 2, 2
        size_tcp_info = self.size_tcp(x[:, 6], pre)
        wind_tcp_info = self.wind_tcp(x[:, 6], pre)
        vtr_info = self.VTR(wind_tcp_info)
        rtv_info = self.RTV(size_tcp_info)

        # 64, 2, 2
        Vmaxf = self.fusion_v(wind_tcp_info, rtv_info)
        RMWf = self.fusion_r(size_tcp_info, vtr_info)

        # b, 4
        t = self.t_coding(t)
        # b, 1, 2, 2
        t = t.unsqueeze(dim=1)
        t = t.reshape(t.shape[0], 1, 2, 2)

        wind_fused_f = [share_feature, Vmaxf, t]
        wind_fused_f = torch.cat(wind_fused_f, dim=1)

        size_fused_f = [share_feature, RMWf, t]
        size_fused_f = torch.cat(size_fused_f, dim=1)

        RMW = self.output_RMW(size_fused_f)
        intensity = self.output_intensity(wind_fused_f)

        RMW = RMW[:, 0]
        intensity = intensity[:, 0]

        return RMW, intensity

