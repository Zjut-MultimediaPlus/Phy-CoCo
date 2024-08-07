# train
epochs = 600
batch_size = 128
device = 'cuda:0'  # cpu or 'cuda:0'

train_npy_path = '/pre_now_npy_156/train/'
valid_npy_path = '/pre_now_npy_156/valid/'

num_workers = 4  # 加载数据集线程并发数
best_loss = 0.005  # 当loss小于等于该值会保存模型
save_model_iter = 25  # 每多少次保存一份模型

model_output_dir = '/data/yht/model/PCC_Net/'

predict_model = '/opt/data/private/model/PCC_Net_240305/epoch_600.pth'

predict_npy_path = '/pre_now_npy_156/test/'

save_fig_dir = '/data/yht/model/PCC_Net/exp_img/'

data_format = 'npy'