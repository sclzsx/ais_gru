# -*- coding:utf-8 -*-
"""
作者：
日期：年月日
"""
import sys
sys.path.append('./process_data')
# sys.path.append('D:/Project/AIS/ckpt')
from split_dataset import firsts
from math import sin, cos, sqrt, atan2, radians
import pandas as pd

# import folium

from models.network import *
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', type=str, default='./dataset/processed/20w_new_train_test_data.npz', help='dir of the dataset')
parser.add_argument('--log_path', type=str, default='./logs/', help='tensorboard logs')
parser.add_argument('--model_path', type=str, default='./ckpt', help='save model path')
parser.add_argument('--dataset_type', type=str, default='new_dataset', help='new_dataset, og_dataset,...')
parser.add_argument('--use_rnn_type', type=str, default='gru', help='lstm gru')
parser.add_argument('--input_size', type=int, default=5, help='the length of every timestep')
parser.add_argument('--hidden_size', type=int, default=20, help='the output length of every timestep ')
parser.add_argument('--num_layers', type=int, default=3, help='the number of lstm layer(vertical)')
parser.add_argument('--out_size', type=int, default=2, help='the length of predicted sequence')
parser.add_argument('--seq_len', type=int, default=5, help='the length of predicted sequence')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
opt = parser.parse_args()
# print(opt)
write = SummaryWriter(opt.log_path + opt.dataset_type + '/' + opt.use_rnn_type)

# Pull Data From Numpy File and load dataset
training_data = np.load(opt.dataset_dir)
x_test = torch.FloatTensor(training_data['x_test']).cuda()
y_test = torch.FloatTensor(training_data['y_test']).cuda()
test_dataset = TensorDataset(x_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)

if opt.use_rnn_type == 'lstm':
    model = lstm(input_size=opt.input_size, hidden_size=opt.hidden_size, output_size=opt.out_size,
                 num_layers=opt.num_layers, seq_len=opt.seq_len)
elif opt.use_rnn_type == 'gru':
    model = gru(input_size=opt.input_size, hidden_size=opt.hidden_size, output_size=opt.out_size,
                 num_layers=opt.num_layers, seq_len=opt.seq_len)

device = torch.device("cuda")
# model.load_state_dict(torch.load(opt.model_path+opt.dataset_type + '/' + opt.use_rnn_type+'/' + 'best_model.pt'))
model.load_state_dict(torch.load('./ckpt/new_dataset/gru/best_model.pt'))
model.to(device)
model.eval()
# Predict Outputs
# Post Processing
firsts = firsts()
first_lat = firsts[0]
first_long = firsts[1]
unique_count = firsts[2]

prediction = []
y_test1 = []
k = 0
last_count = 0
for i, (test_input, test_label) in enumerate(test_loader):
    print(test_input)
    test_y = model(test_input)
    # print(test_y.cpu().detach().numpy()[0].tolist(),'4444444444')
    pred = np.array(test_y.cpu().detach().numpy()[0].tolist())
    # print(prediction,'4444444444')
    pred[0] = pred[0] + first_lat[k]
    pred[1] = pred[1] + first_long[k]
    prediction.append(pred)
    test_label = np.array(test_label.cpu().detach().numpy()[0].tolist())
    test_label[0] = test_label[0] + first_lat[k]
    test_label[1] = test_label[1] + first_long[k]
    y_test1.append(test_label)

    if (last_count - i) == unique_count[k]:
        last_count = i
        k += 1

prediction = np.array(prediction)
#print(prediction.shape,'11111111111111')
y_test1 = np.array(y_test1)


# print(y_test1,'22222222222')
# def sommth_plot(x_arr, y_arr):
#      fig = plt.figure() # 创建一个figure
#      ax = Subplot(fig, 111) # 利用Subplot将figure加入ax
#      fig.add_axes(ax)
#      ax.axis['bottom'].set_axisline_style("->", size=1.5) # x轴加上箭头
#      ax.axis['left'].set_axisline_style("->", size=1.5) # y轴加上上箭头
#      ax.axis['top'].set_visible(False) # 去除上方坐标轴
#      ax.axis['right'].set_visible(False) # 去除右边坐标轴
#      xmin = min(x_arr)
#      xmax = max(x_arr)
#      xnew = np.arange(xmin, xmax, 0.0005) # 在最大最小值间以间隔为0.0005插入点
#      func = interpolate.interp1d(x_arr, y_arr)
#      ynew = func(xnew) # 得到插入x对应的y值
#      plt.plot(xnew, ynew, '-') # 绘制图像
#      plt.show() # show图像
# print(prediction.shape(0),'44444444444')
# for i in range(prediction.size(0)):
#     print(prediction[1])
    # sommth_plot()

    # prediction.append(test_y.cpu().detach().numpy()[0].tolist())
# prediction = np.array(prediction)
# print(prediction.shape,'22222222')

# Adding lats and longs back to give actual predictions
# k = 0
# last_count = 0
# print(len(y_test),'00000000')
# for i in range(len(y_test)):
#     prediction[i, 0] = prediction[i, 0] + first_lat[k]
#     prediction[i, 1] = prediction[i, 1] + first_long[k]
#     y_test[i, 0] = y_test[i, 0] + first_lat[k]
#     y_test[i, 1] = y_test[i, 1] + first_long[k]
#     if (last_count - i) == unique_count[k]:
#         last_count = i
#         k += 1
# Function Definitions
def rmse(y_true, y_pred):
    return torch.sqrt(torch.mean(torch.square(y_pred - y_true), axis=-1))


# Get distance between pairs of lat-lon points (in meters)
def distance(lat1, lon1, lat2, lon2):
    r = 6373.0

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    dist = r*c*1000

    return dist
# Determining average distance between prediction and y_test
dict = {'lat1': prediction[:, 0], 'lon1': prediction[:, 1], 'lat2': y_test1[:, 0], 'lon2': y_test1[:, 1]}
df_lls = pd.DataFrame(dict)
print(df_lls)
import matplotlib.pyplot as plt
import numpy as np
def plot(y_test,y_predict):
    assert len(y_test)==len(y_predict)
    n = len(y_test)
    plt.figure()
    plt.plot(np.arange(n),y_test,label='True')
    plt.plot(np.arange(n),y_predict,label='Prediction')
    plt.legend()
    plt.show()
plot(y_test1[:, 0],prediction[:, 0])
plot(y_test1[:, 1],prediction[:, 1])
def plot_rmse(y_test,y_predict):
    assert len(y_test)==len(y_predict)
    n = len(y_test)
    y_predict = y_predict.reshape(-1, n)
    y_test = y_test.reshape(-1, n)
    rmse_ = np.linalg.norm(y_test - y_predict, ord=2) / n ** 0.5
    print(rmse_)

    mae_ = np.linalg.norm(y_test - y_predict, ord=1) / n
    print(mae_)

plot_rmse(y_test1[:, 0], prediction[:, 0])
plot_rmse(y_test1[:, 1],prediction[:, 1])
dist = np.empty(len(df_lls['lat1']))

for i in range(dist.size):
    dist[i] = distance(df_lls['lat1'][i],
                          df_lls['lon1'][i],
                          df_lls['lat2'][i],
                          df_lls['lon2'][i])

# Find the average distance in meters
#  avg_dist = np.mean(dist) Maybe work
nine_sort = np.sort(dist)
avg_dist = nine_sort[int(0.9*len(dist))]  # currently the bottom 90% distances of sorted distances

print('----------------------------------------------------')
print('Average Distance (m): ', avg_dist, ' m')
print('Average Distance (km): ', avg_dist / 1000, ' km')
print('Average Distance (NM): ', avg_dist * 0.00053996, 'NM')
print('----------------------------------------------------')
print('End of danish_predict.py')
print('----------------------------------------------------')

# graph AOU

location_index = 50

center = [prediction[location_index, 0], prediction[location_index, 1]]
tile='http://rt{s}.map.gtimg.com/realtimerender?z={z}&x={x}&y={y}&type=vector&style=0'
tile='https://t1.tianditu.gov.cn/vec_w/wmts?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0&LAYER=vec&STYLE=default&TILEMATRIXSET=w&FORMAT=tiles&TILECOL={x}&TILEROW={y}&TILEMATRIX={z}&tk=20560cc859c9049571b64d2d816fc446'
# m = folium.Map(location=center, tiles="Stamen Toner", zoom_start=12)


# m = folium.Map(location=center, zoom_start=12,tiles=tile,attr='天地图')


'''
# Real Location
folium.Circle(
    radius=20,
    location=[y_test[location_index, 0], y_test[location_index, 1]],
    popup='Real Location',
    color='crimson',
    fill=True,
    fill_color='#ffcccb'
).add_to(m)

# AOU
folium.Circle(
    location=[prediction[location_index, 0], prediction[location_index, 1]],
    radius=avg_dist,  # might want to take largest distance!
    popup='AOU Radius: ' + str(avg_dist) + ' meters',
    color='#3186cc',
    fill=True,
    fill_color='#3186cc'
).add_to(m)
'''
# size, index = prediction.shape
# for i in range(size):
#     folium.Circle(
#         radius=20,
#         location=[y_test1[i, 0], y_test1[i, 1]],
#         popup='Real Location',
#         color='#cc0000',
#         fill=True,
#         fill_color='#cc0000'
#     ).add_to(m)
#
#     # AOU
#     folium.Circle(
#         location=[prediction[i, 0], prediction[i, 1]],
#         radius=20,  # might want to take largest distance! avg_dist
#         popup='AOU Radius: ' + str(avg_dist) + ' meters',
#         color='#3186cc',
#         fill=True,
#         fill_color='#3186cc'
#     ).add_to(m)
#
# m.save("map_lstm.html")

# TODO:
# can graph multiple prediction points and AOU's
df_lls.to_csv('result/model_{}.csv'.format(opt.use_rnn_type))