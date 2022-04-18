# -*- coding:utf-8 -*-
"""
作者：
日期：年月日
"""
from datetime import datetime
import numpy as np
import os
from network import *
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import ToTensor
from torch.utils.tensorboard import SummaryWriter
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', type=str, default='./dataset/processed/20w_new_train_test_data.npz', help='dir of the dataset')
parser.add_argument('--log_path', type=str, default='./logs/', help='tensorboard logs')
parser.add_argument('--model_path', type=str, default='./ckpt/', help='save model path')
parser.add_argument('--dataset_type', type=str, default='new_dataset', help='new_dataset, og_dataset,...')
# parser.add_argument('--time_step', type=int, default= 27, help=' it indicates day')
# parser.add_argument('--pred_len', type=int, default= 3, help=' it indicates predicted day')
parser.add_argument('--milestones', type=list, default=[15, 25], help=' setup MultiStepLR ')
parser.add_argument('--input_size', type=int, default=5, help='the length of every timestep')
parser.add_argument('--hidden_size', type=int, default=20, help='the output length of every timestep ')
parser.add_argument('--num_layers', type=int, default=3, help='the number of lstm layer(vertical)')
parser.add_argument('--out_size', type=int, default=2, help='the length of predicted sequence')
parser.add_argument('--seq_len', type=int, default=5, help='the length of predicted sequence')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--n_epoch', type=int, default=100, help='number of epochs of training')
parser.add_argument('--lr', type=float, default=0.0001, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.9, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
#parser.add_argument('--start_test_epoch', type=int, default=50, help='epoch of start testing during training')
parser.add_argument('--use_rnn_type', type=str, default='gru', help='lstm gru bi_gru')
parser.add_argument('--checkpoint_batch', type=int, default=2000, help='1000 times in a epoch')

opt = parser.parse_args()
# print(opt)
torch.manual_seed(10007)
write = SummaryWriter(opt.log_path + opt.dataset_type + '/' + opt.use_rnn_type)

# num_features = 5        # The data we are submitting per time step (lat, long, speed, time, course)
# num_timesteps = 5       # The number of time steps per sequence (track)

# Pull Data From Numpy File and load dataset
training_data = np.load(opt.dataset_dir)
x_train = torch.FloatTensor(training_data['x_train']).cuda()
y_train = torch.FloatTensor(training_data['y_train']).cuda()
x_test = torch.FloatTensor(training_data['x_test']).cuda()
y_test = torch.FloatTensor(training_data['y_test']).cuda()
train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)


if opt.use_rnn_type == 'lstm':
    model = lstm(input_size=opt.input_size, hidden_size=opt.hidden_size, output_size=opt.out_size,
                 num_layers=opt.num_layers, seq_len=opt.seq_len, is_bi=False)
elif opt.use_rnn_type == 'gru':
    model = gru(input_size=opt.input_size, hidden_size=opt.hidden_size, output_size=opt.out_size,
                 num_layers=opt.num_layers, seq_len=opt.seq_len, is_bi=False)
elif opt.use_rnn_type == 'bi_gru':
    model = gru(input_size=opt.input_size, hidden_size=opt.hidden_size, output_size=opt.out_size,
                 num_layers=opt.num_layers, seq_len=opt.seq_len, is_bi=True)

# Custom adam optimizer
optimizer  = Adam(model.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))#SGD
# optimizer  = SGD(model.parameters(), lr=opt.lr)#SGD
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestones, gamma=0.5)

cost = nn.MSELoss()#,,,,,,,,,-as

for epoch in range(1, opt.n_epoch + 1):
    train_b = 0
    train_loss = 0
    model = model.cuda()
    scheduler.step()
    model.train()
    for i, (input, label) in enumerate(train_loader):
        train_b +=1
        # print(input.size(),'1111111111111')
        predict= model(input)
        loss = cost(predict, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % opt.checkpoint_batch == 0:
            print("[Epoch %d/%d] [Batch %d/%d] [MSEloss: %f] " %
                  (epoch, opt.n_epoch, i + 1, len(train_loader), loss))
        train_loss += loss
    write.add_scalar('Train MSEloss', train_loss/train_b, epoch)
    #eval
    eval_loss = 0.
    count_batchsize = 0
    model.eval()
    for k, (test_input, test_label) in enumerate(test_loader):
        count_batchsize = count_batchsize + 1
        eval_y = model(test_input)
        loss_v = cost(eval_y, test_label)
        # if (k + 1) % 10 == 0:
        #     print("[Batch %d/%d] [MSEloss: %f] " % (k + 1, len(test_loader), loss_v))
        eval_loss = eval_loss + loss_v
    mse = eval_loss / count_batchsize
    rmse = torch.sqrt((eval_loss / count_batchsize))
    write.add_scalar('Test MSE', mse, epoch)
    write.add_scalar('Test RMSE', rmse, epoch)
    best_mse = 100
    if mse < best_mse:
        best_mse = mse
        model_path = opt.model_path+opt.dataset_type + '/' + opt.use_rnn_type
        if os.path.isdir(model_path):
            torch.save(model.state_dict(), model_path + '/' + 'best_model.pt')
        else:
            os.makedirs(model_path)
            torch.save(model.state_dict(), model_path +'/' + 'best_model.pt')
    # write.add_scalar('Test RMSE', rmse, epoch)
    print('Epoch: {} Test...'.format(epoch))
    print('Test mse: {}'.format(mse))

# tsbd = TensorBoard(log_dir='././logs/'+use_rnn_type,histogram_freq=0,write_graph=True,write_images=False)