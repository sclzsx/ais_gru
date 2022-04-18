# -*- coding:utf-8 -*-
"""
作者：
日期：年月日
"""
import torch.nn as nn
from torch.autograd import Variable

#定义一个初始的函数
class lstm(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, seq_len, dropout_p=0.2, is_bi=False):
        super(lstm, self).__init__()
        # batch_size, seq_len, input_size(embedding_size)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.is_bi = is_bi
        self.dropout_p = dropout_p
        self.lstm1 = nn.LSTM(self.input_size, self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True,
                            dropout=self.dropout_p,
                            bidirectional=self.is_bi)
        self.fc_input_size = 2*self.hidden_size if self.is_bi else self.hidden_size
        # self.act = nn.LeakyReLU()
        # self.dropout = nn.Dropout(p=self.dropout_p)
        self.linear = nn.Linear(self.fc_input_size , self.output_size)

    def forward(self, x):
        # x: [batch_size, seq_len, input_size]
        output, (hidden, cell) = self.lstm1(x)
        # output = self.act(output)
        batch_size, seq_len, hidden_size = output.shape
        output = output.view(batch_size, seq_len, hidden_size)
        output = self.linear(output)
        # output = self.dropout(output)
        return output[:,-1,:].view(-1,self.output_size) # [batch_size, output_size]

class gru(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, seq_len, dropout_p=0.2, is_bi=False):
        super(gru, self).__init__()
        # batch_size, seq_len, input_size(embedding_size)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.is_bi = is_bi
        self.dropout_p = dropout_p
        # nn.GRUCell(3,5)
        #tensor [bc,5,5]
        self.gru1 = nn.GRU(self.input_size, self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True,
                            dropout=self.dropout_p,
                            bidirectional=self.is_bi)
        #print(self.gru1, '111111111111111')
        self.fc_input_size = 2*self.hidden_size if self.is_bi else self.hidden_size
        # self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(self.dropout_p)
        self.linear = nn.Linear(self.fc_input_size , self.output_size)

    def forward(self, x):
        # print(x.size())#[batch_size, seq_len, input_size])[batch_size, seq_len, input_size]
        output, b = self.gru1(x)
        output = self.dropout(output)
        output = self.linear(output)

        return output[:,-1,:].view(-1,self.output_size) # [batch_size, output_size]

