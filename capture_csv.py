# -*- coding:utf-8 -*-
"""
作者：
日期：2022年2月23日
描述：Step1:获取有价值的列重建csv
"""
import pandas as pd
import numpy as np

#获取有价值的列信息重建csv 第一步
fields = [3, 12, 13, 19, 21, 22]
n_rows = 20000              # Pulls this many rows of data, because all of it is too much
df = pd.read_csv('./dataset/raw/20w.csv', skipinitialspace=True, usecols=fields, nrows=n_rows, encoding='gb2312')
# print(df)
#交换顺序
cols = list(df)
# print(cols,'11111111')
cols.insert(0, cols.pop(cols.index('Receivedtime（UTC+8）')))
cols.insert(2, cols.pop(cols.index('Lat_d')))
cols.insert(3, cols.pop(cols.index('Lon_d')))
cols.insert(4, cols.pop(cols.index('Speed')))
# print(cols,'222222222')
df.loc[:, ['MMSI', 'Course', 'Speed', 'Receivedtime（UTC+8）', 'Lon_d', 'Lat_d']] = df[['Receivedtime（UTC+8）', 'MMSI', 'Lat_d', 'Lon_d', 'Speed', 'Course']].values
df.columns = cols
print(df,'==============')
df.to_csv('./dataset/raw/20w_capture.csv',encoding='gb2312',index=False)