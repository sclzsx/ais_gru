# -*- coding:utf-8 -*-
"""
作者：
日期：2022年2月23日
描述：step2:把csv处理成npz，npz是mnist数据集产检格式，常用于Keras框架
"""
import pandas as pd
import numpy as np
# pull only specific columns out
# 'Timestamp', 'MMSI', 'Latitude', 'Longitude', 'SOG', 'COG'
fields = [0, 1, 2, 3, 4, 5]
n_rows = 30000              # Pulls this many rows of data, because all of it is too much
df = pd.read_csv('../dataset/raw/20w_capture.csv', skipinitialspace=True, usecols=fields, nrows=n_rows, encoding='gb2312')
df['Receivedtime（UTC+8）'] = pd.to_datetime(df['Receivedtime（UTC+8）'])
df.set_index('Receivedtime（UTC+8）',inplace=True)
data = df.resample('5T').mean()
data=data.interpolate(method="linear")
data=data.reset_index()

# # get rid of nan rows (in speed and course) - could just set to -1
print(data.head())


data['Receivedtime（UTC+8）'] = data['Receivedtime（UTC+8）'].astype(str)

# change dataframe to numpy array
df = data.values

# new number of rows and columns
n_rows, n_cols = df.shape

# sort by MMSI, then by time/date
df = df[np.lexsort((df[:, 0], df[:, 1]))]

# convert time to int
for i in range(n_rows):
    # p
    df[i][0] = int(df[i][0][8:10])*24 + int(df[i][0][11:13])*3600 + int(df[i][0][14:16])*60 + int(df[i][0][17:])

# create timedeltas
i = 0
while i in range(n_rows):
    end = False
    temp = []
    start = i
    try:
        while df[i+1][1] == df[i][1]:
            temp.append(df[i][0])
            i += 1
            end = True
    except: pass

    if end is True:
        temp.append(df[i][0])
        diff_array = np.diff(temp)

        df[start][0] = 0
        df[start+1:i+1, 0] = diff_array
    i += 1

np.savez('../dataset/raw/20w_new.npz', sorted_data=df)

print('----------------------------------------------------')
print('End of danish_pull_data.py')
print('----------------------------------------------------')