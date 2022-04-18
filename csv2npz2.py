# -*- coding:utf-8 -*-
"""
作者：
日期：2022年2月23日
描述：step2:把csv处理成npz，npz是mnist数据集产检格式，常用于Keras框架
"""
import pandas as pd
import numpy as np

use_resample = 0

# pull only specific columns out
# 'Timestamp', 'MMSI', 'Latitude', 'Longitude', 'SOG', 'COG'
fields = [0, 1, 2, 3, 4, 5]
n_rows = 30000  # Pulls this many rows of data, because all of it is too much
df = pd.read_csv('./dataset/raw/20w_capture.csv', skipinitialspace=True, usecols=fields, nrows=n_rows,
                 encoding='gb2312')

df['Receivedtime（UTC+8）'] = pd.to_datetime(df['Receivedtime（UTC+8）'])
df.set_index('Receivedtime（UTC+8）', inplace=True)

if use_resample:
    data = df.resample('5T').mean()
    data = data.interpolate(method="linear")
else:
    data = df

data = data.reset_index()

# # get rid of nan rows (in speed and course) - could just set to -1
# print(data.head())

data['Receivedtime（UTC+8）'] = data['Receivedtime（UTC+8）'].astype(str)

# change dataframe to numpy array
df = data.values

df = df[np.lexsort((df[:, 0], df[:, 1]))]

# new number of rows and columns
n_rows, n_cols = df.shape

# sort by MMSI, then by time/date
# df = df[np.lexsort((df[:, 0], df[:, 1]))]

# convert time to int
for i in range(n_rows):
    # print(df[i][0], df[i][0][8:10], df[i][0][11:13], df[i][0][14:16], df[i][0][17:])
    df[i][0] = int(df[i][0][8:10]) * 24 * 3600 + int(df[i][0][11:13]) * 3600 + int(df[i][0][14:16]) * 60 + int(
        df[i][0][17:])

##################################################
use_speed = 0
use_time = 1
############################# 按船号分航迹
# 按船号划分子集，存到df2
df2 = []
tmp = []
for i in range(n_rows - 1):
    tmp.append(df[i, :])
    if df[i, 1] != df[i + 1, 1]:
        df2.append(tmp)
        tmp = []
tmp = []
for i in range(n_rows):
    if df[i, 1] == df[-1, 1]:
        tmp.append(df[i, :])
df2.append(tmp)
assert len(set(df[:, 1])) == len(df2)
# 遍历df2中的子集，若船号的信息少于10条，舍弃该船号，新数据为df3
df3 = []
for t in df2:
    if len(t) >= 10:
        df3.append(t)
#############################

df_new = []
if use_time:
    ########### 按时间分航迹
    for i in range(len(df3)):
        set_ship = df3[i]  # 当前船号的所有轨迹

        tmp_sail = []
        sails = []
        for j in range(len(set_ship) - 1):

            this_ = set_ship[j]

            tmp_sail.append(this_)

            next_ = set_ship[j + 1]

            t0 = this_[0]
            t1 = next_[0]

            s0 = this_[4]
            s1 = next_[4]


            assert t1 >= t0

            if (t1 - t0) > 3600 * 6:  # 下一个点的时间比当前时间大于6小时
                # if (s0 > 0.1 and s1 <= 0.1) or (s0 <= 0.1 and s1 > 0.1):
                sails.append(tmp_sail)
                tmp_sail = []

        sails2 = []
        for t in sails:
            if len(t) >= 10:
                sails2.append(t)

        # 验证
        for j in range(len(sails2)):
            set_sail = sails2[j]
            for k in range(len(set_sail) - 1):
                assert set_sail[k + 1][0] - set_sail[k][0] <= 3600 * 6
                # assert (set_sail[k + 1][4] > 0.1 and set_sail[k][4] <= 0.1) or (set_sail[k][4] <= 0.1 and set_sail[k + 1][4] > 0.1)
                print(set_sail[k][4], set_sail[k+1][4], set_sail[k][1], set_sail[k+1][1])
                df_new.append(set_sail[k])
            print('sssssssssssssssssssss a sail')
elif use_speed:
    ################ 按速度分航迹
    # sails3 = []
    # for j in range(len(sails2)):
    #     set_sail = sails2[j]
    #
    #     tmp_sail = []
    #     for k in range(len(set_sail) - 1):
    #         this_ = set_sail[k]
    #
    #         tmp_sail.append(this_)
    #
    #         next_ = set_sail[k + 1]
    #
    #         s0 = this_[4]
    #         s1 = next_[4]
    #         # print(s0, s1)
    #         if (s0 > 0.1 and s1 <= 0.1) or (s0 <= 0.1 and s1 > 0.1):
    #             # print(s0, s1)
    #             sails3.append(tmp_sail)
    #             tmp_sail = []
    #
    # sails4 = []
    # for t in sails3:
    #     if len(t) >= 10:
    #         sails4.append(t)
    #
    # # # 验证
    # for j in range(len(sails4)):
    #     set_sail = sails4[j]
    #     # print(len(set_sail))
    #
    #     for k in range(len(set_sail) - 1):
    #         this_ = set_sail[j]
    #         next_ = set_sail[j + 1]
    #
    #         s0 = this_[4]
    #         s1 = next_[4]
    #         # print(s0, s1)
    #
    #         assert set_sail[k + 1][0] - set_sail[k][0] <= 3600 * 6
    #         if not (s0 > 0.1 and s1 <= 0.1) or (s0 <= 0.1 and s1 > 0.1):
    #             pass
    #         else:
    #
    #             print(s0,s1)
    #             pass
    #
    #         # print(point)
    #
    # print(len(sails))

    # break

    # ########### 按速度分航迹 2
    #
    for i in range(len(df3)):
        set_ship = df3[i]  # 当前船号的所有轨迹

        tmp_sail = []
        sails = []
        for j in range(len(set_ship) - 1):

            this_ = set_ship[j]
            next_ = set_ship[j + 1]

            t0 = this_[0]
            t1 = next_[0]
            s0 = this_[4]
            s1 = next_[4]

            assert t1 >= t0

            if (s1 + s0) > 0:
                tmp_sail.append(this_)

            if (s0 <= 0.0 and s1 > 0.0) or (s1 <= 0.0 and s0 > 0.0):
                # print(s0, s1)
                # if (s0 > 0.1 and s1 <= 0.1) or (s0 <= 0.1 and s1 > 0.1):
                # print(len(tmp_sail))
                sails.append(tmp_sail)
                # print(tmp_sail[0][4])
                tmp_sail = []

        sails2 = []
        for sail in sails:
            diff = []
            for k in range(len(sail) - 1):
                diff.append(abs(sail[k][4] - sail[k+1][4]))
            diff_val = np.mean(diff)
            if diff_val >= 0.1:
                # print(diff_val)
                sails2.append(sail)
        # sails2 = sails

        # 验证
        for j in range(len(sails2)):
            set_sail = sails2[j]
            for k in range(len(set_sail) - 1):
                # assert set_sail[k + 1][0] - set_sail[k][0] <= 3600 * 6
                # assert set_sail[k][4] <= 0.0 and set_sail[k + 1][4] > 0.0
                print(set_sail[k][4], set_sail[k + 1][4], set_sail[k][1], set_sail[k + 1][1], set_sail[k][0], set_sail[k + 1][0])
                # print(set_sail[k], set_sail[k+1])
                df_new.append(set_sail[k])
            print('sssssssssssssssssssss a sail')
    # break
    #
    #             #
    #             # ########### 按时间分航迹2
    #             # df_new2 = []
    #             # tmp_sail = []
    #             # sails = []
    #             # for j in range(len(df_new) - 1):
    #             #
    #             #     this_ = df_new[j]
    #             #
    #             #     tmp_sail.append(this_)
    #             #
    #             #     next_ = df_new[j + 1]
    #             #
    #             #     t0 = this_[0]
    #             #     t1 = next_[0]
    #             #
    #             #     s0 = this_[4]
    #             #     s1 = next_[4]
    #             #
    #             #
    #             #     assert t1 >= t0
    #             #
    #             #     if (t1 - t0) > 3600 * 6:  # 下一个点的时间比当前时间大于6小时
    #             #         # if (s0 > 0.1 and s1 <= 0.1) or (s0 <= 0.1 and s1 > 0.1):
    #             #         sails.append(tmp_sail)
    #             #         tmp_sail = []
    #             #
    #             # sails2 = []
    #             # for t in sails:
    #             #     if len(t) >= 10:
    #             #         sails2.append(t)
    #             #
    #             # # 验证
    #             # for j in range(len(sails2)):
    #             #     set_sail = sails2[j]
    #             #     for k in range(len(set_sail) - 1):
    #             #         assert set_sail[k + 1][0] - set_sail[k][0] <= 3600 * 6
    #             #         # assert (set_sail[k + 1][4] > 0.1 and set_sail[k][4] <= 0.1) or (set_sail[k][4] <= 0.1 and set_sail[k + 1][4] > 0.1)
    #             #         # print(set_sail[k][4], set_sail[k+1][4], set_sail[k][1], set_sail[k+1][1])
    #             #         df_new2.append(set_sail[k])
    #             #     # print('ssssssssssssssssss')

########### 保存航迹
# print(len(df_new), len(df_new[0]))
if len(df_new) > 0:
    df = np.array(df_new)
    print(df.shape)
    pd.DataFrame(df).to_csv('df_new.csv', index=False, header=False)
    count_read = np.array(pd.read_csv('df_new.csv', header=None))
##############################################################################


# create timedeltas
i = 0
while i in range(n_rows):
    end = False
    temp = []
    start = i
    try:
        while df[i + 1][1] == df[i][1]:
            temp.append(df[i][0])
            i += 1
            end = True
    except:
        pass

    if end is True:
        temp.append(df[i][0])
        diff_array = np.diff(temp)

        df[start][0] = 0
        df[start + 1:i + 1, 0] = diff_array
    i += 1

np.savez('./dataset/raw/20w_new.npz', sorted_data=df)

print('----------------------------------------------------')
print('End of danish_pull_data.py')
print('----------------------------------------------------')
