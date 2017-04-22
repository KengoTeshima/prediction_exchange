#!/usr/bin/env python
# coding: utf-8

import numpy as np

#データセットから入力データを作成する(N日分)
def make_xdata(exchange,N_day,M_day):
    x_pri = exchange[:-M_day]
    for i in range(0, x_pri.shape[0] - N_day + 1):
        for j in range(0, N_day):
            if j == 0:
                a = x_pri[i]
            else:
                a = np.hstack((a,x_pri[i + j]))
        if i == 0:
            x_data = a
        else:
            x_data = np.vstack((x_data,a))
    return x_data.astype(np.float32)

#データセットから教師データを作成する(M日の移動平均)
def make_ydata(exchange,N_day,M_day):
    y_pri = exchange[N_day:]
    y_data = y_pri[:- M_day + 1]
    for i in range(0, y_data.shape[0] - M_day + 1):
        for j in range(y_data.shape[1]):
            m_sum = 0
            for k in range(0,M_day):
                m_sum += y_pri[i+k][j]
            y_data[i][j] = m_sum / M_day
    return y_data.astype(np.float32)

#データセットをロードする
def load_exchange_data():
    exchange = np.loadtxt("quote.csv",delimiter=",",skiprows=1,usecols=(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23))
    return exchange.astype(np.float32)

def load_teach_data(country):
    file_name = "csv/%s.csv" % country
    teacher = np.loadtxt(file_name,delimiter=",")
    return teacher.astype(np.float32)

#国ごとにデータの正規化
def normalize(exchange):
    #転置行列の作成
    t = exchange.T
    #列のサイズを求める
    N = t.shape[0]
    #正規化（転置行列の行文繰り返す）
    for i in range(0,N):
        t[i] = (t[i] - np.amin(t[i])) / np.ptp(t[i])
    #転置を戻してリターン
    return t.T

#正規化したデータを復元
def restore(y_restore,y_data):
    #転置行列の作成
    y_t = y_data.T
    t_t = y_restore.T
    #列のサイズを求める
    N = t_t.shape[0]
    #復元（転置行列の行文繰り返す）
    for i in range(0,N):
        y_t[i] = (y_t[i] * np.ptp(t_t[i])) + np.amin(t_t[i])
    return y_t.T

