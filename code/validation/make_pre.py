# -*- coding: utf-8 -*-

import numpy as np
from model import SdA5
from chainer import cuda
from preprocessing import data
from common import ex_num

N_day = 60
M_day = 30
N = 2000
n_hidden=[1024,512,256,128,64]
corruption_levels=[0.03,0.03,0.03,0.03,0.03]

cuda.init(0)

# Prepare dataset
#データセットのロード
print ('load dataset')
ex = data.load_exchange_data()
# データの正規化
exchange = data.normalize(ex)

x_all = data.make_xdata(exchange, N_day, M_day)
x_train, x_test = np.vsplit(x_all, [N])
in_units = x_train.shape[1]  # 入力層のユニット数
train = [x_train,x_test]

teach = exchange[:, ex_num.ExNumber(1)].reshape(len(exchange[:, 0]), 1)
y_all = data.make_ydata(teach, N_day, M_day)
y_train, y_test = np.vsplit(y_all, [N])
test = [y_train,y_test]

sda_name= "SDA5"
rng = np.random.RandomState(1)
sda_train = SdA5.SDA(rng=rng,
                     data=train,
                     target=test,
                     n_inputs=in_units,
                     n_hidden=n_hidden,
                     corruption_levels=corruption_levels,
                     n_outputs=1,
                     gpu=0)

sda_train.pre_train(n_epoch=500,sda_name=sda_name)
