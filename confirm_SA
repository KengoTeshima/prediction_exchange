# -*- coding: utf-8 -*-

import numpy as np
import SdA
from chainer import cuda,serializers,Variable
import data
import ex_num

N_day = 30
M_day = 30
N = 2000
hidden=[128,64,32]
corruption_levels=[0.0,0.0,0.0]

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

teach = exchange[:,ex_num.ExNumber(1)].reshape(len(exchange[:,0]),1)
y_all = data.make_ydata(teach, N_day, M_day)
y_train, y_test = np.vsplit(y_all, [N])
test = [y_train,y_test]

for i in [1,2,4]:
    SA_file = "SA%d.model" % i
    n_hidden = [x*i for x in hidden]

    rng = np.random.RandomState(1)
    sda_train = SdA.SDA(rng=rng,
                        data=train,
                        target=test,
                        n_inputs=in_units,
                        n_hidden=n_hidden,
                        corruption_levels=corruption_levels,
                        n_outputs=1,
                        gpu=0)

    serializers.load_hdf5(SA_file,sda_train.model)
    loss = sda_train.forward(x_test, x_test, train=False)
    print (loss)
