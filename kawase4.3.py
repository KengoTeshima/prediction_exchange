# -*- coding: utf-8 -*-

import numpy as np
import SdA5
from chainer import cuda,serializers,Variable
import csv
import data
import ex_num

N_day = 60
M_day = 30
N = 2000

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

for a in range(1,24):
    print("trading %s"%ex_num.ExNumber(a).name)
    loss_file = "results/%s_loss.csv"%ex_num.ExNumber(a).name
    output_file = "results/%s_output.csv"%ex_num.ExNumber(a).name

    teach = exchange[:,ex_num.ExNumber(a)-1].reshape(len(exchange[:,0]),1)
    y_all = data.make_ydata(teach, N_day, M_day)
    y_train, y_test = np.vsplit(y_all, [N])
    test = [y_train,y_test]

    rng = np.random.RandomState(1)
    sda_train = SdA5.SDA(rng=rng,
                        data=train,
                        target=test,
                        n_inputs=in_units,
                        n_hidden=[1024,512,256,128,64],
                        corruption_levels=[0.1,0.1,0.1,0.1,0.1],
                        n_outputs=1,
                        gpu=0)

    #sda_train.pre_train(n_epoch=1000)
    serializers.load_hdf5('SDA5.model',sda_train.model)
    serializers.load_hdf5('SDA5.state',sda_train.optimizer)
    train_losses,test_losses = sda_train.fine_tune(n_epoch=10000)

    with open(loss_file,"w") as op:
        op.write("train_loss,test_loss\n")
        for i in range(0,len(train_losses)):
            op.write("%f,%f\n"%(train_losses[i],test_losses[i]))

    with open(output_file,"w") as r:
        r.write("output,teach\n")
        out = sda_train.forward(cuda.cupy.asarray(x_all),y_all,train=False,output=True)
        for i in range(0,x_all.shape[0]):
            r.write("%f,%f\n"%(out.data[i],y_all[i]))
