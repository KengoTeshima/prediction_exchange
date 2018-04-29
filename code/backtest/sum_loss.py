# -*- coding: utf-8 -*-

import numpy as np
from ex_num import ExNumber as en

sum_train=0
sum_test=0

with open("sum_lossSDAn.csv","w") as op:
    op.write("currency,train_loss,test_loss\n")
    for num in range(1,24):
        load_file = "results_SDAn/%s_loss.csv"%en(num).name

        loss = np.loadtxt(load_file,delimiter=",",skiprows=1)
        i_loss = np.argmin(loss[:,1])
        best_loss = loss[i_loss]

        op.write("%s,%f,%f\n" %(en(num).name, best_loss[0], best_loss[1]))

        sum_train += best_loss[0]
        sum_test += best_loss[1]

    op.write("sum,%f,%f\n"%(sum_train/23,sum_test/23))

