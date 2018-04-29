import matplotlib.pyplot as plt
import numpy as np
from backtest import CurrencyType as ct
import math

for num in range(1,24):
    load_file = "resultsSDA5/%s_output.csv"%ct(num).name
    fig_file = "outSDA5/%s_result.eps"%ct(num).name
    title = "%s/JPY"%ct(num).name

    learn = np.loadtxt(load_file,delimiter=",",skiprows=1)
    ex_rate = np.loadtxt("quote.csv",delimiter=",",skiprows=1,usecols=(num,))
    learn = (learn * np.ptp(ex_rate)) + np.amin(ex_rate)

    output,teach = np.hsplit(learn,2)

    plt.plot(output,label="output")
    plt.plot(teach,label="teach")
    plt.legend(loc=3)
    plt.plot([2000,2000],[0,np.amax(ex_rate)],linewidth=2.5,linestyle="--")
    #plt.annotate("train <- | -> test",
    #         xy=(2000, np.amax(ex_rate)), xycoords='data',
    #         xytext=(+10, +30), textcoords='offset points', fontsize=20,
    #         color='red',arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    plt.title(title,fontsize=22)
    plt.xlabel("day",fontsize=20)
    plt.ylabel("yen",fontsize=20)
    plt.savefig(fig_file)
    plt.close()