# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import backtest
import seaborn as sns
from backtest import Rate,CurrencyType, MyState,Order,AIOrder

ini_money = 10000000
lot = 1000

with open("evaluate.csv","w") as ev:
    ev.write("Currency,last_assets,max_draw_down,max_assets,win_trade,lose_trade,total_profit,total_loss,PF\n")
    for num in range(1,24):
        load_file = "results_SDAn/%s_output.csv"%CurrencyType(num).name
        asset_file = "assetV/%s_asset.csv"%CurrencyType(num).name
        fig_file = "assetV/%s_assets.eps"%CurrencyType(num).name

        ex_rate = np.loadtxt("quote.csv",delimiter=",",skiprows=1,usecols=(num,))
        learn = np.loadtxt(load_file,delimiter=",",skiprows=1)

        spread = backtest.spread(num)

        #復元（転置行列の行文繰り返す）
        learn = (learn * np.ptp(ex_rate)) + np.amin(ex_rate)

        out = learn[:,0]
        teach = learn[:,1]

        rating = ex_rate[29:-30]

        rate = Rate(currency_type=CurrencyType(num),
                   rating=rating,
                   spread=spread,
                   now=2000)
        my_state = MyState(Rate=rate,init_money=ini_money)
        order = Order(MyState=my_state,
                      lot=lot)

        while rate.end == False:
            order.settlement()

            AIOrder(Order=order, profit_take=out)

            my_state.confirm_now()
            rate.now_rate()

            rate.next_day(end_time=rate.ask.size-1)
            my_state.next_day()

            print "---------------------------------------------------------------------------------------"

        order.settlement()
        my_state.confirm_now()

        ev.write("%s,%d,%d,%d,%d,%d,%d,%d,%f\n"%(CurrencyType(num).name,my_state.now_asset,my_state.draw_down,my_state.max_asset,my_state.win_trade,my_state.lose_trade,my_state.total_profit,my_state.total_loss,my_state.total_profit/my_state.total_loss))

        np.savetxt(asset_file,my_state.assets,delimiter=",")

        print(np.min(my_state.assets))


        plt.plot(my_state.assets/100000,label=CurrencyType(num).name)
        plt.title("total assets",fontsize=20)
        plt.xlabel("day",fontsize=20)
        plt.ylabel("assets percentage",fontsize=20)
        plt.savefig(fig_file)
        plt.close()