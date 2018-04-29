# -*- coding: utf-8 -*-

import numpy as np
from enum import Enum,IntEnum

class OrderType(Enum):
    Buy = 1
    Wait = 0
    Sell = -1

class CurrencyType(IntEnum):
    USD=1
    GBP=2
    EUR=3
    CAD=4
    CHF=5
    SEK=6
    DKK=7
    NOK=8
    AUD=9
    NZD=10
    ZAR=11
    BHD=12
    HKD=13
    INR=14
    PHP=15
    SGD=16
    THB=17
    KWD=18
    SAR=19
    AED=20
    MXN=21
    IDR=22
    TWD=23

def spread(Exnum):
    USD=0.003
    GBP=0.01
    EUR=0.006
    CAD=0.017
    CHF=0.018
    AUD=0.007
    NZD=0.014
    ZAR=0.01
    other=0.01
    if Exnum == 1:
        return USD
    elif Exnum == 2:
        return GBP
    elif Exnum == 3:
        return EUR
    elif Exnum == 4:
        return CAD
    elif Exnum == 5:
        return CHF
    elif Exnum == 9:
        return AUD
    elif Exnum == 10:
        return NZD
    elif Exnum == 11:
        return ZAR
    else:
        return other


class HaveCurrency:
    def __init__(self,
                 currency_type,
                 order_type,
                 trade_unit,
                 order_rate,
                 profit_taking,
                 loss_cut,
                 ):
        self.have_count=0
        self.currency_type=currency_type
        self.trade_unit=trade_unit
        self.order_rate=order_rate
        self.order_type=order_type
        self.profit_taking=profit_taking
        self.loss_cut=loss_cut

    def confirm(self,rate):
        print ("%s"%self.currency_type.name,
               "have count:%d"%self.have_count,
               "have unit:%d"%self.trade_unit,
               "now value:%d"%self.now_value(rate=rate),
               "order rate:%f"%self.order_rate,
               "order type:%s"%self.order_type.name,
               "profit take:%f"%self.profit_taking,
               "loss cut:%f"%self.loss_cut)

    def now_value(self,rate):
        if self.order_type == OrderType.Buy:
            return rate.now_bid() * self.trade_unit
        if self.order_type == OrderType.Sell:
            return rate.now_ask() * self.trade_unit

    def next_day(self):
        self.have_count += 1

class MyState:
    def __init__(self,
                 Rate,
                 init_money=1000000):
        self.rate=Rate
        self.init_money=init_money
        self.now_money=init_money
        self.debt=0
        self.have_currency=[]
        self.assets=np.asarray([init_money])
        self.draw_down=init_money
        self.max_asset=init_money
        self.win_trade=0
        self.lose_trade=0
        self.total_profit=0
        self.total_loss=0

    def update_evaluation(self):
        if self.draw_down > self.now_asset:
            self.draw_down = self.now_asset
        elif self.max_asset < self.now_asset:
            self.max_asset = self.now_asset

    def confirm_now(self):
        self.now_asset = self.now_money
        for hc in self.have_currency:
            hc.confirm(self.rate)
            if hc.order_type == OrderType.Buy:
                self.now_asset += hc.now_value(self.rate)
            elif hc.order_type == OrderType.Sell:
                self.now_asset += ((hc.order_rate * hc.trade_unit) - hc.now_value(self.rate))
        print(("money:%d"%self.now_money),
              ("debt:%d"%(self.debt * self.rate.now_ask())),
              ("asset:%d"%self.now_asset),
              (("now day:%d/%d")%(self.rate.now,self.rate.size - 1)))
        self.assets = np.append(self.assets,self.now_asset)
        self.update_evaluation()

    def next_day(self):
        for hc in self.have_currency:
            hc.next_day()

class Rate:
    def __init__(self,
                 currency_type,
                 rating,
                 spread,
                 now=0):
        self.now=now
        self.currency_type=currency_type
        self.spread=spread
        self.ask=rating
        self.bid=rating - spread
        self.size=rating.size
        self.end=False
        self.ask_rates=np.asarray(self.ask[self.now])
        self.bid_rates=np.asarray(self.bid[self.now])

    def now_ask(self):
        return self.ask[self.now]

    def now_bid(self):
        return self.bid[self.now]

    def next_day(self,end_time=3268-1):
        if self.now < end_time:
            self.now += 1
            self.ask_rates = np.append(self.ask_rates,self.ask[self.now])
            self.bid_rates = np.append(self.bid_rates,self.bid[self.now])
        else:
            self.end = True
            print "Rate can't go to next"

    def now_rate(self):
        print (("%s, day:%d/%d, ask:%f, bid:%f")%(self.currency_type.name,self.now,self.size - 1,self.now_ask(),self.now_bid()))

class Order:
    def __init__(self,
                 MyState,
                 lot):
        self.my_state=MyState
        self.rate=MyState.rate
        self.lot=lot

    def buy_order(self,profit_taking,loss_cut):
        trade_unit = (self.my_state.now_money - (self.my_state.debt * self.rate.now_ask() * 2)) // (self.lot * self.rate.now_ask())
        if trade_unit > 0:
            self.my_state.now_money -= (trade_unit * self.lot * self.rate.now_ask())
            order_currency = HaveCurrency(currency_type=self.rate.currency_type,
                                          trade_unit=trade_unit * self.lot,
                                          order_type=OrderType.Buy,
                                          order_rate=self.rate.now_ask(),
                                          profit_taking=profit_taking,
                                          loss_cut=loss_cut)
            self.my_state.have_currency.append(order_currency)
            print "Successed buy order"
        else:
            print("you can't buy order")

    def sell_order(self,profit_taking,loss_cut):
        trade_unit = (self.my_state.now_money - (self.my_state.debt * self.rate.now_bid())) // (self.lot * self.rate.now_bid())
        if trade_unit > 0:
            self.my_state.debt += (trade_unit * self.lot)
            order_currency = HaveCurrency(currency_type=self.rate.currency_type,
                                          trade_unit=trade_unit * self.lot,
                                          order_type=OrderType.Sell,
                                          order_rate=self.rate.now_bid(),
                                          profit_taking=profit_taking,
                                          loss_cut=loss_cut)
            self.my_state.have_currency.append(order_currency)
            print "Successed sell order"
        else:
            print("you can't sell order")

    def trade_count(self,profit):
        if profit > 0:
            self.my_state.win_trade+=1
            self.my_state.total_profit+=profit
        else:
            self.my_state.lose_trade+=1
            self.my_state.total_loss-=profit

    def settlement(self):
        for hc in self.my_state.have_currency:
            if hc.order_type == OrderType.Buy:
                if self.rate.now_bid() >= hc.profit_taking or hc.have_count == 30:
                    self.my_state.now_money += (hc.trade_unit * self.rate.now_bid())
                    self.my_state.have_currency.remove(hc)
                    print "sell settlement is done"
                    profit=(self.rate.now_bid() - hc.order_rate) * hc.trade_unit
                    print "profit:%f"%(profit)
                    self.trade_count(profit)

            elif hc.order_type == OrderType.Sell:
                if self.rate.now_ask() <= hc.profit_taking or hc.have_count == 30:
                    self.my_state.now_money += ((hc.order_rate - self.rate.now_ask()) * hc.trade_unit)
                    self.my_state.debt -= hc.trade_unit
                    self.my_state.have_currency.remove(hc)
                    print "buy settlement is done"
                    profit=(hc.order_rate - self.rate.now_ask())*hc.trade_unit
                    print "profit:%f"%(profit)
                    self.trade_count(profit)
            else:
                print "having currency couldn't Buy or Sell"

def AIOrder(Order,profit_take):
    if Order.rate.now_ask() + (Order.rate.now_ask() * 0.01) < profit_take[Order.rate.now]:
        Order.buy_order(profit_taking=profit_take[Order.rate.now],loss_cut=Order.rate.now_ask() - 5)
    elif Order.rate.now_bid() - (Order.rate.now_bid() * 0.01) > profit_take[Order.rate.now]:
        Order.sell_order(profit_taking=profit_take[Order.rate.now],loss_cut=Order.rate.now_bid() + 5)
    else:
        print "Nothing Order"
