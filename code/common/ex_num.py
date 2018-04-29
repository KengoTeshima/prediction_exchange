from enum import Enum,IntEnum

class ExNumber(IntEnum):
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


