import math
import random
import numpy as np

class BaseDevice:
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '_instance'):
            orig = super(BaseDevice, cls)
            cls._instance = orig.__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        self.loading_time = None
        self.solo_time = None
        self.interference = None

    def get_loading_time(self):
        return self.loading_time

    def get_solo_time(self):
        return self.solo_time

    def get_interference(self):
        return self.interference

    def get_instance(self): 
        return self

class Nano(BaseDevice):  # NVIDIA Jetson Nano
    # local
    def __init__(self):
        super().__init__()
        self.loading_time = [1.4401875638961792, 1.0951584815979003, 3.2074293661117554, 4.379654240608216,
                             7.446058874130249, 7.463520197868347, 7.754869656562805]
        self.solo_time = [4.345708990097046, 2.752386140823364, 3.9813384771347047, 3.993636071681976,
                          3.056140387058258, 3.262002551555633, 3.3621537923812865]
        self.interference = [0.3402885344293384, 0.8103660172886319, 1.1495774613486396, 1.7088506221771242,
                             2.406694934103224, 2.551536374621921, 3.150218531820509]

class Tx2(BaseDevice):  # NVIDIA Jetson Tx2
    def __init__(self):
        super().__init__()
        self.loading_time = [0.595761456489563, 0.7009875106811524, 3.303107500076294, 2.355398964881897,
                             7.261156735420227, 7.250781755447388, 7.516464762687683]
        self.solo_time = [0.8005391955375671, 0.6116957068443298, 0.6301509618759156, 0.6640055537223816,
                          0.5856670022010804, 0.6037039637565613, 0.6113351345062256]
        self.interference = [-0.21628718243704903, -0.05269568893644547, -0.037881456481085866, -0.07746853166156348,
                             0.32604961792627973, 0.3588355567720201, 0.3730367130703396]

class Xavier(BaseDevice):  # NVIDIA Xavier
    def __init__(self):
        super().__init__()
        self.loading_time = [0.2705937623977661, 0.3410001993179321, 0.6764320039749145, 1.015671625137329,
                             0.4037298583984375, 0.4089188194274902, 0.4378145790100097]
        self.solo_time = [0.5200192093849182, 0.4531603932380676, 0.5038563728332519, 0.3841231107711792,
                          0.3516596317291259, 0.3453783988952636, 0.3441032886505127]
        self.interference = [-0.017996739016638865, 0.04089776939815945, 0.03350325557920668, 0.015349580181969543,
                             0.6479825377464294, 0.6790465156237286, 0.697715171178182]

class Raspberry(BaseDevice):
    def __init__(self):
        super(Raspberry, self).__init__()
        self.solo_time = [1000, 1000, 1000, 1000, 1000, 1000, 1000]


class Trans:
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '_instance'):
            orig = super(Trans, cls)
            cls._instance = orig.__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        self.wire_trans_latency = 0.011921
        self.coef = math.pow(10, 0)

    def cal_trans_time(self, rate):
        if rate <= 0:
            return 10000
        return 158.62574521935713 / rate + 0.0036795665703229976

    def bandwidth2rate(self, bandwidth, coef=None):
        if coef is None:
            return bandwidth * self.coef
        return bandwidth * coef

    def rate2bandwidth(self, rate, coef=None):
        if coef is None:
            return rate / self.coef
        if coef == 0:
            raise Exception("coef is zero, can not be a divide")
        return rate / coef

    def bandwidth2trans_time(self, bandwidth, coef=None):
        if coef is None:
            return self.cal_trans_time(self.bandwidth2rate(bandwidth, self.coef))
        return self.cal_trans_time(self.bandwidth2rate(bandwidth, coef))

    def get_wire_trans_latency(self):
        return self.wire_trans_latency

    def get_instance(self):
        return self

def get_edge_idx():
    return random.randint(0, 1)

def get_edge_device(idx):
    if idx == 0:
        return Tx2().get_instance() 
    if idx == 1:
        return Xavier().get_instance() 
    raise Exception("idx<=1")

def get_all_edge_idx(edges_num):  # 0/1
    return np.random.randint(0, 2, (edges_num, 1))


