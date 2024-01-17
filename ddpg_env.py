import csv
import math
import random
import numpy as np
from simulateRealEnv import *

# -------------------------- #
# environment
# -------------------------- #
M = 20  # mobile device
N = 7  # model
S = 3  # edge server

# -------------------------- #
# model
# -------------------------- #
model_name = ["ResNet34", "ResNet50", "ResNet101", "ResNet152", "VGG11", "VGG13", "VGG16"]  # model name

model_size = [83.28613186, 97.79685879, 170.5429068, 230.4920187, 506.8408499, 507.5460196, 527.8030233]  # model size

model_num = len(Raspberry().get_instance().get_solo_time())

inference_time_edge = [0.00904020357131958, 0.011850108623504638, 0.019543111324310303,
                       0.029253121852874757, 0.006320668697357178, 0.006898420333862305, 0.009620986461639404]

workload = [3.671, 4.111, 7.833, 11.558, 7.616, 11.320, 15.483]  # G flops

# xxx
interface = [0.004923200425313544, 0.005820645389037619, 0.00626463663519147, 0.006376227712114536,
             0.006415996596729947, 0.007335975865960702, 0.006857571519806857]
solo_time = [0.004117003146006036, 0.006029463234467019, 0.013278474689118833, 0.02287689414076022,
             -9.532789937276905e-05, -0.0004375555320983975, 0.0027634149418325465]
loading_time_edge = [0.0819, 0.1030, 0.1942, 0.2884, 0.4053, 0.3705, 0.3832]


interference_loc = [4.345708990097046, 2.752386140823364, 3.9813384771347047, 3.993636071681976,
                    3.056140387058258, 3.262002551555633, 3.3621537923812865]

action_space = np.array(np.arange(3 ** 7))
n_actions = len(action_space)


class DDPGEnv(object):
    M = 20
    N = 7  
    S = 3 


    s_dim = 127   # 52 77 102 127 152(mobile num)
    action_dim = S * 3 + M
    action_bound = [-1, 1]

    action_space = np.array(np.arange(3 ** 7))
    n_actions = len(action_space)

    r_n = [83.28613186, 97.79685879, 170.5429068, 230.4920187, 506.8408499, 507.5460196, 527.8030233]  # size
    c_n = [3.671, 4.111, 7.833, 11.558, 7.616, 11.320, 15.483]  # computing resource
    wire_trans_latency = 0.011921 

    simu_trans = Trans()

    with open("train_data.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
    U_m = np.array(list(map(int, lines[2].split(',')))[:M])  # connect edge
    D_m = np.array(list(map(float, lines[5].split(',')))[:M])  # distance
    local_model = np.array(list(map(int, lines[8].split(',')))[:M])  # place model
    edges_idx = np.array(list(map(int, lines[47].split(','))))

    connect_num = [0 for _ in range(S)] 

    T_loading_s = np.zeros((S, N))
    T_solo_s = np.zeros((S, N))
    T_interference_s = np.zeros((S, N))
    for i in range(S):
        T_loading_s[i, :] = get_edge_device(edges_idx[i]).get_loading_time()
        T_solo_s[i, :] = get_edge_device(edges_idx[i]).get_solo_time()
        T_interference_s[i, :] = get_edge_device(edges_idx[i]).get_solo_time()

    rou_s_n = 0.05 + 0.05 * np.random.rand(S, N) 
    T_extra = 0.3 + 0.2 * np.random.rand(S, M) 
    trans_power = np.full(M, math.pow(10, 2.7) / 1000)
    datasize = np.full(M, 224 * 224 * 3) 
    G_m = np.zeros(M)
    for i in range(M):
        G_m[i] = math.pow(D_m[i], -4) * 1 ** 2

    cached_model = np.array([[0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0]])
    R_s = None
    C_s = None
    C_s_start = None

    B_s = np.full((S, 1), 15 * math.pow(10, 6))  # bandwidth

    task_list = None
    task_latency = None
    B_m = [0 for _ in range(M)]

    caching = {i: [] for i in range(S)}
    caching_remain_t = {i: [] for i in range(S)}  # remain time

    time_delay = []

    reward_list = np.zeros((1, M)).astype('float64')

    th_last = 0
    th_batch_time = 0

    update_action = 1

    csv_reader = list(csv.reader(open("task_list.csv")))

    def __init__(self, cfg):
        # ------------state-------------- #
        # cached_model, caching resource, computing resource,
        # task type, task max_latency, distance, connection, bandwidth
        self.start_state = np.append(self.cached_model, self.R_s)
        self.start_state = np.append(self.start_state, self.local_model)
        self.start_state = np.append(self.start_state, self.C_s)
        self.start_state = np.append(self.start_state, self.task_list)
        self.start_state = np.append(self.start_state, self.D_m)
        self.start_state = np.append(self.start_state, self.U_m)
        self.start_state = np.append(self.start_state, self.B_m)
        self.state = self.start_state
        self.cfg = cfg

    def reset(self):
        self.gen_task(0) 
        self.reset_env1()
        self.reset_env2()
        return self._get_obs()

    def _get_obs(self):
        self.state = np.append(self.cached_model, self.R_s)
        self.state = np.append(self.state, self.local_model)
        self.state = np.append(self.state, self.C_s)
        self.state = np.append(self.state, self.task_list)
        self.state = np.append(self.state, self.D_m)
        self.state = np.append(self.state, self.U_m)
        self.state = np.append(self.state, self.B_m)
        return self.state

    def gen_task(self, t):
        MAX_LATENCY = self.cfg['MAX_LATENCY']
        self.task_list = list(map(int, self.csv_reader[t]))  # [:self.M]
        self.task_latency = np.full(self.M, MAX_LATENCY)

    def get_bandwidth_mean(self):
        self.connect_num = [0 for _ in range(S)]
        for mobile in range(self.M):
            for edge in range(self.S):
                if self.U_m[mobile] == edge:
                    self.connect_num[edge] += 1
        self.B_m = [0 for _ in range(self.M)]
        for mobile in range(self.M):
            i = int(self.U_m[mobile])
            self.B_m[mobile] = self.B_s[i] / self.connect_num[i]

    def modify_bandwidth(self):
        self.B_m = [0 for _ in range(self.M)]
        for mobile in range(self.M):
            i = int(self.U_m[mobile])
            # print(self.connect_num)
            if self.connect_num[i] != 0:
                self.B_m[mobile] = self.B_s[i] / self.connect_num[i]
            else:
                pass

    def reset_env1(self):
        with open("train_data.txt", "r", encoding="utf-8") as f:
            lines = f.readlines()
        self.R_s = np.array(list(map(float, lines[44].split(','))))  # storage resources
        self.C_s = np.array(list(map(float, lines[26].split(','))))  # computing resources
        self.C_s_start = self.C_s.copy()
        self.cached_model = [[0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0]]

    def reset_env2(self):
        self.get_bandwidth_mean()

    def reset_env3(self):
        with open("train_data.txt", "r", encoding="utf-8") as f:
            lines = f.readlines()
        self.C_s = np.array(list(map(float, lines[26].split(','))))  # computing resources
        self.C_s_start = self.C_s.copy()

    def get_transrate(self):
        transrate = np.zeros((self.M, 1))
        for j in range(M):
            transrate[j] = self.simu_trans.bandwidth2rate(self.B_m[j])
        return transrate

    def get_transtime(self, transrate):
        transtime = np.full((self.M, self.S), 100000.00000000)
        for j in range(self.M):
            for i in range(self.S):
                if transrate[j] > 0:
                    transtime[j][i] = 158.62574521935713 / transrate[j] + 0.0036795665703229976
                    if self.U_m[j] != i + 1:
                        transtime[j][i] += self.wire_trans_latency
        return transtime

    def get_com_time(self, transtime, mobile, batch_size, offload_act):
        ii = int(self.task_list[mobile])
        jj = offload_act[ii]
        time_total = transtime[mobile][jj] + (self.T_interference_s[jj][ii] * batch_size[jj][ii]) + \
                    self.rou_s_n[jj][ii] + self.T_extra[jj][mobile]
        return time_total

    def step(self, action, slot):
        self.reward_list = np.zeros((1, M)).astype('float32')
        total_reward = 0 
        total_throughput = 0

        caching_act = [round(action[i] * self.N) for i in range(self.S * 3)]
        offload_act = [round(action[self.S * 3 + i] * self.S) for i in range(self.M)]

        for edge in range(self.S):
            for j in range(3):
                i = caching_act[j+(edge*3)]
                if i >= self.N or i < 0:
                    pass
                elif self.cached_model[edge][i] == 1 and i in self.caching[edge]:
                    pass
                elif self.cached_model[edge][i] == 0 and self.R_s[edge] >= self.r_n[i]:  
                    self.caching[edge].append(i)
                    temp = self.T_loading_s[edge][i]
                    self.caching_remain_t[edge].append(temp)

                    self.R_s[edge] -= self.r_n[i]
                else:
                    pass


        batch_size = np.zeros((self.S, self.N)) 
        batch_time = {i: 0 for i in range(M)}
        # time_total = {i: [10000] for i in range()}
        for mobile in range(self.M):
            i = int(self.task_list[mobile])
            j = offload_act[i]
            if j > self.S or j < 0:
                batch_time[mobile] = 10
            elif 0 <= j < self.S:
                if self.cached_model[j][i] == 1 and self.C_s[j] >= self.c_n[i]:
                    batch_size[j][i] += 1 
                    self.C_s[j] -= self.c_n[i]
                else:
                    connect_edge = self.U_m[mobile]
                    self.connect_num[connect_edge] -= 1
                    if self.local_model[mobile] == i:
                        batch_time[mobile] = interference_loc[i]
                    else:
                        batch_time[mobile] = 10
            elif j == self.S and self.local_model[mobile] == i:
                batch_time[mobile] = interference_loc[i]
            else:
                batch_time[mobile] = 10

        self.modify_bandwidth()
        transrate = self.get_transrate()
        transtime = self.get_transtime(transrate)

        for mobile in range(self.M):
            if batch_time[mobile] == 0:
                batch_time[mobile] = self.get_com_time(transtime, mobile, batch_size, offload_act)
            if batch_time[mobile] <= self.task_latency[mobile]:
                total_throughput += 1
            else:
                pass

        for edge in range(self.S):
            m = 0
            for i in range(len(self.caching[edge])):
                self.caching_remain_t[edge][m] -= 0.5
                if self.caching_remain_t[edge][m] <= 0:
                    j = self.caching[edge][m]
                    self.cached_model[edge][j] = 1
                    self.caching[edge].remove(self.caching[edge][m])
                    self.caching_remain_t[edge].remove(self.caching_remain_t[edge][m])
                else:
                    m += 1

        total_reward = (total_throughput + self.th_last)/2
        self.th_last = total_throughput
        self.th_batch_time = sum(batch_time.values())

        self.reset_env2() 
        self.reset_env3() 
        self.gen_task(slot + 1) 
        self._get_obs()  

        return self._get_obs(), total_reward
