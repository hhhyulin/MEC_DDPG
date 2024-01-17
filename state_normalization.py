import numpy as np
import ddpg_env

class StateNormalization(object):
    def __init__(self):
        self.high_state = np.array(np.full(21, 1))  # cached_model
        self.high_state = np.append(self.high_state, np.full(3, 5200))  # R_s
        self.high_state = np.append(self.high_state, np.full(20, 6))  # local_model
        self.high_state = np.append(self.high_state, np.full(3, 50))  # C_s
        self.high_state = np.append(self.high_state, np.full(20, 6))  # task_list
        # self.high_state = np.append(self.high_state, np.full(20, 2))  # latency
        self.high_state = np.append(self.high_state, np.full(20, 50))  # D_m
        self.high_state = np.append(self.high_state, np.full(20, 3))  # U
        self.high_state = np.append(self.high_state, np.full(20, 15*106))  # B

        self.low_state = np.zeros(127)
        # self.low_state[21+3+20+3+20: 21+3+20+3+20+20] = np.full(20, 2)
        # print(self.low_state[66:68])
        # print(self.low_state.shape)

    def state_normal(self, state):
        # state[21+3+20+3+20: 21+3+20+3+20+20] -= 2
        res = state / (self.high_state - self.low_state)
        return res