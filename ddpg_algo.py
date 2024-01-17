#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import math
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import ddpg_env
from hyps import get_args
# from state_normalization import StateNormalization
from state_normalization_mobile import StateNormalizationMobile

class Net(nn.Module):
    def __init__(self, n_actions, n_features, hidden_dim1=256, hidden_dim2=100, hidden_dim3=64):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_features, hidden_dim1)
        # self.fc1.weight.data.normal_(0, 0.1)

        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        # self.fc2.weight.data.normal_(0, 0.1)

        self.fc22 = nn.Linear(hidden_dim2, hidden_dim3)
        # self.fc22.weight.data.normal_(0, 0.1)

        self.fc3 = nn.Linear(hidden_dim3, hidden_dim2)
        # self.fc3.weight.data.normal_(0, 0.1)

        self.out = nn.Linear(hidden_dim2, n_actions)
        # self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc22(x))
        x = F.relu(self.fc3(x))
        # q_actions = F.softmax(self.out(x), dim=0)
        q_actions = F.relu(self.out(x))
        return q_actions


class DeepQNetwork:
    def __init__(self, n_actions, n_features, cfg):
        self.path = './model/{}_{}_(1)'.format('mobile_num', cfg['num'])
        # self.path = './model/{}_{}'.format('latency', '2')

        self.device = torch.device(cfg['device'])

        self.model = Net(n_actions, n_features)
        if False:
            # self.eval_net = torch.load(path)
            self.eval_net = self.model.to(self.device)
            self.eval_net.load_state_dict(torch.load(path))
            print("using %s" % cfg['model_name'])
        else:
            self.eval_net = self.model.to(self.device)
        self.target_net = self.model.to(self.device)
        # 复制eval网络参数到target网络
        for target_param, param in zip(self.target_net.parameters(), self.eval_net.parameters()):
            # zip(a,b)把a里面的每一个值和b里面的每一个值对应
            target_param.data.copy_(param.data)

        self.n_actions = n_actions
        self.n_features = n_features

        self.batch_size = cfg['batch_size']
        self.learn_step_counter = 0  # 统计执行step总数
        self.memory_capacity = cfg['memory_capacity']  # 经验池容量
        self.memory_counter = 0
        self.memory_counter1 = 0
        self.replace_target_iter = cfg['replace_target_iter']  # 复制网络参数的步数

        self.lr = cfg['lr']
        self.gamma = cfg['gamma']  # 折现因子
        self.epsilon = cfg['epsilon_start']
        self.epsilon_start = cfg['epsilon_start']
        self.epsilon_end = cfg['epsilon_end']
        self.epsilon_decay = cfg['epsilon_decay']

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_capacity, self.n_features * 2 + 2))  # 2指reward和a
        self.memory1 = np.zeros((self.memory_capacity, self.n_features * 2 + 2))  # 2指reward和a

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss()

    def store_transition(self, s, a, r, s_, update_action):
        transition = np.hstack((s, a, r, s_))
        # print("transition: ", transition)
        if update_action == 0:  # 原记忆池
            # replace the old memory with new memory
            index = self.memory_counter % self.memory_capacity
            self.memory[index, :] = transition
            self.memory_counter += 1
        else:
            # replace the old memory with new memory
            index = self.memory_counter1 % self.memory_capacity
            self.memory1[index, :] = transition
            self.memory_counter1 += 1

    def choose_action(self, state, update_action):
        state = state[np.newaxis, :]
        # state = torch.unsqueeze(torch.FloatTensor(state, device=self.device), 0)  # 使用GPU训练不能直接用FloatTensor传参
        state = torch.tensor(state, device=self.device)
        state = torch.unsqueeze(state.float(), 0)
        if np.random.uniform() > self.epsilon:
            # get q value for every actions
            actions_value = self.eval_net.forward(state)
            # print("self.out(x) shape: ", actions_value.shape)

            # print("actions_value of choosing: ", actions_value)
            if update_action == 0:
                # action = torch.max(actions_value, 2)[1].data.cpu().numpy()
                action = torch.argmax(actions_value).data.cpu().numpy()  # 使用cuda需要加上：.cpu()

            else:
                # print("actions: ", torch.argmax(actions_value[0, 0, :2187]).data.cpu().numpy())
                action = torch.argmax(actions_value[0, 0, :2187]).data.cpu().numpy()  # 使用cuda需要加上：.cpu()

                # print("choose action index: ", action)  # print的action数量代表使用eval_net预测action的次数
                # print("action value: ", actions_value[0, 0, action])
                # with open('actions_value.txt', 'a') as file_obj:  # 本episode结束记录结果
                #     file_obj.write("\n actions_value: %s\n" % str(actions_value))
                #     file_obj.write(" action: %s\n" % str(action))
        else:
            if update_action == 0:
                action = np.random.randint(0, self.n_actions)
            else:
                # actions_value = self.eval_net.forward(state)
                action = np.random.randint(0, 2187)
        return action

    def learn(self):
        # update the target_param
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            print("success replace!")

        # sample batch from memory
        if self.memory_counter > self.memory_capacity:
            sample_index = np.random.choice(self.memory_capacity, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        batch_state = torch.tensor(batch_memory[:, :self.n_features], device=self.device).float()
        batch_action = torch.tensor(batch_memory[:, self.n_features].astype(int), device=self.device).long()
        batch_reward = torch.tensor(batch_memory[:, self.n_features + 1], device=self.device).float()
        batch_next_state = torch.tensor(batch_memory[:, -self.n_features:], device=self.device).float()
        # q_eval
        q_eval = self.eval_net(batch_state).gather(dim=1, index=batch_action.view(self.batch_size, 1))  # [64, 1]
        q_next = self.target_net(batch_next_state).detach()
        # print(q_next.size())  # [64, 5619712]
        # 当前状态得到的reward加上下一状态预计q值的折现和
        # q_max = q_next.max(1)[0]
        q_target = batch_reward.view(self.batch_size, 1) + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)
        loss = self.loss_func(q_eval, q_target)
        # print("q eval: ", q_eval)
        # print("q target: ", q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_step_counter += 1
        # print("memory_counter: ", self.memory_counter)
        print("loss: ", loss)
        return loss.item()

    def learn1(self):
        # update the target_param
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            print("success replace!")

        # sample batch from memory1
        if self.memory_counter1 > self.memory_capacity:
            sample_index = np.random.choice(self.memory_capacity, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter1, size=self.batch_size)
        batch_memory = self.memory1[sample_index, :]

        batch_state = torch.tensor(batch_memory[:, :self.n_features], device=self.device).float()
        batch_action = torch.tensor(batch_memory[:, self.n_features].astype(int), device=self.device).long()
        batch_reward = torch.tensor(batch_memory[:, self.n_features + 1], device=self.device).float()
        batch_next_state = torch.tensor(batch_memory[:, -self.n_features:], device=self.device).float()
        # print("batch_state size: ", batch_state.shape)  # [64, 147]
        # print("batch_action size: ", batch_action.shape)  # [64]
        # print("batch_reward type of learning: ", type(batch_reward[12]))  # [64] tensor
        # print("batch_next_state size: ", batch_next_state.shape)  # [64, 147]

        # q_eval
        q_eval = self.eval_net(batch_state).gather(dim=1, index=batch_action.view(self.batch_size, 1))  # [64, 1]
        q_next = self.target_net(batch_next_state).detach()
        # print(q_next.size())  # [64, 5619712]
        # 当前状态得到的reward加上下一状态预计q值的折现和
        # q_max = q_next.max(1)[0]
        q_target = batch_reward.view(self.batch_size, 1) + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_step_counter += 1
        # print("memory_counter1: ", self.memory_counter1)
        return loss.item()


def smooth(data, weight=0.9):
    ''' 用于平滑曲线，类似于Tensorboard中的smooth曲线
    '''
    last = data[0]
    smoothed = []
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point  # 计算平滑值
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def plot_throughput(throughput_list, tag='train'):
    plt.figure()
    plt.title(f"{tag}")
    plt.xlabel('Episodes')
    # plt.plot(throughput_list, label=tag)
    smooth_th = smooth(throughput_list)
    plt.plot(smooth_th, label='smoothed')
    plt.legend()
    plt.show()
    plt.savefig(f'training {tag}')

def cal_episode_throughput(throughput_list):  # 计算每100轮的平均吞吐量
    list1 = throughput_list
    avg_episode_list = []
    while len(list1)!=0:
        num_sum = sum(list1[:100]) / 100
        avg_episode_list.append(num_sum)
        list1 = list1[100:]
    print(avg_episode_list)
    return avg_episode_list

def plot_throughput1(throughput_list, tag='train'):
    plt.figure()
    plt.title(f"{tag}")
    plt.xlabel('Episodes')
    x = []
    for i in range(len(throughput_list)):
        x.append(i*100)
    plt.plot(x, throughput_list, label=tag)
    plt.legend()
    plt.show()
    plt.savefig(f'training {tag}')

def cal_avg_throughput(ep_throughput_list, MAX_STEPS):
    avg_throughput = sum(ep_throughput_list) / MAX_STEPS
    return avg_throughput

# 训练
def train(env, DQN, cfg):
    Normal = StateNormalizationMobile()  # 输入状态归一化
    total_th_list = []  # 每轮的总吞吐量
    total_reward_list = []
    total_loss_list = []
    avg_throughput_list = []
    print("start training: ", time.time())
    for i in range(cfg['train_eps']):
        s = env.reset()
        ep_reward = 0  # 每轮的总reward
        ep_throughput = 0  # 每轮的总throughput
        ep_loss = 0
        ep_throughput_list = []  # 每轮的每步分别的吞吐量
        # ep_loss_list = []
        j = 0
        for j in range(cfg['ep_steps']):
            a = DQN.choose_action(Normal.state_normal(s), env.update_action)  # 使用归一化后的状态
            # print("action: ", a)
            s_, r = env.step(a, j)
            # print("cached: ", env.cached_model)
            ep_throughput_list = np.append(ep_throughput_list, env.th_last)
            # print("transition: ", s, a, r, s_)
            DQN.store_transition(Normal.state_normal(s), a, r, Normal.state_normal(s_), env.update_action)  # 保存归一化后的状态
            # epsilon指数衰减
            # DQN.epsilon = DQN.epsilon_end + (DQN.epsilon_start - DQN.epsilon_end) * \
            #               math.exp(-1. * DQN.learn_step_counter / DQN.epsilon_decay)
            if env.update_action == 0:
                if DQN.memory_counter > DQN.batch_size:  # 记忆池内的transition数量超过一个batch_size时开始学习
                    # epsilon指数衰减
                    DQN.epsilon = DQN.epsilon_end + (DQN.epsilon_start - DQN.epsilon_end) * \
                                  math.exp(-1. * DQN.learn_step_counter / DQN.epsilon_decay)
                    # t_start = time.time()  # 开始训练时间
                    loss = DQN.learn()  # 每个step都会训练得出一个loss
            else:
                if DQN.memory_counter1 > DQN.batch_size:  # 记忆池内的transition数量超过一个batch_size时开始学习
                    # epsilon指数衰减
                    DQN.epsilon = DQN.epsilon_end + (DQN.epsilon_start - DQN.epsilon_end) * \
                                  math.exp(-1. * DQN.learn_step_counter / DQN.epsilon_decay)
                    # t_start = time.time()  # 开始训练时间
                    loss = DQN.learn1()  # 每个step都会训练得出一个loss
                    ep_loss += loss
            s = s_
            ep_reward += r  # MAX_EP_STEPS步的reward总和（即每局游戏的总reward）
            ep_throughput += env.th_last
        avg_throughput = cal_avg_throughput(ep_throughput_list, cfg['ep_steps'])
        avg_throughput_list = np.append(avg_throughput_list, avg_throughput)
        total_reward_list = np.append(total_reward_list, ep_reward)
        total_th_list = np.append(total_th_list, ep_throughput)
        total_loss_list = np.append(total_loss_list, ep_loss)
        print('Episode:', i, ' Steps: %2d' % j, ' Reward: %7.2f' % ep_reward,
              'Throughput: %2d' % ep_throughput, 'Explore: %.3f' % DQN.epsilon)
        print("throughput_list: ", ep_throughput_list)

        file_name = cfg['train_result']
        with open(file_name, 'a') as file_obj:  # 本episode结束记录结果
            file_obj.write("\n======== This %d training episode is done. ========" % i)
            file_obj.write("\n throughput_list: %s" % str(ep_throughput_list))
            file_obj.write("\n total_throughput: %s" % str(ep_throughput))

    with open(file_name, 'a') as file_obj:  # 本episode结束记录结果
        file_obj.write("\n total throughput list: %s\n" % str(total_th_list))
    print("finish training: ", time.time())
    print("完成训练！")

    # 保存模型
    # torch.save(DQN.eval_net.state_dict(), DQN.path)
    print("Successfully save model!")

    return total_th_list, total_reward_list, total_loss_list, avg_throughput_list


# 测试
def test(env, cfg):
    Normal = StateNormalizationMobile()  # 输入状态归一化
    model = Net(env.n_actions, env.s_dim)
    eval_net = model.to(cfg['device'])
    eval_net.load_state_dict(torch.load(DQN.path))
    print('开始随机测试！')
    ep_reward_list = []  # 每轮总回报列表
    total_th_list = []  # 每轮总吞吐量列表
    measured_time = []
    for i in range(cfg['test_eps']):
        # ep_step = 0
        ep_reward = 0
        ep_throughput = 0
        s = env.reset()
        predict_t = []
        ep_throughput_list = []
        ep_time = 0
        for _ in range(cfg['ep_steps']):
            t0 = time.time()
            # 预测动作
            s = Normal.state_normal(s)
            s = s[np.newaxis, :]
            s = torch.tensor(s, device=cfg['device'])
            s = torch.unsqueeze(s.float(), 0)
            q_eval = eval_net(s)
            action = torch.argmax(q_eval).data.cpu().numpy()
            # action = np.random.randint(0, n_actions)
            # t1 = time.time()
            # t2 = t1 - t0
            # predict_t = np.append(predict_t, t2)  # 预测动作的时间
            s_, r = env.step(action, _)
            t1 = time.time()
            t2 = t1 - t0
            ep_time += t2
            ep_throughput_list = np.append(ep_throughput_list, env.th_last)
            ep_reward += r
            ep_throughput += env.th_last
            s = s_
        ep_reward_list = np.append(ep_reward_list, ep_reward)  # 每轮总回报列表
        total_th_list = np.append(total_th_list, ep_throughput)  # 每轮总吞吐量列表（最高40x20）
        measured_time = np.append(measured_time, ep_time)  # total slot的总执行时间

        # throughput = np.append(throughput, ep_throughput)
        print('Episode:', i, ' Steps: %2d' % _, ' Reward: %7.2f' % ep_reward,
              'Throughput: %2d' % ep_throughput)
        # print("predict_time: ", predict_t)
        print("throughput: ", ep_throughput_list)
        print("measured time: ", measured_time)
        file_name = cfg['test_result']
        with open(file_name, 'a') as file_obj:  # 本episode结束记录结果
            file_obj.write("\n======== This %d testing episode is done. ========" % i)
            file_obj.write("\n predict_time: %s" % str(predict_t))
            file_obj.write("\n throughput_list: %s" % str(ep_throughput_list))
    with open(file_name, 'a') as file_obj:  # 本episode结束记录结果
        file_obj.write("\n total throughput list: %s\n" % str(total_th_list))
    print("完成测试！")
    return total_th_list, measured_time


if __name__ == '__main__':
    cfg = get_args()
    env = dqn_env.DQNEnv(cfg)
    np.random.seed(cfg['seed'])
    s_dim = env.s_dim
    n_actions = env.n_actions
    DQN = DeepQNetwork(n_actions, s_dim, cfg)
    total_th_list, total_reward_list, total_loss_list, avg_throughput_list = train(env, DQN, cfg)
    print("throughput list: ", total_th_list)
    print("total_loss_list: ", total_loss_list)
    plot_throughput(total_reward_list, tag="reward")
    # plot_throughput1(cal_episode_throughput(avg_throughput_list), tag="average throughput")
    plot_throughput(total_loss_list, tag="loss")

    # total_th_list, measured_time = test(env, cfg)
    # print("throughput list: ", total_th_list)
    # # plot_throughput(total_th_list, tag="test")

    # np.save('./result/{}/test_{}_(1).npy'.format(cfg['VARI_TYPE'], cfg['num']), total_th_list)