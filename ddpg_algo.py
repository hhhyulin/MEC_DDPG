#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from hyps import get_args
import ddpg_env
from state_normalization import StateNormalization


class Actor(nn.Module):
    def __init__(self, action_dim, n_features, hidden_dim1=256, hidden_dim2=100, hidden_dim3=64, init_w=3e-3):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(n_features, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.out = nn.Linear(hidden_dim2, action_dim)

        self.out.weight.data.uniform_(-init_w, init_w)
        self.out.bias.data.uniform_(-init_w, init_w)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.out(x))
        return x


class Critic(nn.Module):
    def __init__(self, action_dim, n_features, hidden_dim1=256, hidden_dim2=100, hidden_dim3=64, init_w=3e-3):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(n_features + action_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.out = nn.Linear(hidden_dim2, 1)

        self.out.weight.data.uniform_(-init_w, init_w)
        self.out.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.out(x))
        return x


class DDPGNetwork:
    def __init__(self, action_dim, n_features, cfg):
        self.path_critic = './model/critic/{}_{}'.format(cfg['VARI_TYPE'], cfg['num'])
        self.path_actor = './model/actor/{}_{}'.format(cfg['VARI_TYPE'], cfg['num'])

        self.device = torch.device(cfg['device'])

        self.models = {"actor": Actor(action_dim, n_features), "critic": Critic(action_dim, n_features)}

        self.critic = self.models['critic'].to(self.device)
        self.actor = self.models['actor'].to(self.device)
        if False:
            self.critic.load_state_dict(torch.load(self.path_critic))
            self.actor.load_state_dict(torch.load(self.path_actor))
        self.target_critic = self.models['critic'].to(self.device)
        self.target_actor = self.models['actor'].to(self.device)

        # copy
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        self.action_dim = action_dim
        self.n_features = n_features

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg['critic_lr'])
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg['actor_lr'])

        self.memory_capacity = cfg['memory_capacity']
        self.memory = np.zeros((self.memory_capacity, self.n_features * 2 + self.action_dim + 1), dtype=np.float32)
        self.memory_counter = 0

        self.batch_size = cfg['batch_size']
        self.gamma = cfg['gamma']
        self.tau = cfg['tau']  # soft update
        self.epsilon = 1.0
        self.bound = 1

    def learn(self):
        if self.memory_counter < self.batch_size:
            return
        elif self.memory_counter > self.memory_capacity:
            sample_index = np.random.choice(self.memory_capacity, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        batch_state = torch.tensor(batch_memory[:, :self.n_features], device=self.device).float()
        batch_action = torch.tensor(batch_memory[:, self.n_features: self.n_features + self.action_dim],
                                    device=self.device).float()
        batch_reward = torch.tensor(batch_memory[:, self.n_features + self.action_dim + 1],
                                    device=self.device).float()
        batch_next_state = torch.tensor(batch_memory[:, -self.n_features:], device=self.device).float()

        actor_loss = self.critic(batch_state, self.actor(batch_state))
        actor_loss = -actor_loss.mean()

        batch_next_action = self.target_actor(batch_next_state)
        target_value = self.target_critic(batch_next_state, batch_next_action.detach())

        expected_value = batch_reward.view(self.batch_size, 1) + self.gamma * target_value.view(self.batch_size, 1)
        expected_value = torch.clamp(expected_value, -np.inf, np.inf)

        actual_value = self.critic(batch_state, batch_action)
        critic_loss = nn.MSELoss()(actual_value, expected_value.detach())

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # update the target network parameters
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

    def choose_action(self, state):
        state = state[np.newaxis, :]
        state = torch.tensor(state, device=self.device)
        state = torch.unsqueeze(state.float(), 0)
        action = self.actor(state)
        return action.detach().cpu().numpy()[0, 0]

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, r, s_))
        index = self.memory_counter % self.memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1


def OU(action, mu=0, theta=0.15, sigma=0.2):
    return theta * (mu - action) + sigma * np.random.randn(1)


def train(env, DDPG, cfg):
    Normal = StateNormalization()
    total_th_list = []
    total_reward_list = []
    var = cfg['var']
    a_bound = env.action_bound
    print("Start training: ", time.time())
    for i in range(cfg['train_eps']):
        s = env.reset()
        ep_reward = 0
        ep_throughput = 0
        ep_throughput_list = []
        j = 0
        for j in range(cfg['ep_steps']):
            a = DDPG.choose_action(Normal.state_normal(s))
            noise = OU(a)
            a = np.clip(a + noise, -1, 1)
            a = np.abs(a)
            a = np.clip(np.random.normal(a, var), *a_bound)
            s_, r = env.step(a, j)
            ep_throughput_list = np.append(ep_throughput_list, env.th_last)
            DDPG.store_transition(Normal.state_normal(s), a, r, Normal.state_normal(s_))
            DDPG.learn()

            s = s_
            ep_reward += r
            ep_throughput += env.th_last
        total_reward_list = np.append(total_reward_list, ep_reward)
        total_th_list = np.append(total_th_list, ep_throughput)
        print('Episode:', i, ' Steps: %2d' % j, ' Reward: %7.2f' % ep_reward,
              'Throughput: %2d' % ep_throughput)

        file_name = cfg['train_result']
        with open(file_name, 'a') as file_obj: 
            file_obj.write("\n======== This %d training episode is done. ========" % i)
            file_obj.write("\n throughput_list: %s" % str(ep_throughput_list))
            file_obj.write("\n total_throughput: %s" % str(ep_throughput))

    with open(file_name, 'a') as file_obj:
        file_obj.write("\n total throughput list: %s\n" % str(total_th_list))
    print("finish training time: ", time.time())
    print("finish training！")

    torch.save(DDPG.critic.state_dict(), DDPG.path_critic)
    torch.save(DDPG.actor.state_dict(), DDPG.path_actor)
    print("Successful save model!")

    return total_th_list


if __name__ == '__main__':
    cfg = get_args()
    env = ddpg_env.DDPGEnv(cfg)
    np.random.seed(cfg['seed'])
    s_dim = env.s_dim
    action_dim = env.action_dim
    DDPG = DDPGNetwork(action_dim, s_dim, cfg)
    total_th_list = train(env, DDPG, cfg)