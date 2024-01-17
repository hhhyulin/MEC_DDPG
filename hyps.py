#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="hyperparameters")
    parser.add_argument('--algo_name', default='DDPG', type=str, help='name of algorithm')
    parser.add_argument('--train_eps', default=200, type=int, help='episodes of training')
    parser.add_argument('--test_eps', default=100, type=int, help='episodes of testing')
    parser.add_argument('--ep_steps', default=400, type=int, help='steps per episode, much larger value can '
                                                                 'simulate infinite steps')
    parser.add_argument('--gamma', default=0.001, type=float, help='discounted factor')  #  0.001
    parser.add_argument('--epsilon_start', default=0.1, type=float, help='epsilon of action selection')
    parser.add_argument('--epsilon_decay', default=50000, type=int, help='discount rate for epsilon')
    parser.add_argument('--epsilon_end', default=0.1, type=float, help='final value of epsilon')
    parser.add_argument('--var', default=0.01, type=float, help='control exploration')
    parser.add_argument('--critic_lr', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--actor_lr', default=0.0002, type=float, help='learning rate')
    parser.add_argument('--memory_capacity', default=10000, type=int, help='memory capacity')
    parser.add_argument('--batch_size', default=64, type=int, help='training data size per step')
    parser.add_argument('--tau', default=200, type=int, help='update')
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--device', default='cuda', type=str, help='cpu or cuda')
    parser.add_argument('--seed', default=130, type=int, help='seed')
    parser.add_argument('--replace_target_iter', default=800, type=int, help='copy param from eval_param '
                                                                             'to target_param')
    parser.add_argument('--model_name', default='ddpg_model', type=str, help='name of model when training or testing')
    parser.add_argument('--train_result', default='train_result.txt', type=str, help='name of file when recoding'
                                                                                      'training results')
    parser.add_argument('--test_result', default='test_result.txt', type=str, help='name of file when recoding'
                                                                                    'testing results')
    parser.add_argument('--MAX_LATENCY', default=2, type=int)
    parser.add_argument('--VARI_TYPE', default='mobile_num', type=str, help='different types of experiment')
    parser.add_argument('--num', default=20, type=int)  # mobile num

    args = parser.parse_args([])
    args = {**vars(args)}
    return args
