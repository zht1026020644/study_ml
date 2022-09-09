# coding: utf-8 -*- coding: utf-
import numpy as np
import pandas as pd
from pyparsing import actions

'''
env定义、动作集a、状态集s、奖励集r,以及转移矩阵
'''
np.set_printoptions(suppress=True)

class QLearningTable(object):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # 动作集 list
        self.lr = learning_rate  # 学习率
        self.gamma = reward_decay  # # 汇报的衰减系数
        self.epsilon = e_greedy  # 贪婪系数
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)  # 定义q表

    def choose_action(self, observation):
        self.check_state_exist(observation)
        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[observation, :]
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q_table
            self.q_table = self.q_table.append(
                pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))
