# -*- coding: utf-8 -*-

import numpy as np


class ModelFreeAgent(object):
    def __init__(self, env, policy):
        self.env = env
        self.policy = policy

        self.s_len = env.observation_space.n
        self.a_len = env.action_space.n

        # 动作向量
        self.pi = np.array([0 for s in range(0, self.s_len)])
        # 状态-行动值函数
        self.value_q = np.zeros((self.s_len, self.a_len))
        # 状态-行动采样次数
        self.value_n = np.zeros((self.s_len, self.a_len))
        self.gamma = 0.8

    def play(self, state, epsilon=0.0):
        # 在执行阶段，不需要epsilon
        if np.random.rand() < epsilon:
            return np.random.randint(self.a_len)
        else:
            return self.pi[state]

    def learn(self):
        """
        学习
        """
        self.policy.iteration(self, self.env)

    def get_pi(self):
        """
        状态-动作映射
        """
        return list(self.pi[1:])
