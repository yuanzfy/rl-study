# -*- coding: utf-8 -*-


import numpy as np
from snake_env import SnakeEnv


class TableAgent(object):
    def __init__(self, env, policy):
        """
        env: gym环境
        policy: 迭代策略
        """
        self.env = env
        self.policy = policy

        self.s_len = env.observation_space.n  # 状态空间维度
        self.a_len = env.action_space.n  # 动作空间维度

        # 状态对应的奖励向量
        self.r = [env.reward(s) for s in range(0, self.s_len)]
        # 策略向量，agent优化目标，每个状态下所选择的动作
        self.pi = np.array([0 for s in range(0, self.s_len)])
        # 状态转移矩阵，维度为A * S * S，
        self.p = np.zeros([self.a_len, self.s_len, self.s_len], dtype=np.float)

        # np.vectorize把函数的数据输出都转成向量格式
        ladder_move = np.vectorize(lambda x: env.ladders[x] if x in env.ladders else x)

        # 生成状态转移矩阵
        for i, dice in enumerate(env.dices):
            prob = 1.0 / dice
            for src in range(1, 100):
                step = np.arange(dice)
                step += src
                step = np.piecewise(step, [step > 100, step <= 100], [lambda x: 200 - x, lambda x: x])
                step = ladder_move(step)
                for dst in step:
                    self.p[i, src, dst] += prob

        self.p[:, 100, 100] = 1  # 终点
        self.value_pi = np.zeros((self.s_len))  # 状态值函数(最优值)
        self.value_q = np.zeros((self.s_len, self.a_len))  # 状态-行动值函数(最优值)
        self.gamma = 0.8

    def play(self, state):
        """
        根据state给出下一步action
        """
        return self.pi[state]

    def learn(self):
        """
        学习
        """
        self.policy.iteration(self)

    def get_pi(self):
        """
        状态-动作映射
        """
        return list(self.pi[1:])
