# -*- coding: utf-8 -*-

import copy
import numpy as np
import gym
from gym.spaces import Discrete


class SnakeEnv(gym.Env):
    """
    蛇棋游戏模拟环境
    使用gym模拟器
    """
    SIZE = 100

    def __init__(self, ladder_num, dices):
        """
        ladder_num: 梯子数量
        dices: 骰子种类，即动作类型
        """
        self.ladder_num = ladder_num
        self.dices = dices

        # 随机生成梯子
        self.ladders = dict(np.random.randint(1, self.SIZE, size=(self.ladder_num, 2)))

        # 定义状态和动作空间
        self.observation_space = Discrete(self.SIZE + 1)  # 状态空间
        self.action_space = Discrete(len(dices))  # 动作空间

        # 生成梯子
        kvs = copy.deepcopy(self.ladders)
        for k, v in kvs.items():
            self.ladders[v] = k
        # 状态值，从1-100，不会取值0
        self.pos = 1

    def reset(self):
        """
        初始化环境状态
        """
        self.pos = 1
        return self.pos

    def step(self, a):  # a表示一个动作
        """
        返回observation, reward, done, info
        """
        step = np.random.randint(1, self.dices[a] + 1)
        self.pos += step
        if self.pos == 100:
            return 100, self.reward(self.pos), 1, {}  # observation, reward, done, info
        elif self.pos > 100:
            self.pos = 200 - self.pos

        if self.pos in self.ladders:
            self.pos = self.ladders[self.pos]
        return self.pos, self.reward(self.pos), 0, {}

    def reward(self, s):
        """
        非gym env函数
        """
        if s == 100:
            return 100
        else:
            return -1

    def render(self):
        """
        如果需要render，参考:https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
        """
        pass

    def close(self):
        pass
