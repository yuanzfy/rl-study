# -*- coding: utf-8 -*-

import copy
import numpy as np
import gym
from gym.spaces import Discrete
from gym.envs.registration import register


class SnakeEnv(gym.Env):
    """
    蛇棋游戏模拟环境
    使用gym模拟器
    """
    SIZE = 100

    def __init__(self, ladder_num=0, dices=[3, 6]):
        """
        ladder_num: 梯子数量
        dices: 骰子种类，即动作类型
        """
        self.num_envs = 1 # 愚蠢的东西，为了a2c

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

    def step(self, actions):
        """
        actions: 一个动作或一个动作list
        返回observation, reward, done, info
        """
        def _step(a):
            step = np.random.randint(1, self.dices[a] + 1)
            self.pos += step
            if self.pos == 100:
                return 100, self.reward(self.pos), 1, {}  # observation, reward, done, info
            elif self.pos > 100:
                self.pos = 200 - self.pos

            if self.pos in self.ladders:
                self.pos = self.ladders[self.pos]
            return self.pos, self.reward(self.pos), 0, {}

        if isinstance(actions, np.ndarray):
            obs, rewards, dones, infos = [], [], [], []
            for _a in actions:
                _ob, _reward, _done, _info = _step(_a)
                obs.append(_ob)
                rewards.append(_reward)
                dones.append(_done)
                infos.append(_info)
            return obs, rewards, dones, infos
        else:
            return _step(actions)

    def reward(self, s):
        """
        非gym env函数
        """
        if s == 100:
            return 100
        else:
            return -1

    def render(self, mode, **kwargs):
        """
        如果需要render，参考:https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
        """
        pass

    def close(self):
        pass


# 注册该env
register(
    id='SnakeEnv-v111',
    entry_point='snake_env:SnakeEnv',
)
