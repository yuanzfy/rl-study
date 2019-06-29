# -*- coding: utf-8 -*-

from baselines.a2c import a2c
from baselines.a2c import runner
import numpy as np

np.random.seed(1)

class OpenAiAgent:
    def __init__(self, env):
        self.env = env
        self.s_len = env.observation_space.n

    def learn(self):
        self.model = a2c.learn(env=self.env, network='mlp', total_timesteps=30000)

    def play(self, observation):
        actions, values, states, _ = self.model.step([observation])
        return actions[0]

    def get_pi(self):
        actions = []
        for observation in range(1, self.s_len):
            action = self.play(observation)
            actions.append(action)
        return actions


if __name__ == '__main__':
    from snake_env import SnakeEnv
    env = SnakeEnv(0, [3, 6])
    DQN = OpenAiAgent(env)
    DQN.learn()
    print (DQN.get_pi())
