# -*- coding: utf-8 -*-

import numpy as np

from snake_env import SnakeEnv

class MonteCarlo(object):
    def __init__(self, epsilon=0.0):
        self.epsilon = epsilon

    def evaluation(self, agent, env):
        state = env.reset()
        episode = []
        while True:
            ac = agent.play(state, self.epsilon)
            next_state, reward, terminate, _ = env.step(ac)
            # 通过大数定理采样统计状态-行动值函数期望
            episode.append((state, ac, reward))
            state = next_state
            if terminate:
                break

            value = []
            return_val = 0
            for item in reversed(episode):
                return_val = return_val * agent.gamma + item[2]
                value.append((item[0], item[1], return_val))

            # every visit
            for item in reversed(value):
                agent.value_n[item[0]][item[1]] += 1
                agent.value_q[item[0]][item[1]] += (item[2] - agent.value_q[item[0]][item[1]]) / agent.value_n[item[0]][item[1]]

    def improvement(self, agent):
        new_policy = np.zeros_like(agent.pi)
        for i in range(1, agent.s_len):
            new_policy[i] = np.argmax(agent.value_q[i, :])
        if np.all(np.equal(new_policy, agent.pi)):
            return False
        else:
            agent.pi = new_policy
            return True

    def iteration(self, agent, env):
        for i in range(100):
            for j in range(100):
                self.evaluation(agent, env)
            self.improvement(agent)
