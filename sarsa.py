# -*- coding: utf-8 -*-

import numpy as np

from snake_env import SnakeEnv


class Sarsa(object):
    def __init__(self, epsilon=0.0):
        self.epsilon = epsilon

    def evaluation(self, agent, env):
        state = env.reset()
        prev_state = -1
        prev_act = -1
        while True:
            act = agent.play(state, self.epsilon)
            next_state, reward, terminate, _ = env.step(act)
            if prev_act != -1:
                return_val = reward + agent.gamma * (0 if terminate else agent.value_q[state][act])
                agent.value_n[prev_state][prev_act] += 1
                agent.value_q[prev_state][prev_act] += (return_val - agent.value_q[prev_state][prev_act]) / agent.value_n[prev_state][prev_act]

            prev_act = act
            prev_state = state
            state = next_state

            if terminate:
                break

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
