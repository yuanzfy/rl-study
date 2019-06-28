# -*- coding: utf-8 -*-

import numpy as np


class PolicyIter(object):
    def __init__(self, max_iter):
        self.max_iter = max_iter

    def evaluation(self, agent):
        iteration = 0
        while True:
            iteration += 1
            new_value_pi = agent.value_pi.copy()
            for i in range(1, agent.s_len):
                ac = agent.pi[i]
                transition = agent.p[ac, i, :]
                value_sa = np.dot(transition, agent.r + agent.gamma * agent.value_pi)
                new_value_pi[i] = value_sa

            diff = np.sqrt(np.sum(np.power(agent.value_pi - new_value_pi, 2)))
            if diff < 1e-6:
                break
            else:
                agent.value_pi = new_value_pi
            if iteration == self.max_iter:
                break

    def improvement(self, agent):
        new_policy = np.zeros_like(agent.pi)
        for i in range(1, agent.s_len):
            for j in range(0, agent.a_len):
                agent.value_q[i, j] = np.dot(agent.p[j, i, :], agent.r + agent.gamma * agent.value_pi)
            # 更新策略
            max_act = np.argmax(agent.value_q[i, :])
            new_policy[i] = max_act
        if np.all(np.equal(new_policy, agent.pi)):
            return False
        else:
            agent.pi = new_policy
            return True

    def iteration(self, agent):
        iteration = 0
        while True:
            iteration += 1
            self.evaluation(agent)
            ret = self.improvement(agent)
            if not ret:
                break
        print ("iter {} rounds converge".format(iteration))
