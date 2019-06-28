# -*- coding: utf-8 -*-

import numpy as np


class ValueIter(object):
    def __init__(self, max_iter):
        self.max_iter = max_iter

    def evaluation(self, agent):
        iteration = 0
        while True:
            iteration += 1
            new_value_pi = agent.value_pi.copy()
            for i in range(1, agent.s_len):
                value_sas = []
                # 价值迭代，每次都选择价值最大的动作作为价值迭代后结果
                for ac in range(0, agent.a_len):
                    transition = agent.p[ac, i, :]
                    value_sa = np.dot(transition, agent.r + agent.gamma * agent.value_pi)
                    value_sas.append(value_sa)
                new_value_pi[i] = max(value_sas)

            diff = np.sqrt(np.sum(np.power(agent.value_pi - new_value_pi, 2)))
            if diff < 1e-6:
                break
            else:
                agent.value_pi = new_value_pi
            if iteration == self.max_iter:
                break
        print ("valueiter, evaluation {} iters".format(iteration))

    def improvement(self, agent):
        for i in range(1, agent.s_len):
            for j in range(0, agent.a_len):
                agent.value_q[i, j] = np.dot(agent.p[j, i, :], agent.r + agent.gamma * agent.value_pi)
            # 更新策略
            max_act = np.argmax(agent.value_q[i, :])
            agent.pi[i] = max_act

    def iteration(self, agent):
        self.evaluation(agent)
        self.improvement(agent)
