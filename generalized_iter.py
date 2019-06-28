# -*- coding: utf-8 -*-

import numpy as np
from value_iter import ValueIter
from policy_iter import PolicyIter

class GeneralizedIter(object):
    def __init__(self, policy_max_iter, value_max_iter):
        self.policy_iter = PolicyIter(policy_max_iter)
        self.value_iter = ValueIter(value_max_iter)

    def iteration(self, agent):
        self.value_iter.iteration(agent)
        self.policy_iter.iteration(agent)
