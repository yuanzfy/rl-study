# -*- coding: utf-8 -*-

from snake_env import SnakeEnv

# 策略迭代
from table_agent import TableAgent
from dqn_agent import DQNAgent
from dqn_openai import OpenAiAgent
from policy_iter import PolicyIter
from value_iter import ValueIter
from generalized_iter import GeneralizedIter

# Q-learning
from model_free_agent import ModelFreeAgent
from monte_carlo import MonteCarlo
from sarsa import Sarsa
from q_learning import QLearning


def eval_policy(env, agent):
    state = env.reset()
    return_val = 0
    while True:
        act = agent.play(state)
        state, reward, terminate, _ = env.step(act)
        return_val += reward
        if terminate:
            break
    return return_val


def test_agent(env, agent):
    agent.learn()
    print ('states->actions pi = {}'.format(agent.get_pi()))
    sum_reward = 0
    for i in range(100):
        sum_reward += eval_policy(env, agent)
    print ("avg reward = {}".format(sum_reward / 100.))


if __name__ == '__main__':
    env = SnakeEnv(0, [3, 6])
    # test_agent(env, TableAgent(env, PolicyIter(-1)))
    # test_agent(env, TableAgent(env, ValueIter(-1)))
    # test_agent(env, TableAgent(env, GeneralizedIter(1, 10)))
    # test_agent(env, ModelFreeAgent(env, MonteCarlo(0.1)))
    # test_agent(env, ModelFreeAgent(env, Sarsa(0.1)))
    # test_agent(env, ModelFreeAgent(env, QLearning(0.1)))
    # test_agent(env, DQNAgent(env)) # SB玩意不收敛
    test_agent(env, OpenAiAgent(env))
