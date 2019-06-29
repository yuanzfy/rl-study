# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)

class DQNAgent:
    def __init__(self, env):
        self.env = env
        self.s_len = env.observation_space.n
        self.a_len = env.action_space.n

        self.learning_rate = 0.01
        self.gamma = 0.8
        self.epsilon = 0.8
        self.replace_target_iter = 300
        self.replay_buffer_size = 1024
        self.batch_size = 32

        self.learning_step = 0

        # [s, a, r, s_]
        self.replay_buffer = np.zeros((self.replay_buffer_size, 4))

        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if True:
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):
        # input
        self.s = tf.placeholder(tf.int32, [None, ], name="s")
        self.s_ = tf.placeholder(tf.int32, [None, ], name="s_")
        self.r = tf.placeholder(tf.float32, [None, ], name="r")
        self.a = tf.placeholder(tf.int32, [None, ], name="a")

        print (self.s, self.s_, self.r, self.a)
        w_initializer = tf.random_normal_initializer(0., 0.3)
        b_initializer = tf.constant_initializer(0.1)

        self.onehot_s = tf.one_hot(indices=self.s, depth=self.s_len, name="s_onehot")
        self.onehot_s_ = tf.one_hot(indices=self.s_, depth=self.s_len, name="s_onehot_")

        with tf.variable_scope('eval_net'):
            e1 = tf.layers.dense(self.onehot_s, 20, tf.nn.relu, name="e1",
                                 kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer)
            self.q_eval = tf.layers.dense(e1, self.a_len, name="q",
                                          kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer)

        with tf.variable_scope('target_net'):
            t1 = tf.layers.dense(self.onehot_s_, 20, tf.nn.relu, name="t1",
                                 kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer)
            self.q_next = tf.layers.dense(t1, self.a_len, name="t2",
                                          kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer)

        with tf.variable_scope('q_target'):
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name="q_max_s_")
            self.q_target = tf.stop_gradient(q_target)
        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_on_a = tf.gather_nd(params=self.q_eval, indices=a_indices)
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_on_a, name="loss"))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

    def _store_transition(self, s, a, r, s_):
        if not hasattr(self, 'replay_counter'):
            self.replay_counter = 0

        transition = np.hstack((s, [a, r], s_))

        index = self.replay_counter % self.replay_buffer_size
        self.replay_buffer[index, :] = transition

        self.replay_counter += 1

    def play(self, observation, e_greedy=False):
        # observation = observation[np.newaxis, :]
        if e_greedy and np.random.uniform() < self.epsilon:
                actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
                action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.a_len)
        return action

    def _learn(self):
        if self.learning_step % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        if self.replay_counter > self.replay_buffer_size:
            sample_index = np.random.choice(self.replay_buffer_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.replay_counter, size=self.batch_size)
        batch_sample = self.replay_buffer[sample_index, :]

        # s, onehot_s, q_next = self.sess.run([self.s, self.onehot_s, self.q_next],
        _, cost = self.sess.run([self._train_op, self.loss],
                                feed_dict={
                                    self.s: batch_sample[:, 1],
                                    self.a: batch_sample[:, 1],
                                    self.r: batch_sample[:, 2],
                                    self.s_: batch_sample[:, -1]})
        self.cost_his.append(cost)

        self.learning_step += 1

    def _plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

    def learn(self, max_iter=1000):
        step = 0
        for episode in range(max_iter):
            observation = self.env.reset()
            while True:
                self.env.render()

                action = self.play(np.array([observation]), e_greedy=True)

                observation_, reward, done, info = self.env.step(action)

                self._store_transition([observation], action, reward, [observation_])

                if (step > 200) and (step % 5) == 0:
                    self._learn()

                observation = observation_

                if done:
                    break
                step += 1

    def get_pi(self):
        actions = []
        for observation in range(1, self.s_len):
            action = self.play(np.array([observation]))
            actions.append(action)
        return actions


if __name__ == '__main__':
    from snake_env import SnakeEnv
    env = SnakeEnv(0, [3, 6])
    DQN = DQNAgent(env)
    DQN.learn()
    print (DQN.get_pi())
