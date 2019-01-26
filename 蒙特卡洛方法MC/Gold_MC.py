import gym
from Gold_coin_Env import Gold_coin_Env
import numpy as np
import pygame
import time
import random


class Gold_MC:
    def __init__(self, Gold):
        # 利用矩阵来保存行为值函数，x坐标索引是状态，对应的y为动作
        self.qvalue = np.zeros((len(Gold.states), len(Gold.actions)))
        self.n = 0.0001 * np.ones((len(Gold.states), len(Gold.actions)))
        self.actions = Gold.actions
        self.Gold = Gold
        self.gamma = Gold.gamma
        self.learn_num = 0
        self.pi = dict()

    def action_num(self, action):  # 计算值函数时使用
        for i in range(len(self.actions)):
            if action == self.actions[i]:
                return i

    def greedy_policy(self, qfun, state):
        amax = qfun[state, :].argmax()
        return self.actions[amax]

    def e_greedy_policy(self, qfun, state, e):
        amax = qfun[state, :].argmax()
        if np.random.uniform() < 1 - e:
            return self.actions[amax]
        else:
            return self.actions[int(random.random() * len(self.actions))]

    def MC_learning(self, num_iter, e):  #
        """
        策略评估，计算值函数
        :param num_iter: 需采集episode的数量
        :param e: 贪婪策略epislon的值
        """
        for iter1 in range(num_iter):  # 迭代num_iter次，采集num_iter串数据
            s_sample = []
            r_sample = []
            a_sample = []
            state = self.Gold._reset()
            is_terminal = False
            step_num = 0
            while is_terminal == False and step_num < 8:  # 采集数据
                a = self.e_greedy_policy(self.qvalue, state, e)
                next_state, reward, is_terminal = self.Gold._step(a)
                s_sample.append(state)
                r_sample.append(reward)
                a_sample.append(a)
                step_num += 1
                state = next_state
            g = 0.0
            for i in range(len(s_sample) - 1, -1, -1):  # 计算各状态累积回报
                g *= self.gamma
                g += r_sample[i]
                # 在这里，并没有用贝尔曼方程，没有用值函数表示动作值函数，用的是动作值函数的定义
                # 即发出动作之后的累积回报，并以表的形式存储
                self.qvalue[s_sample[i], self.action_num(a_sample[i])] += g
                self.n[s_sample[i], self.action_num(a_sample[i])] += 1.0
        self.qvalue = self.qvalue / self.n
        return self.qvalue

    def MC_improve(self):  # 策略改善(贪婪策略)
        for s_step in range(len(self.Gold.states)):
            a = self.greedy_policy(self.qvalue, s_step)
            self.pi[s_step] = a
        return self.pi


if __name__ == "__main__":
    goldrobot = Gold_coin_Env()
    agent = Gold_MC(goldrobot)
    agent.MC_learning(2000, 0.1)
    agent.pi = agent.MC_improve()
    print(sum(agent.qvalue))
    while True:
        goldrobot._reset()
        goldrobot._render()
        time.sleep(1)
        terminal = False
        while False == terminal:
            print(goldrobot.state)
            a = agent.pi[goldrobot.state]
            print(a)
            s, r, t = goldrobot._step(a)
            goldrobot.state = s
            terminal = t
            print(terminal)
            goldrobot._render()
            time.sleep(1)
