# /bin/python
import numpy as np
import random
import os
import pygame
import time
import matplotlib.pyplot as plt
from yuanyang_env import YuanYangEnv


class Policy_Value:
    def __init__(self, yuanyang):
        self.states = yuanyang.states
        self.actions = yuanyang.actions
        self.v = [0.0 for i in range(len(self.states) + 1)]
        self.pi = dict()
        self.yuanyang = yuanyang
        self.gamma = yuanyang.gamma
        # 初始化策略
        for state in self.states:
            flag1 = 0
            flag2 = 0
            flag1 = yuanyang.collide(yuanyang.state_to_position(state))
            flag2 = yuanyang.find(yuanyang.state_to_position(state))
            if flag1 == 1 or flag2 == 1: continue
            self.pi[state] = self.actions[int(random.random() * len(self.actions))]
            # print(self.pi)

    def policy_improve(self):
        # 利用更新后的值函数，进行策略改进v
        for state in self.states:
            flag1 = 0
            flag2 = 0
            flag1 = yuanyang.collide(yuanyang.state_to_position(state))
            flag2 = yuanyang.find(yuanyang.state_to_position(state))
            if flag1 == 1 or flag2 == 1: continue
            a1 = self.actions[0]
            s, r, t = yuanyang.transform(state, a1)
            v1 = r + self.gamma * self.v[s]
            # 找状态s时，采用哪种动作，值函数最大
            for action in self.actions:
                s, r, t = yuanyang.transform(state, action)
                if v1 < r + self.gamma * self.v[s]:
                    a1 = action
                    v1 = r + self.gamma * self.v[s]
            # 贪婪策略，进行更新
            self.pi[state] = a1
            # 输出新的策略self.pi[STATE]
        # print(self.pi)

    def policy_evaluate(self):
        # 策略评估在计算值函数#高斯塞德尔迭代
        for i in range(100):
            delta = 0.0
            for state in self.states:
                flag1 = 0
                flag2 = 0
                flag1 = yuanyang.collide(yuanyang.state_to_position(state))
                flag2 = yuanyang.find(yuanyang.state_to_position(state))
                if flag1 == 1 or flag2 == 1: continue
                action = self.pi[state]
                s, r, t = yuanyang.transform(state, action)
                # 更新值
                new_v = r + self.gamma * self.v[s]
                delta += abs(self.v[state] - new_v)
                # 更新值替换原来的值函数
                self.v[state] = new_v
            if delta < 1e-6:
                break

    def policy_iterate(self):
        for i in range(100):
            # 策略评估,变的时v
            self.policy_evaluate()
            # 策略改进
            # 变的是pi
            self.policy_improve()


if __name__ == "__main__":
    yuanyang = YuanYangEnv()
    policy_value = Policy_Value(yuanyang)
    policy_value.policy_iterate()
    flag = 1
    s = 0
    # print(policy_value.pi)
    step_num = 0
    # 将最优路径打印出来
    while flag:
        a = policy_value.pi[s]
        print('%d->%s\t' % (s, a))
        yuanyang.bird_male_position = yuanyang.state_to_position(s)
        yuanyang.render()
        time.sleep(0.2)
        step_num += 1
        s_, r, t = yuanyang.transform(s, a)
        if t == True or step_num > 20:
            flag = 0
        s = s_

    # print('value:')
    # for i in range(1, 100):
    #     print('%d:%f\t' % (i, policy_value.v[i]))
    # print('')
    print('optimal policy is \t')
    print(policy_value.pi)
    print('optimal value function is \t')
    print(policy_value.v)
    print(sum(policy_value.v))
    xx = np.linspace(0, len(policy_value.v) - 1, 101)
    yy = policy_value.v
    plt.figure()
    plt.plot(xx, yy)
    plt.show()
    # 将值函数的图像显示出来
    z = []
    for i in range(100):
        z.append(1000 * policy_value.v[i])
    zz = np.array(z).reshape(10, 10)
    plt.figure(num=2)
    plt.imshow(zz, interpolation='none')
    plt.show()
    # print('policy:')
    # for i in range(1, 100):
    #     print('%d->%s\t' % (i, policy_value.pi[i]))
    # print('')
