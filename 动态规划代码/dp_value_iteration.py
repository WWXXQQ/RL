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

    def value_iteration(self):
        for i in range(1000):
            delta = 0.0
            for state in self.states:
                v0 = self.v[state]
                flag1 = 0
                flag2 = 0
                flag1 = yuanyang.collide(yuanyang.state_to_position(state))
                flag2 = yuanyang.find(yuanyang.state_to_position(state))
                if flag1 == 1 or flag2 == 1: continue
                # a1 = self.actions[int(random.random() * 4)]
                a1 = self.actions[0]
                # yuanyang.bird_male_position = yuanyang.state_to_position(state)
                # yuanyang.render()
                s, r, t = yuanyang.transform(state, a1)
                # 策略评估
                v1 = r + self.gamma * self.v[s]
                # 策略改进
                for action in self.actions:
                    s, r, t = yuanyang.transform(state, action)
                    if v1 < r + self.gamma * self.v[s]:
                        a1 = action
                        v1 = r + self.gamma * self.v[s]
                delta += abs(v1 - self.v[state])
                self.pi[state] = a1
                self.v[state] = v1
                # self.v[state] += 0.1*(v1-v0)
            if delta < 1e-6:
                print("迭代完成，共%d次" % i)
                break


if __name__ == "__main__":
    yuanyang = YuanYangEnv()
    policy_value = Policy_Value(yuanyang)
    policy_value.value_iteration()
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
        time.sleep(0.5)
        step_num += 1
        s_, r, t = yuanyang.transform(s, a)
        if t == True or step_num > 20:
            flag = 0
        s = s_
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
