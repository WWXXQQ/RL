"""
策略迭代算法，采用soft-greedy策略
一开始忘记了写这两条语句，但是发现也可以成功
if self.env.is_terminal(s):  # 终止状态不用估计值函数也不用改善
    continue
"""
from yuanyang_env2 import YuanYangEnv
import numpy as np
import random
import time
import matplotlib.pyplot as plt


class DP_soft(object):
    def __init__(self, env, episolon):
        self.episolon = episolon
        self.env = env
        self.gamma = 0.9
        self.V = [0.0 for _ in range(len(self.env.states))]
        # 依据初始化的值函数初始化策略，索引为状态，对应有该状态下每个动作的概率
        self.pi = self.policy_improvment()

    def policy_evaluation(self):
        for i in range(10):
            delta = 0.0
            for s in self.env.states:
                # if self.env.is_terminal(s):  # 终止状态不用估计值函数也不用改善
                #     continue
                Q = []
                for a in self.env.actions:
                    self.env.state = s
                    s_, r, done = self.env.step(a)
                    if done:
                        Q.append(r)
                    else:
                        Q.append(r + self.gamma * self.V[s_])
                # 算V[S]=∑pi(a|s)Q(s,a)
                v = 0
                for a in range(len(self.env.actions)):
                    v += Q[a] * self.pi[s][a]
                delta += abs(v - self.V[s])
                self.V[s] = v
            if delta < 1e-6:
                print("迭代了%d次" % i)
                break

    def policy_improvment(self):
        # 策略改善就是依据评估好的状态值函数重新赋值每个动作对应的概率
        prob_s_a = dict()
        for s in self.env.states:
            # if self.env.is_terminal(s):  # 终止状态不用估计值函数也不用改善
            #     continue
            Q = []
            for a in self.env.actions:
                self.env.state = s
                s_, r, done = self.env.step(a)
                if done:
                    Q.append(r)
                else:
                    Q.append(r + self.gamma * self.V[s_])
            a_maxQ = self.env.actions[Q.index(max(Q))]
            # 重新赋值每个动作对应的概率
            a_prob = [self.episolon / len(self.env.actions) for _ in range(len(self.env.actions))]
            a_prob[a_maxQ] += 1 - self.episolon

            prob_s_a[s] = a_prob

        return prob_s_a

    def policy_iteration(self):
        for i in range(10):
            self.policy_evaluation()
            self.pi = self.policy_improvment()

    def choose_action(self, s):
        a_select = 0
        rd = random.random()
        prob = 0
        for i in range(len(self.env.actions)):
            prob += self.pi[s][i]
            if rd <= prob:
                a_select = self.env.actions[i]
                break
        return a_select


if __name__ == '__main__':
    env = YuanYangEnv()
    agent = DP_soft(env, 0)
    agent.policy_iteration()
    Q = np.zeros((len(env.states), len(env.actions)), dtype=np.float32)
    for state in env.states:
        if env.is_terminal(state):
            continue
        for action in env.actions:
            # state = int(state)
            # action = int(action)
            next_state, r, done = env.step(action)
            if done:
                Q[state, action] = r
            else:
                Q[state, action] = r + agent.gamma*agent.V[next_state]
    print("动作值函数总和为：", np.sum(Q))


    flag = 1
    s = 0
    # print(policy_value.pi)
    step_num = 0
    # 将最优路径打印出来
    while flag:
        a = agent.choose_action(s)
        print('%d->%s\t' % (s, a))
        env.bird_male_position = env.state_to_position(s)
        env.render()
        time.sleep(0.2)
        step_num += 1
        env.state = s
        s_, r, t = env.step(a)
        if t == True or step_num > 20:
            flag = 0
        s = s_

    print('value:')
    for i in range(1, 100):
        print('%d:%f\t' % (i, agent.V[i]))
    print('')
    print('optimal policy is \t')
    print(agent.pi)
    print('optimal value function is \t')
    print(agent.V)
    print(sum(agent.V))
    xx = np.linspace(0, len(agent.V) - 1, 100)
    yy = agent.V
    plt.figure()
    plt.plot(xx, yy)
    plt.show()
    # 将值函数的图像显示出来
    z = []
    for i in range(100):
        z.append(agent.V[i])
    zz = np.array(z).reshape(10, 10)
    plt.figure(num=2)
    plt.imshow(zz, interpolation='none')
    plt.show()
