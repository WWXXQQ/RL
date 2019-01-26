from yuanyang_env2 import YuanYangEnv
import random
import time
import numpy as np


class Sarsa(object):
    def __init__(self, env):
        self.gamma = 0.9
        self.lr = 0.1
        self.env = env
        self.epislon = 1
        self.Q = np.zeros((len(self.env.states), len(self.env.actions)), dtype=np.float32)

    def td_learning(self):
        for i in range(50000):
            s = self.env.reset()
            step_n = 0
            while 1:
                a = self.e_greedy_policy(s)
                s_, r, t = self.env.step(a)
                if t:
                    td_error = r - self.Q[s, a]
                else:
                    a_ = self.e_greedy_policy(s_)
                    td_error = r + self.gamma*self.Q[s_, a_] - self.Q[s, a]
                self.Q[s, a] += self.lr * td_error
                step_n += 1
                if t or step_n > 40:
                    break
                s = s_

    # 训练结束后，使用贪婪策略试验
    def greedy_policy(self, s):
        a_max = np.argmax(self.Q[s])
        return self.env.actions[a_max]

    # 训练时，使用soft-greedy策略
    def e_greedy_policy(self, s):
        a_max = np.argmax(self.Q[s])
        if np.random.uniform() < 1 - self.epislon:
            return self.env.actions[a_max]
        else:
            return self.env.actions[int(random.random() * len(self.env.actions))]


if __name__ == '__main__':
    env = YuanYangEnv()
    agent = Sarsa(env)
    agent.td_learning()
    print(np.sum(agent.Q))
    flag = 1
    s = 0
    step_num = 0
    # 将最优路径打印出来
    while flag:
        a = agent.greedy_policy(s)
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
