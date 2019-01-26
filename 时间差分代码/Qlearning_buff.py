from yuanyang_env2 import YuanYangEnv
import random
import time
import numpy as np


class Qlearning(object):
    def __init__(self, env):
        self.gamma = 0.9
        self.lr = 0.1
        self.env = env
        self.epislon = 1
        self.Q = np.zeros((len(self.env.states), len(self.env.actions)), dtype=np.float32)

    def td_learning(self):
        for i in range(500):
            s = self.env.reset()
            step_n = 0
            while 1:
                a = self.e_greedy_policy(s)
                s_, r, t = self.env.step(a)
                td_error = r + self.gamma*np.max(self.Q[s_]) - self.Q[s, a]
                self.Q[s, a] += self.lr * td_error
                step_n += 1
                if t or step_n > 40:
                    break
                s = s_

    def td_learning_buffer(self, s_buff, a_buff, r_buff):
        for i in range(len(s_buff))[::-1]:
            s = s_buff[i]
            a = a_buff[i]
            r = r_buff[i]
            if i < len(s_buff)-1:
                td_error = r + self.gamma*np.max(self.Q[s_buff[i+1]]) - self.Q[s, a]
            else:
                td_error = r - self.Q[s, a]
            self.Q[s, a] += self.lr * td_error

    # 训练时，使用soft-greedy策略
    def e_greedy_policy(self, s):
        a_max = np.argmax(self.Q[s])
        if np.random.uniform() < 1 - self.epislon:
            return self.env.actions[a_max]
        else:
            return self.env.actions[int(random.random() * len(self.env.actions))]

    # 训练结束后，使用贪婪策略试验
    def greedy_policy(self, s):
        a_max = np.argmax(self.Q[s])
        return self.env.actions[a_max]


if __name__ == '__main__':
    env = YuanYangEnv()
    agent = Qlearning(env)
    agent.td_learning()
    print(np.sum(agent.Q))
    # 用训练好的agent记录一条从起点到终点的路径供agent2学习
    s_buf, a_buf, r_buf = [], [], []
    s = 0
    t = False
    while not t:
        a = agent.greedy_policy(s)
        # print("aaaaaaaaaaaaaaaaaa")
        # print('%d->%s\t' % (s, a))
        env.bird_male_position = env.state_to_position(s)
        env.render()
        time.sleep(0.2)
        env.state = s
        s_, r, t = env.step(a)
        s_buf.append(s)
        a_buf.append(a)
        r_buf.append(r)
        s = s_

    env2 = YuanYangEnv()
    agent2 = Qlearning(env2)
    agent2.td_learning_buffer(s_buf, a_buf, r_buf)
    flag = 1
    s = 0
    step_num = 0
    # 将最优路径打印出来
    while flag:
        a = agent2.greedy_policy(s)
        print('%d->%s\t' % (s, a))
        env2.bird_male_position = env2.state_to_position(s)
        env2.render()
        time.sleep(0.2)
        step_num += 1
        env2.state = s
        s_, r, t = env2.step(a)
        if t == True or step_num > 30:
            flag = 0
        s = s_

