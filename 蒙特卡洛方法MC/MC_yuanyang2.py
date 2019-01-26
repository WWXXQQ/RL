from yuanyang_env2 import YuanYangEnv
import numpy as np
import time
import random


class MC(object):
    def __init__(self, env, episilon):
        self.env = env
        self.episilon = episilon
        self.Q = np.zeros((len(self.env.states), len(self.env.actions)), dtype=np.float32)
        self.n = np.ones((len(self.env.states), len(self.env.actions)), dtype=np.float32)*1e-3
        self.pi = dict()
        self.pi = self.mc_improve()
        self.gamma = 0.8

    def choose_action(self, s):
        """
        依据soft-greedy策略self.pi选取动作
        :param s: 输入状态
        :return: 选择的动作
        """
        a_select = 0
        rd = random.random()
        prob = 0
        for i in range(len(self.env.actions)):
            prob += self.pi[s][i]
            if rd <= prob:
                a_select = self.env.actions[i]
                break
        return a_select

    def mc_learn(self, num=10):
        """
        MC策略评估
        :param num:采集num条数据
        :return: 0
        """
        # self.Q = np.zeros((len(self.env.states), len(self.env.actions)), dtype=np.float32)
        # self.n = np.ones((len(self.env.states), len(self.env.actions)), dtype=np.float32)*1e-3
        for iter in range(num):
            s_seq, r_seq, a_seq = [], [], []
            # s = self.env.states[0]
            s = self.env.reset()
            # 采一条数据，计算一次Gt
            done = False
            step_num = 0
            while done == False and step_num < 40:
                a = self.choose_action(s)
                s_, r, done = self.env.step(a)
                a_seq.append(a)
                s_seq.append(s)
                r_seq.append(r)
                step_num += 1
                s = s_
            Gt = 0
            for i in range(len(s_seq))[::-1]:
                Gt = Gt*self.gamma + r_seq[i]
                self.Q[int(s_seq[i]), int(a_seq[i])] += Gt
                self.n[int(s_seq[i]), int(a_seq[i])] += 1.0
            # if np.sum(self.Q) > 0:
            #     print("采集了%d条数据" % iter)
            #     break
        self.Q = self.Q/self.n

        return 0
    """"""
    def mc_improve(self):
        prob_s_a = dict()
        for s in self.env.states:
            if self.env.is_terminal(s):  # 终止状态不用估计值函数也不用改善
                continue
            a_max = int(np.argmax(self.Q[s]))
            # 重新赋值每个动作对应的概率
            a_prob = [self.episilon / len(self.env.actions) for _ in range(len(self.env.actions))]
            a_prob[a_max] += 1 - self.episilon

            prob_s_a[s] = a_prob
        return prob_s_a

    def mc_iteration(self):
        print(np.sum(self.Q))
        e_terminal = 0.1
        e_inital = self.episilon
        iters = 100000
        for i in range(iters):
            self.episilon -= (e_inital-e_terminal)/iters
            # print(self.episilon)
            self.mc_learn()
            self.pi = self.mc_improve()
        print(np.sum(self.Q))


if __name__ == '__main__':
    env = YuanYangEnv()
    agent = MC(env, 1)
    agent.mc_iteration()

    flag = 1
    s = 0
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

    # print('value:')
    # for i in range(1, 100):
    #     print('%d:%f\t' % (i, agent.V[i]))
    # print('')
    # print('optimal policy is \t')
    # print(agent.pi)
    # print('optimal value function is \t')
    # print(agent.V)
    # print(sum(agent.V))
    # xx = np.linspace(0, len(agent.V) - 1, 100)
    # yy = agent.V
    # plt.figure()
    # plt.plot(xx, yy)
    # plt.show()
    # # 将值函数的图像显示出来
    # z = []
    # for i in range(100):
    #     z.append(agent.V[i])
    # zz = np.array(z).reshape(10, 10)
    # plt.figure(num=2)
    # plt.imshow(zz, interpolation='none')
    # plt.show()
