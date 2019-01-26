import pygame
import gym
from gym.envs.classic_control import rendering
import random
import time


class Gold_coin_Env:
    def __init__(self):  # S,A,P,R,GAMMA
        self.states = [1, 2, 3, 4, 5, 6, 7, 8]
        self.actions = ['e', 'w', 's', 'n']
        self.R = dict()
        self.R['1_n'] = -1.0
        self.R['3_n'] = 1.0
        self.R['5_n'] = -1.0
        self.t = dict()
        self.t['1_n'] = 6
        self.t['1_e'] = 2
        self.t['2_w'] = 1
        self.t['2_e'] = 3
        self.t['3_e'] = 4
        self.t['3_w'] = 2
        self.t['3_n'] = 7
        self.t['4_e'] = 5
        self.t['4_w'] = 3
        self.t['5_n'] = 8
        self.t['5_w'] = 4
        self.gamma = 0.8
        self.viewer = None
        self.state = None

        self.x = [50, 150, 250, 350, 450, 50, 250, 450]
        self.y = [50, 50, 50, 50, 50, 150, 150, 150]
        self.terminate_states = dict()
        self.terminate_states[6] = 1
        self.terminate_states[7] = 1
        self.terminate_states[8] = 1

    def _step(self, action):
        state = self.state
        if state in self.terminate_states:  # 如果当前状态是终止状态，则终止
            return state, 0, True

        key = "%d_%s" % (state, action)
        if key in self.t:
            next_state = self.t[key]
        else:
            next_state = state
        self.state = next_state

        is_terminal = False

        if next_state in self.terminate_states:  # 如果下一状态是终止状态，则终止
            is_terminal = True
        if key in self.R:
            reward = self.R[key]
        else:
            reward = 0.0
        return next_state, reward, is_terminal

    def _reset(self):
        self.state = self.states[abs(int(random.random() * len(self.states)) - 4)]
        return self.state

    def _render(self):
        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 200)
            self.line1 = rendering.Line((0, 0), (500, 0))
            self.line2 = rendering.Line((0, 100), (500, 100))
            self.line3 = rendering.Line((0, 200), (100, 200))
            self.line4 = rendering.Line((200, 200), (300, 200))
            self.line5 = rendering.Line((400, 200), (500, 200))
            self.line6 = rendering.Line((0, 0), (0, 200))
            self.line7 = rendering.Line((100, 0), (100, 200))
            self.line8 = rendering.Line((200, 0), (200, 200))
            self.line9 = rendering.Line((300, 0), (300, 200))
            self.line10 = rendering.Line((400, 0), (400, 200))
            self.line11 = rendering.Line((500, 0), (500, 200))

            self.kulou1 = rendering.make_circle(30)
            self.circletrans = rendering.Transform(translation=(50, 150))
            self.kulou1.add_attr(self.circletrans)
            self.kulou1.set_color(0, 0, 0)

            self.kulou2 = rendering.make_circle(30)
            self.circletrans = rendering.Transform(translation=(450, 150))
            self.kulou2.add_attr(self.circletrans)
            self.kulou2.set_color(0, 0, 0)

            self.gold = rendering.make_circle(30)
            self.circletrans = rendering.Transform(translation=(250, 150))
            self.gold.add_attr(self.circletrans)
            self.gold.set_color(1, 0.9, 0)

            self.robot = rendering.make_circle(40)
            self.robottrans = rendering.Transform()
            self.robot.add_attr(self.robottrans)
            self.robot.set_color(0.8, 0.6, 0.4)

            self.line1.set_color(0, 0, 0)
            self.line2.set_color(0, 0, 0)
            self.line3.set_color(0, 0, 0)
            self.line4.set_color(0, 0, 0)
            self.line5.set_color(0, 0, 0)
            self.line6.set_color(0, 0, 0)
            self.line7.set_color(0, 0, 0)
            self.line8.set_color(0, 0, 0)
            self.line9.set_color(0, 0, 0)
            self.line10.set_color(0, 0, 0)
            self.line11.set_color(0, 0, 0)
            self.viewer.add_geom(self.line1)
            self.viewer.add_geom(self.line2)
            self.viewer.add_geom(self.line3)
            self.viewer.add_geom(self.line4)
            self.viewer.add_geom(self.line5)
            self.viewer.add_geom(self.line6)
            self.viewer.add_geom(self.line7)
            self.viewer.add_geom(self.line8)
            self.viewer.add_geom(self.line9)
            self.viewer.add_geom(self.line10)
            self.viewer.add_geom(self.line11)
            self.viewer.add_geom(self.kulou1)
            self.viewer.add_geom(self.kulou2)
            self.viewer.add_geom(self.gold)
            self.viewer.add_geom(self.robot)
        if self.state is None: return None
        self.robottrans.set_translation(self.x[self.state - 1], self.y[self.state - 1])
        return self.viewer.render()


if __name__ == "__main__":
    gold = Gold_coin_Env()
    gold._reset()
    gold.state = 5
    gold._render()
    time.sleep(1)
    print(gold.state)
    state, r, flag = gold._step('n')
    gold._render()
    print(state)
    time.sleep(3)

    # for i in range(8):
    #     gold.state = i+1
    #     gold._render()
    #     time.sleep(0.5)
