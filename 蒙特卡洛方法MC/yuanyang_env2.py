import pygame
from load import *
import math
import time
import random
import gym


class YuanYangEnv:  # (S,A,P,R,γ)
    def __init__(self):  # 设置程序需要用到的各种基本参数
        self.viewer = None
        self.FPSCLOCK = pygame.time.Clock()  # 控制渲染频率，体现在小鸟移动快慢
        self.actions = [0, 1, 2, 3]  # e,s,w,n →↓←↑
        self.state = 0
        self.states = []
        for i in range(0, 100):
            self.states.append(i)  # append()方法用于在列表末尾添加新的对象
        # 屏幕大小
        self.screen_size = (400, 300)
        self.bird_position = (0, 0)
        self.limit_distance_x = 40  # 用于限制小鸟与障碍物之间的距离
        self.limit_distance_y = 30
        self.obstacle_size = [40, 20]  # 每一块砖的尺寸
        self.obstacle1_x = []  # 用于存放障碍物的坐标
        self.obstacle1_y = []
        self.obstacle2_x = []
        self.obstacle2_y = []
        self.state = 0
        self.next_state = [0, 0]
        # 设置障碍物和小鸟的坐标位置
        for i in range(12):
            # 第一个障碍物
            self.obstacle1_x.append(120)
            if i <= 5:
                self.obstacle1_y.append(20 * i)
            else:
                self.obstacle1_y.append(20 * (i + 3))
            # 第二个障碍物
            self.obstacle2_x.append(240)
            if i <= 7:
                self.obstacle2_y.append(20 * i)
            else:
                self.obstacle2_y.append(20 * (i + 3))

        self.bird_male_init_position = [0.0, 0.0]
        self.bird_male_position = [0, 0]
        self.bird_female_init_position = [360, 0]

    def reset(self):  # 碰撞或者找到之后就初始化
        while 1:
            self.state = self.states[int(random.random() * len(self.states))]
            if not self.is_terminal(self.state):
                break
        return self.state

    # 将状态转换为坐标值
    def state_to_position(self, state):
        i = int(state / 10)
        j = state % 10
        position = [0, 0]
        position[0] = 40 * j
        position[1] = 30 * i
        return position

    def position_to_state(self, position):
        i = position[0] / 40
        j = position[1] / 30
        return int(i + 10 * j)

    def step(self, action):  # step()函数的输入是动作，输出是下一个时刻的状态、回报、是否终止和调试信息
        # 将当前状态转化为坐标
        current_position = self.state_to_position(self.state)
        next_position = [0, 0]
        # # 判断当前坐标是否与障碍物碰撞
        # flag_collide = self.collide(current_position)
        # # 判断状态是否是终点
        # flag_find = self.find(current_position)
        # if flag_collide == 1 or flag_find == 1:
        #     return self.state, 0, True
        # 状态转移
        if action == 0:
            next_position[0] = current_position[0] + 40
            next_position[1] = current_position[1]
        if action == 1:
            next_position[0] = current_position[0]
            next_position[1] = current_position[1] + 30
        if action == 2:
            next_position[0] = current_position[0] - 40
            next_position[1] = current_position[1]
        if action == 3:
            next_position[0] = current_position[0]
            next_position[1] = current_position[1] - 30
        # 判断next_state是否与障碍物碰撞
        flag_collide = self.collide(next_position)
        # 如果碰撞，那么回报为-1，并结束
        if flag_collide == 1:
            return self.state, -1, True

        self.state = self.position_to_state(next_position)
        # 判断是否终点
        flag_find = self.find(next_position)
        if flag_find == 1:
            return self.state, 1, True
        return self.state, 0, False

    def gameover(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                exit()

    def render(self):
        if self.viewer is None:
            pygame.init()
            # 画一个窗口
            self.viewer = pygame.display.set_mode(self.screen_size, 0, 32)
            # 下载图片
            self.bird_male = load_bird_male()
            self.bird_female = load_bird_female()
            self.background = load_background()
            self.obstacle = load_obstacle()
            # self.viewer.blit(self.bird_male, self.bird_male_init_position)
            # 在幕布上画图片
            self.viewer.blit(self.bird_female, self.bird_female_init_position)
            self.viewer.blit(self.background, (0, 0))
            # 画障碍物
        # self.viewer.empty()
        # 擦除
        self.viewer.blit(self.background, (0, 0))
        self.viewer.blit(self.bird_female, self.bird_female_init_position)
        # 画障碍物
        for i in range(12):
            self.viewer.blit(self.obstacle, (self.obstacle1_x[i], self.obstacle1_y[i]))
            self.viewer.blit(self.obstacle, (self.obstacle2_x[i], self.obstacle2_y[i]))
        # self.viewer.clear()
        self.viewer.blit(self.bird_male, self.bird_male_position)
        # self.viewer.blit(self.bird_female, self.bird_female_init_position)
        pygame.display.update()
        self.gameover()
        time.sleep(0.1)
        self.FPSCLOCK.tick(30)

    def collide(self, state_position):
        flag = 1
        flag1 = 1
        flag2 = 1

        # 判断第一个障碍物
        dx = []
        dy = []
        for i in range(12):
            dx1 = abs(self.obstacle1_x[i] - state_position[0])
            dx.append(dx1)
            dy1 = abs(self.obstacle1_y[i] - state_position[1])
            dy.append(dy1)
        mindx = min(dx)
        mindy = min(dy)
        if mindx >= self.limit_distance_x or mindy >= self.limit_distance_y:
            flag1 = 0
        # 判断第二个障碍物
        second_dx = []
        second_dy = []
        for i in range(12):
            dx2 = abs(self.obstacle2_x[i] - state_position[0])
            second_dx.append(dx2)
            dy2 = abs(self.obstacle2_y[i] - state_position[1])
            second_dy.append(dy2)
        mindx = min(second_dx)
        mindy = min(second_dy)
        if mindx >= self.limit_distance_x or mindy >= self.limit_distance_y:
            flag2 = 0
        if flag1 == 0 and flag2 == 0:
            flag = 0
        if state_position[0] > 360 or state_position[0] < 0 or state_position[1] > 270 or state_position[1] < 0:
            flag = 1
        return flag

    def find(self, state_position):
        flag = 0
        if abs(state_position[0] - self.bird_female_init_position[0]) < self.limit_distance_x and abs(
                state_position[1] - self.bird_female_init_position[1]) < self.limit_distance_y:
            flag = 1
        return flag

    def is_terminal(self, s):
        flag = 0
        flag1 = self.collide(self.state_to_position(s))
        flag2 = self.find(self.state_to_position(s))
        if flag1 == 1 or flag2 == 1:
            flag = 1
        return flag



# if __name__ == "__main__":
#     yy = YuanYangEnv()
#     yy.render()
#     speed = 50
#     clock = pygame.time.Clock()
#     state = 0
#     # for i in range(12):
#     #     flag_collide = 0
#     #     obstacle1_coord = [yy.obstacle1_x[i],yy.obstacle1_y[i]]
#     #     obstacle2_coord = [yy.obstacle2_x[i],yy.obstacle2_y[i]]
#     #     flag_collide = yy.collide(obstacle1_coord)
#     #     print(flag_collide)
#     #     print(yy.collide(obstacle2_coord))
#
#     while True:
#         for event in pygame.event.get():
#             if event.type == QUIT:
#                 exit()
#         # time_passed_second = clock.tick()/1000
#         # i= int(state/10)
#         # j=state%10
#         # yy.bird_male_position[0]=j*40
#         # yy.bird_male_position[1]=i*30
#         # time.sleep(0.2)
#         # pygame.display.update()
#         # state+=1
#         # yy.render()
#        print(yy.collide())
