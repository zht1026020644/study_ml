# coding: utf-8 -*- coding: utf-
import numpy as np
import cv2
from PIL import Image
import time
import pickle
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

SIZE = 10  # 环境空间大小 10 * 10
EPISODES = 30000
SHOW_EVERY = 3000
#
# FOOD_REWARD = 25  # 吃掉食物奖励
# ENEMY_PENALITY = 300  # 被敌人吃掉惩罚
# MOVE_PENALITY = 1  # 移动惩罚
#
epilon = 0.6  # 随机抽取下一个动作概率0.6
EPS_DECAY = 0.99998
DISCOUNT = 0.95
LEARNING_RATE = 0.1
#
# q_table = 'q_table_1663058558.pickle'
#
# d = {1: (255, 0, 0),  # blue
#      2: (0, 255, 0),  # green
#      3: (0, 0, 255)  # red
#      }
# PLAYER_N = 1
# FOOD_N = 2
# ENEMY_N = 3


# 环境类
class EnvCube(object):
    SIZE = 10
    # 如果是图像的采用RGB
    # OBSERVATION_SPACE_VALUES = (SIZE, SIZE, 3)
    OBSERVATION_SPACE_VALUES = (4,)
    ACTION_SPACE_VALUES = 9
    RETURN_IMAGE = False
    FOOD_REWARD = 25  # 吃掉食物奖励
    ENEMY_PENALITY = -300  # 被敌人吃掉惩罚
    MOVE_PENALITY = -1  # 移动惩罚
    d = {1: (255, 0, 0),  # blue
         2: (0, 255, 0),  # green
         3: (0, 0, 255)  # red
         }
    PLAYER_N = 1
    FOOD_N = 2
    ENEMY_N = 3

    def reset(self):
        # 初始化玩家 食物 敌人
        self.player = Cube(self.SIZE)
        self.food = Cube(self.SIZE)
        while self.player == self.food:
            self.food = Cube(self.SIZE)
        self.enemy = Cube(self.SIZE)
        while self.enemy == self.food or self.enemy == self.player:
            self.enemy = Cube(self.SIZE)
        # 返回一个observation
        if self.RETURN_IMAGE:
            observation = np.array(self.get_image())
        else:
            observation = (self.player- self.food) + (self.enemy - self.enemy)
        self.episode_step = 0
        return observation
    def step(self,action):
        self.episode_step += 1
        self.player.action(action)
        self.food.move()
        self.enemy.move()
        if self.RETURN_IMAGE:
            new_observation = np.array(self.get_image())
        else:
            new_observation = (self.player- self.food) + (self.player - self.enemy)
        if self.player == self.food:
            reward = self.FOOD_REWARD
        elif self.player == self.enemy:
            reward = self.ENEMY_PENALITY
        else:
            reward = self.MOVE_PENALITY
        done = False
        if self.player == self.enemy or self.player == self.food or self.episode_step >=200:
            done = True
        return new_observation,reward,done,{}
    # 图像处理方法
    def render(self,mode='human'):
        img = self.get_image()
        img = img.resize((800, 800))
        cv2.imshow('Predator', np.array(img))
        cv2.waitKey(1)

    def get_qtable(self,qtable_name=None):
        if qtable_name is None:
            q_table = {}
            # x1: 玩家和食物之间横坐标差值，y1:玩家和食物之间纵坐标差值，x2：玩家和敌人之间横坐标差值，y2:玩家和敌人纵坐标之间的差值
            for x1 in range(-self.SIZE + 1, self.SIZE):
                for y1 in range(-self.SIZE + 1, self.SIZE):
                    for x2 in range(-self.SIZE + 1, self.SIZE):
                        for y2 in range(-self.SIZE + 1, self.SIZE):
                            q_table[(x1, y1,x2, y2)] = [np.random.uniform(-5, 0) for i in range(self.ACTION_SPACE_VALUES)]  # 9个动作对应未来的得分
        else:
            with open(qtable_name, 'rb') as f:
                q_table = pickle.load(f)
        return q_table


    def get_image(self):
        env = np.zeros((self.SIZE, self.SIZE, 3), dtype=np.uint8)
        env[self.food.x][self.food.y] = self.d[self.FOOD_N]
        env[self.player.x][self.player.y] = self.d[self.PLAYER_N]
        env[self.enemy.x][self.enemy.y] = self.d[self.ENEMY_N]
        img = Image.fromarray(env, 'RGB')
        return img


class Cube(object):
    def __init__(self, size):
        # 随机生成一个玩家位置
        self.size = size
        self.x = np.random.randint(0, self.size)
        self.y = np.random.randint(0, self.size)

    def __str__(self):
        return f'{self.x},{self.y}'

    def __sub__(self, other):
        return (self.x - other.x, self.y - other.y)

    def __eq__(self, other):
        return (self.x == other.x) and (self.y == other.y)

    def action(self, choise):
        if choise == 0:
            self.move(x=1, y=1)
        elif choise == 1:
            self.move(x=-1, y=1)
        elif choise == 2:
            self.move(x=1, y=-1)
        elif choise == 3:
            self.move(x=-1, y=-1)
        elif choise == 4:
            self.move(x=0, y=1)
        elif choise == 5:
            self.move(x=0, y=-1)
        elif choise == 6:
            self.move(x=1, y=0)
        elif choise == 7:
            self.move(x=-1, y=0)
        elif choise == 8:
            self.move(x=0, y=0)
        else:
            print(f'{choise} is error')

    def move(self, x=False, y=False):
        if not x:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x
        if not y:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y
        if self.x < 0:
            self.x = 0
        elif self.x >= self.size:
            self.x = self.size - 1
        if self.y < 0:
            self.y = 0
        elif self.y >= self.size:
            self.y = self.size - 1


# if q_table is None:
#     q_table = {}
#     # x1: 玩家和食物之间横坐标差值，y1:玩家和食物之间纵坐标差值，x2：玩家和敌人之间横坐标差值，y2:玩家和敌人纵坐标之间的差值
#     for x1 in range(-SIZE + 1, SIZE):
#         for y1 in range(-SIZE + 1, SIZE):
#             for x2 in range(-SIZE + 1, SIZE):
#                 for y2 in range(-SIZE + 1, SIZE):
#                     q_table[(x1, y1), (x2, y2)] = [np.random.uniform(-5, 0) for i in range(4)]  # 4个动作对应未来的得分
# else:
#     with open(q_table, 'rb') as f:
#         q_table = pickle.load(f)


# env = EnvCube()
# q_table = env.get_qtable()
#
# episode_rewards = []
# for episode in range(EPISODES):
#     obs = env.reset()
#     done = False
#     episode_reward = 0
#     if episode % SHOW_EVERY == 0:
#         print(f'episode #{episode},epsilon:{epilon}')
#         print(f'mean reward:{np.mean(episode_rewards[-SHOW_EVERY:])}')
#         show = True
#     else:
#         show = False
#
#     while not done:
#         if np.random.random() > epilon:
#             action = np.argmax(q_table[obs])
#         else:
#             action = np.random.randint(0, env.ACTION_SPACE_VALUES)
#         new_obs,reward,done =env.step(action)
#
#
#         # update the q_table
#         current_q = q_table[obs][action]
#         max_future_q = np.max(q_table[new_obs])
#         if reward == env.FOOD_REWARD:
#             new_q = env.FOOD_REWARD
#         else:
#             new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
#         q_table[obs][action] = new_q
#         obs = new_obs
#         # 界面显示代码
#         if show:
#             env.render()
#
#         episode_reward += reward
#     episode_rewards.append(episode_reward)
#     epilon *= EPS_DECAY
#
# moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode='valid')
# plt.plot([i for i in range(len(moving_avg))], moving_avg)
# plt.xlabel('episode #')
# plt.ylabel(f'mean {SHOW_EVERY} reward')
# plt.show()
#
# with open(f'q_table_{int(time.time())}.pickle', 'wb') as f:
#     pickle.dump(q_table, f)

# if __name__ == "__main__":
#     player = Cube()
#     print(player)
#     player.action(1)
#     print(player)
