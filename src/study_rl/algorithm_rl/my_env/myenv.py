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

FOOD_REWARD = 25  # 吃掉食物奖励
ENEMY_PENALITY = 300  # 被敌人吃掉惩罚
MOVE_PENALITY = 1  # 移动惩罚

epilon = 0.6  # 随机抽取下一个动作概率0.6
EPS_DECAY = 0.99998
DISCOUNT = 0.95
LEARNING_RATE = 0.1

q_table = 'q_table_1663058558.pickle'

d = {1: (255, 0, 0),  # blue
     2: (0, 255, 0),  # green
     3: (0, 0, 255)  # red
     }
PLAYER_N = 1
FOOD_N = 2
ENEMY_N = 3


class Cube(object):
    def __init__(self):
        # 随机生成一个玩家位置
        self.x = np.random.randint(0, SIZE)
        self.y = np.random.randint(0, SIZE)

    def __str__(self):
        return f'{self.x},{self.y}'

    def __sub__(self, other):
        return (self.x - other.x, self.y - other.y)

    def action(self, choise):
        if choise == 0:
            self.move(x=1, y=1)
        elif choise == 1:
            self.move(x=-1, y=1)
        elif choise == 2:
            self.move(x=1, y=-1)
        elif choise == 3:
            self.move(x=-1, y=-1)

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
        elif self.x >= SIZE:
            self.x = SIZE - 1
        if self.y < 0:
            self.y = 0
        elif self.y >= SIZE:
            self.y = SIZE - 1


if q_table is None:
    q_table = {}
    # x1: 玩家和食物之间横坐标差值，y1:玩家和食物之间纵坐标差值，x2：玩家和敌人之间横坐标差值，y2:玩家和敌人纵坐标之间的差值
    for x1 in range(-SIZE + 1, SIZE):
        for y1 in range(-SIZE + 1, SIZE):
            for x2 in range(-SIZE + 1, SIZE):
                for y2 in range(-SIZE + 1, SIZE):
                    q_table[(x1, y1), (x2, y2)] = [np.random.uniform(-5, 0) for i in range(4)]  # 4个动作对应未来的得分
else:
    with open(q_table, 'rb') as f:
        q_table = pickle.load(f)


episode_rewards = []
for episode in range(EPISODES):
    player = Cube()
    food = Cube()
    enemy = Cube()
    episode_reward = 0
    if episode % SHOW_EVERY == 0:
        print(f'episode #{episode},epsilon:{epilon}')
        print(f'mean reward:{np.mean(episode_rewards[-SHOW_EVERY:])}')
        show = True
    else:
        show = False

    for i in range(200):
        # 游戏的状态值
        obs = (player - food,player - enemy)
        if np.random.random() > epilon:
            action = np.argmax(q_table[obs])
        else:
            action = np.random.randint(0,4)
        player.action(action)
        if player.x == food.x and player.y == food.y:
            reward = FOOD_REWARD
        elif player.x == enemy.x and player.y == enemy.y:
            reward = -ENEMY_PENALITY
        else:
            reward = -MOVE_PENALITY

        # update the q_table
        current_q = q_table[obs][action]
        new_obs = (player - food,player - enemy)
        max_future_q = np.max(q_table[new_obs])
        if reward == FOOD_REWARD:
            new_q = FOOD_REWARD
        else:
            new_q = (1-LEARNING_RATE) * current_q + LEARNING_RATE* (reward + DISCOUNT * max_future_q)
        q_table[obs][action] = new_q

        # 界面显示代码
        if show:
            env = np.zeros((SIZE,SIZE,3),dtype=np.uint8)
            env[food.x][food.y] = d[FOOD_N]
            env[player.x][player.y] = d[PLAYER_N]
            env[enemy.x][enemy.y] = d[ENEMY_N]
            img = Image.fromarray(env,'RGB')
            img = img.resize((800 ,800))
            cv2.imshow('',np.array(img))
            if reward == FOOD_REWARD or reward == -ENEMY_PENALITY:
                if cv2.waitKey(500) & 0xFF == ord('q'):
                    break
            else:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        episode_reward += reward

        if reward == FOOD_REWARD or reward == -ENEMY_PENALITY:
            break
    episode_rewards.append(episode_reward)
    epilon *= EPS_DECAY

moving_avg = np.convolve(episode_rewards,np.ones((SHOW_EVERY,))/SHOW_EVERY,mode= 'valid')
plt.plot([i for i in range(len(moving_avg))],moving_avg)
plt.xlabel('episode #')
plt.ylabel(f'mean {SHOW_EVERY} reward')
plt.show()

with open(f'q_table_{int(time.time())}.pickle','wb') as f:
      pickle.dump(q_table, f)

# if __name__ == "__main__":
#     player = Cube()
#     print(player)
#     player.action(1)
#     print(player)
