import time

from my_env_pro import EnvCube
import pickle
import numpy as np
import gym
from gym.envs.classic_control import rendering
from gym.envs.classic_control.rendering import LineWidth

def test(q_table, episodes, show_enable=True):
    env = EnvCube()
    avg_reward = 0
    for episode in range(episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = np.argmax(q_table[obs])
            obs, reward, done = env.step(action)
            if show_enable == True:
                env.render()
            episode_reward += reward
            print(f'episode:{episode},episode_reward:{episode_reward}')
        avg_reward += episode_reward
    avg_reward /= episodes
    print(f'avg_reward:{avg_reward}')


class Test(gym.Env):
    # 如果你不想改参数，下面可以不用写
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    # 我们在初始函数中定义一个 viewer ，即画板
    def __init__(self,x,y):
        self.viewer = rendering.Viewer(600, 400)  # 600x400 是画板的长和框
        self.x = x
        self.y = y

    # 继承Env render函数
    def render(self, mode='human', close=False):
        start = [100, 100]
        end = [100, 300]

        for i in range(11):
            line = rendering.Line(tuple(start), tuple(end))
            if i % 3 == 0:
                line.set_color(1,0,0)
                line.linewidth = LineWidth(10)
                self.viewer.add_geom(line)
            else:
                self.viewer.add_geom(line)
            start[0] +=  40
            end[0] +=  40
        start = [100,100]
        end = [500,100]

        for j in range(11):
            line = rendering.Line(tuple(start), tuple(end))
            if j % 3 == 0:
                line.set_color(1, 0, 0)
                line.linewidth = LineWidth(10)
                self.viewer.add_geom(line)
            else:
                self.viewer.add_geom(line)
            start[1] += 20
            end[1] += 20
        self.cicle = rendering.make_circle(10)
        # self.circletrans = rendering.Transform(translation=(320, 150))
        self.circletrans = rendering.Transform(translation=(self.x, self.y))
        self.cicle.add_attr(self.circletrans)
        self.cicle.set_color(0,0.8,0.5)
        self.viewer.add_geom(self.cicle)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')


#         SIZE = 10
#         env = np.ones((SIZE, SIZE, 3), dtype=np.uint8)
#         for i in range(3):
#             x = np.random.randint(10)
#             y = np.random.randint(10)
#             env[x][y] = (0, 255, 0)
#             img = Image.fromarray(env,'RGB')
#             cv2.imshow('test',np.array(img))
#             cv2.waitKey(1)


if __name__ == '__main__':
    # with open('q_table_1663135594.pickle', 'rb') as f:
    #     q_table_s = pickle.load(f)
    # test(q_table_s,30)
    action_dict = {0:[0,1],1:[0,-1],2:[0,0],3:[-1,-1],4:[-1,0],5:[-1,1],6:[1,-1],7:[1,0],8:[1,1]}
    # t = Test(320,150)
    while True:
        t = Test(320, 150)
        for i in range(100):
            num = np.random.randint(9)
            action = action_dict[num]
            t.x = t.x + action[0] * 40
            t.y = t.y + action[1] * 20
            if t.x<0 or t.x>500 or t.y<0  or t.y >300:
                break
            time.sleep(2)
            t.render()

