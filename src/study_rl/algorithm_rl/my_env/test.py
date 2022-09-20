from my_env_pro import EnvCube
import pickle
import numpy as np


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


if __name__ == '__main__':
    with open('q_table_1663135594.pickle', 'rb') as f:
        q_table_s = pickle.load(f)
    test(q_table_s,30)
