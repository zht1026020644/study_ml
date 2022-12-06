import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from src.study_rl.algorithm_rl.my_env import my_env_pro


# 根据状态和动作创建Q的神经网络
def build_model(status, nb_actions):
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + status))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(nb_actions, activation='linear'))
    return model


def build_agent(model, nb_actions):
    # window_length:mini_batch
    memory = SequentialMemory(limit=50000, window_length=1)
    policy = BoltzmannQPolicy()
    # nb_steps_warmup:热身步数
    # target_model_update 解决bootstrap问题
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=1000,
                   target_model_update=1e-2, policy=policy)
    dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])
    return dqn


if __name__ == '__main__':
    # env = my_env_pro.EnvCube()
    # model = build_model(env.OBSERVATION_SPACE_VALUES,env.ACTION_SPACE_VALUES)
    # dqn = build_agent(model,env.ACTION_SPACE_VALUES)
    # dqn.fit(env, nb_steps=100000, visualize=False, verbose=1)
    # dqn.save_weights('dqn_weights_r88.h5f',overwrite=True)
    # scores = dqn.test(env, nb_episodes=5, visualize=True)
    # print(np.mean(scores.history['episode_reward']))
    # print((1,)+env.OBSERVATION_SPACE_VALUES)
    # coefficient = np.random.randint(10, 100, 10)
    # print(3**10)
    a = np.random.randint(3**10)



