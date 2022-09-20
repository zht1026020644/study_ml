from src.study_rl.algorithm_rl.my_env import my_env_pro
from dqn_exmple import build_agent,build_model
import numpy as np
if __name__ == '__main__':
    # 不进行训练 直接拿保存的权重参数进行测试
    env = my_env_pro.EnvCube()
    model = build_model(env.OBSERVATION_SPACE_VALUES, env.ACTION_SPACE_VALUES)
    dqn = build_agent(model, env.ACTION_SPACE_VALUES)
    dqn.load_weights('dqn_weights_r88.h5f')
    scores = dqn.test(env, nb_episodes=10, visualize=True)
    print(np.mean(scores.history['episode_reward']))
