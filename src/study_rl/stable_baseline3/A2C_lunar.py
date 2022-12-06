import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
import gym
# Create environment
env = gym.make('LunarLander-v2')
model = A2C('MlpPolicy', env, verbose=1, tensorboard_log='./logs', learning_rate=5e-4,
            policy_kwargs={'net_arch': [256, 256]})
model.learn(total_timesteps=int(1e6),tb_log_name='A2C_Net256')
model.save("A2C_Net256_lunar")