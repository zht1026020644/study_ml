import gym
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

# Create environment
env = gym.make('LunarLander-v2')

# Instantiate the agent
model = DQN('MlpPolicy', env, verbose=1, tensorboard_log='./logs', learning_rate=5e-4,
            policy_kwargs={'net_arch': [256, 256]})
# Train the agent
# model.learn(total_timesteps=int(1e6),tb_log_name='DQN_Net256')
# # Save the agent
# model.save("dqn_Net256_lunar")
# del model  # delete trained model to demonstrate loading

# Load the trained agent
# NOTE: if you have loading issue, you can pass `print_system_info=True`
# to compare the system on which the model was trained vs the current one
# model = DQN.load("dqn_lunar", env=env, print_system_info=True)
model = DQN.load("dqn_Net256_lunar", env=env)

# Evaluate the agent
# NOTE: If you use wrappers with your environment that modify rewards,
#       this will be reflected here. To evaluate with original rewards,
#       wrap environment in a "Monitor" wrapper before other wrappers.
# mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10,render=True,deterministic=True)
# print(mean_reward)
# Enjoy trained agent
# obs = env.reset()
# for i in range(1000):
#     action, _states = model.predict(obs, deterministic=True)
#     obs, rewards, dones, info = env.step(action)
#     env.render()

episodes = 10
for episode in range(episodes):
    obs = env.reset()
    done = False
    rewards = 0
    while not done:
        # action = env.action_space.sample()
        # action, _states = model.predict(obs, deterministic=True)
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        rewards += reward
    print(rewards)
