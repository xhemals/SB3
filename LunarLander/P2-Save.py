import gym
from stable_baselines3 import A2C, PPO
import os
import time

alModel = A2C
dirModel = "A2C"

modelsDir = f"models/{dirModel}-{int(time.time())}"
logDir = f"logs/{dirModel}-{int(time.time())}"

if not os.path.exists(modelsDir):
    os.makedirs(modelsDir)

if not os.path.exists(logDir):
    os.makedirs(logDir)

env = gym.make("LunarLander-v2")
env.reset()

model = alModel("MlpPolicy", env, verbose=1, tensorboard_log=logDir)

timesteps = 1000
for i in range(1, 10000):
    model.learn(
        total_timesteps=timesteps, reset_num_timesteps=False, tb_log_name=dirModel
    )
    model.save(f"{modelsDir}/{timesteps*i}")

env.close()
