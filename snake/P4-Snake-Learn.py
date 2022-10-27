from stable_baselines3 import PPO
import os
import time
from snakeenvFast import snakeEnv


alModel = PPO
dirModel = "PPO"

modelsDir = f"models/{dirModel}-{int(time.time())}"
logDir = f"logs/{dirModel}-{int(time.time())}"

if not os.path.exists(modelsDir):
    os.makedirs(modelsDir)

if not os.path.exists(logDir):
    os.makedirs(logDir)

env = snakeEnv()
env.reset()

model = alModel("MlpPolicy", env, verbose=1, tensorboard_log=logDir)

timesteps = 10000
for i in range(1, 1000000):
    model.learn(
        total_timesteps=timesteps, reset_num_timesteps=False, tb_log_name=dirModel
    )
    model.save(f"{modelsDir}/{timesteps*i}")

env.close()
