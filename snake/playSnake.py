from stable_baselines3 import PPO
from snakeenv import snakeEnv

alModel = PPO
dirModel = "PPO"

env = snakeEnv()
env.reset()

modelPath = f"best-model/model.zip"
model = alModel.load(modelPath, env=env)

episodes = 100

for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        print(rewards)
