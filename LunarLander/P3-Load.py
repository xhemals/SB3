import gym
from stable_baselines3 import A2C, PPO

alModel = PPO
dirModel = "PPO"
modelsDir = f"models/{dirModel}"

env = gym.make("LunarLander-v2")
env.reset()

modelPath = f"best-model/model.zip"
model = alModel.load(modelPath, env=env)

episodes = 10

for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        env.render()
        action, _ = model.predict(obs)
        obs, rewards, done, info = env.step(action)

env.close()
