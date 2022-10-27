import gym
from gym import spaces
import numpy as np
import cv2
import random
import time
from collections import deque

snakeLenGoal = 30


def collision_with_apple(apple_position, score):
    apple_position = [random.randrange(1, 50) * 10, random.randrange(1, 50) * 10]
    score += 1
    return apple_position, score


def collision_with_boundaries(snake_head):
    if (
        snake_head[0] >= 500
        or snake_head[0] < 0
        or snake_head[1] >= 500
        or snake_head[1] < 0
    ):
        return 1
    else:
        return 0


def collision_with_self(snake_position):
    snake_head = snake_position[0]
    if snake_head in snake_position[1:]:
        return 1
    else:
        return 0


class snakeEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self):
        super(snakeEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(4)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(
            low=-500, high=500, shape=(5 + snakeLenGoal,), dtype=np.float64
        )

    def step(self, action):
        self.prevActions.append(action)
        cv2.imshow("a", self.img)

        cv2.waitKey(1)
        self.img = np.zeros((500, 500, 3), dtype="uint8")
        # Display Apple
        cv2.rectangle(
            self.img,
            (self.apple_position[0], self.apple_position[1]),
            (self.apple_position[0] + 10, self.apple_position[1] + 10),
            (0, 0, 255),
            3,
        )
        # Display Snake
        x = 1
        for position in self.snake_position:
            if x == 1:
                cv2.rectangle(
                    self.img,
                    (position[0], position[1]),
                    (position[0] + 10, position[1] + 10),
                    (191, 255, 0),
                    3,
                )
            else:
                cv2.rectangle(
                    self.img,
                    (position[0], position[1]),
                    (position[0] + 10, position[1] + 10),
                    (0, 255, 0),
                    3,
                )
            x += 1

        # Takes step after fixed time
        t_end = time.time() + 0.00001
        k = -1
        while time.time() < t_end:
            if k == -1:
                k = cv2.waitKey(1)
            else:
                continue

        # 0-Left, 1-Right, 3-Up, 2-Down, q-Break
        # a-Left, d-Right, w-Up, s-Down

        # Change the head position based on the button direction
        if action == 1:
            self.snake_head[0] += 10
        elif action == 0:
            self.snake_head[0] -= 10
        elif action == 2:
            self.snake_head[1] += 10
        elif action == 3:
            self.snake_head[1] -= 10

        appleReward = 0
        # Increase Snake length on eating apple
        if self.snake_head == self.apple_position:
            self.apple_position, self.score = collision_with_apple(
                self.apple_position, self.score
            )
            self.snake_position.insert(0, list(self.snake_head))
            appleReward = 10000

        else:
            self.snake_position.insert(0, list(self.snake_head))
            self.snake_position.pop()

        # On collision kill the snake and print the score
        if (
            collision_with_boundaries(self.snake_head) == 1
            or collision_with_self(self.snake_position) == 1
        ):
            font = cv2.FONT_HERSHEY_SIMPLEX
            self.img = np.zeros((500, 500, 3), dtype="uint8")
            cv2.putText(
                self.img,
                "Your Score is {}".format(self.score),
                (140, 250),
                font,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("a", self.img)
            self.done = True

        appleDist = np.linalg.norm(
            np.array(self.snake_head) - np.array(self.apple_position)
        )

        self.totalReward = ((250 - appleDist) + appleReward) / 100

        self.reward = self.totalReward - self.prevAward
        self.prevAward = self.totalReward
        if self.done:
            self.reward = -10

        headX = self.snake_head[0]
        headY = self.snake_head[1]

        appleDeltaX = headX - self.apple_position[0]
        appleDeltaY = headY - self.apple_position[1]

        snakeLength = len(self.snake_position)

        self.observation = [headX, headY, appleDeltaX, appleDeltaY, snakeLength] + list(
            self.prevActions
        )
        self.observation = np.array(self.observation)

        info = {}
        return self.observation, self.reward, self.done, info

    def reset(self):
        self.done = False
        self.img = np.zeros((500, 500, 3), dtype="uint8")
        # Initial Snake and Apple position
        self.snake_position = [[250, 250], [240, 250], [230, 250]]
        self.apple_position = [
            random.randrange(1, 50) * 10,
            random.randrange(1, 50) * 10,
        ]
        self.score = 0
        self.reward = 0
        self.prevAward = 0
        self.prev_button_direction = 1
        self.button_direction = 1
        self.snake_head = [250, 250]

        headX = self.snake_head[0]
        headY = self.snake_head[1]

        appleDeltaX = headX - self.apple_position[0]
        appleDeltaY = headY - self.apple_position[1]

        snakeLength = len(self.snake_position)

        self.prevActions = deque(maxlen=snakeLenGoal)

        for _ in range(snakeLenGoal):
            self.prevActions.append(-1)

        self.observation = [headX, headY, appleDeltaX, appleDeltaY, snakeLength] + list(
            self.prevActions
        )
        self.observation = np.array(self.observation)

        return self.observation  # reward, done, info can't be included
