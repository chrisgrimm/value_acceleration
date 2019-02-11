import numpy as np
import cv2
from gym.spaces import Discrete, Box


class Gridworld:

    def __init__(self, dim):
        self.dim = dim
        self.agent_x, self.agent_y = self.random_agent_pos()
        self.action_space = Discrete(4)
        self.observation_space = Box(0, 255, shape=(64, 64, 1), dtype=np.uint8)
        self.action_map = {
            0: (-1, 0),
            1: (1, 0),
            2: (0, -1),
            3: (0, 1)
        }
        self.human_action_map = {
            'a': (-1, 0),
            'd': (1, 0),
            'w': (0, -1),
            's': (0, 1)
        }


    def get_all_states(self):
        all_obs = []
        for y in range(self.dim):
            for x in range(self.dim):
                obs = self.get_observation(x, y)
                all_obs.append(obs)
        return np.array(all_obs)


    def step(self, action):
        delta_x, delta_y = self.action_map[action]
        new_x, new_y = np.clip(self.agent_x + delta_x, 0, self.dim-1), np.clip(self.agent_y + delta_y, 0, self.dim-1)
        self.agent_x, self.agent_y = new_x, new_y
        obs = self.get_observation(self.agent_x, self.agent_y)
        return obs, 0, False, {}

    def reset(self):
        self.agent_x, self.agent_y = self.random_agent_pos()
        obs = self.get_observation(self.agent_x, self.agent_y)
        return obs


    def get_observation(self, x, y):
        canvas = 255 * np.ones(shape=[self.dim, self.dim, 1], dtype=np.uint8)
        canvas[y, x, 0] = 0
        canvas = cv2.resize(canvas, (64, 64), interpolation=cv2.INTER_NEAREST)
        canvas = np.reshape(canvas, (64, 64, 1))
        return canvas

    def random_agent_pos(self):
        return np.random.randint(0, self.dim), np.random.randint(0, self.dim)



