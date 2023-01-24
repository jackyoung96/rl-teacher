import gym
import os
import numpy as np

class RLteacherWrapper(gym.Wrapper):
    def __init__(self, env, predictor):
        super().__init__(env)
        self.predictor = predictor
        self.obs = None

    def reset(self, seed=None):
        observation, info = self.env.reset(seed)
        self.obs = observation
    
    def step(self, action):
        # TODO: how to get path
        obs, _, terminated, truncated, info = self.env.step(action)
        reward = self.predictor.predict_reward_from_pair(self.obs, action)
        self.obs = obs
        return obs, reward, terminated, truncated, info
