"""Environment-related modules."""

from typing import Callable
import gym
import numpy as np

class Environment:
    """
    This is a wrapper around an OpenAI Gym environment, so that
    we can store the features function along with the environment itself.
    
    Parameters:
        env (gym.Env): An OpenAi Gym environment.
        features (Callable):  Given a :class:`.Trajectory`, this function
            must return a :class:`numpy.array` of features.
    
    Attributes:
        env (gym.Env): The wrapped environment.
        features (Callable): Features function.
        action_space: Inherits from :py:attr:`env`.
        observation_space: Inherits from :py:attr:`env`.
        reset (Callable): Inherits from :py:attr:`env`.
        step (Callable): Inherits from :py:attr:`env`.
        render (Callable): Inherits from :py:attr:`env`, if it exists; None otherwise.
        render_exists (bool): True if :py:attr:`render` exists.
        close (Callable): Inherits from :py:attr:`env`, if it exists; None otherwise.
        close_exists (bool): True if :py:attr:`close` exists.
    """
    def __init__(self, env: gym.Env, feature_func: Callable):
        self.env = env
        self.features = feature_func

        # Mirror the main functionality
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.reset = self.env.reset
        self.step = self.env.step

        # Mirror the render function only if it exists
        self.render_exists = callable(getattr(self.env, "render", None))
        self.render = self.env.render if self.render_exists else None

        # Mirror the close function only if it exists
        self.close_exists = callable(getattr(self.env, "close", None))
        self.close = self.env.close if self.close_exists else None

class TrajectoryRewardEnvironment(gym.Env):
    """A wrapper around an OpenAI Gym environment, which gives a reward based on a given feature function
    of an entire trajectory."""
    def __init__(self, original_env, feature_func, max_episode_length, weights):
        super().__init__()
        self.original_env = original_env
        self.feature_func = feature_func
        self.max_episode_length = max_episode_length
        self.weights = weights

        self.action_space = self.original_env.action_space
        self.observation_space = self.original_env.observation_space

        self.current_trajectory = []
        self.current_length = 0

    def step(self, action):
        # Execute the action in the original environment
        state, _, done, info = self.original_env.step(action)
        self.current_trajectory.append((state, action))

        self.current_length += 1
        reward = 0

        # Check if the episode should end
        if self.current_length >= self.max_episode_length or done:
            feature_vector = self.feature_func(self.current_trajectory)
            reward = np.dot(self.weights, feature_vector)
            self.current_trajectory = []
            self.current_length = 0
            done = True

        return state, reward, done, info

    def reset(self):
        # Reset the original environment and the trajectory
        initial_state = self.original_env.reset()
        self.current_trajectory = []
        self.current_length = 0
        return initial_state

    def render(self, mode='human', close=False):
        return self.original_env.render(mode, close)

    # Implement other necessary methods based on your requirements
