import gym
import numpy as np
from gym import spaces
from gym.core import ObservationWrapper
from gym.spaces.box import Box
#from universe import vectorized
#from universe.wrappers import Unvectorize, Vectorize

import cv2


# Taken from https://github.com/openai/universe-starter-agent
def create_atari_env(env_id):
    env = gym.make(env_id)
    if len(env.observation_space.shape) > 1:
        print('Preprocessing env')
        env = Vectorize(env)
        env = AtariRescale42x42(env)
        env = NormalizedEnv(env)
        env = Unvectorize(env)
    else:
        print('No preprocessing because env is too small')
    return env


def _process_frame42(frame):
    frame = frame[34:34 + 160, :160]
    # Resize by half, then down to 42x42 (essentially mipmapping). If
    # we resize directly we lose pixels that, when mapped to 42x42,
    # aren't close enough to the pixel boundary.
    frame = cv2.resize(frame, (80, 80))
    frame = cv2.resize(frame, (42, 42))
    frame = frame.mean(2)
    frame = frame.astype(np.float32)
    frame *= (1.0 / 255.0)
    frame = np.reshape(frame, [1, 42, 42])
    return frame

def _process_frame42(frame):
    # Resize the frame to 42x42
    frame = cv2.resize(frame, (42, 42), interpolation=cv2.INTER_AREA)
    # Normalize pixel values to [0, 1]
    frame = frame / 255.0
    # Add a channel dimension if needed
    if len(frame.shape) == 2:  # If the frame is grayscale
        frame = frame[:, :, np.newaxis]
    return frame

class AtariRescale42x42(ObservationWrapper):
    def __init__(self, env):
        super(AtariRescale42x42, self).__init__(env)
        self.observation_space = Box(0.0, 1.0, (42, 42, 1), dtype=np.float32)

    def observation(self, observation):
        return _process_frame42(observation)


class NormalizedEnv(ObservationWrapper):
    def __init__(self, env):
        super(NormalizedEnv, self).__init__(env)
        self.state_mean = 0
        self.state_std = 0
        self.alpha = 0.9999
        self.max_episode_length = 0

    def observation(self, observation):
        # Update statistics
        self.max_episode_length += 1
        self.state_mean = self.state_mean * self.alpha + observation.mean() * (1 - self.alpha)
        self.state_std = self.state_std * self.alpha + observation.std() * (1 - self.alpha)

        denom = (1 - np.power(self.alpha, self.max_episode_length))
        unbiased_mean = self.state_mean / denom
        unbiased_std = self.state_std / denom

        # Normalize the observation
        return (observation - unbiased_mean) / (unbiased_std + 1e-8)