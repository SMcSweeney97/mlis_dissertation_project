from gymnasium import Wrapper
import numpy as np

class Compat(Wrapper):
    """Compatibility wrapper for gym. By default stat-mech-gym envs return jax DeviceArrays, this wrapper ensures:
    1. obsrvations are Numpy arrays
    2. rewards are floats.

    This makes the environment compatible with typical RL libraries, e.g., stable-baselines3.


    Args:
        Wrapper (class): base class
    """
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        res = list(self.env.step(action))
        res[0] = np.array(res[0])
        res[1] = float(res[1])
        return tuple(res)

    def reset(self):
        res = list(self.env.reset())
        res[0] = np.array(res[0])
        return tuple(res)

class OldGym(Wrapper):
    """Compatibility wrapper for gym. By default stat-mech-gym envs used new gym return style. This wrapper ensures:
    1. step() returns s_tp1, r_t, done, info.
    2. reset() returns s_0.

    In new-style step() returns s_tp1, r_t, terminated, truncated, info and reset() returns s_0, info.

    This treats "terminated" as equivalent to "done". "truncated" is skipped.

    Args:
        Wrapper (class): base class
    """
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        res = list(self.env.step(action))
        return tuple([res[0], res[1], res[2], res[4]])

    def reset(self):
        res = list(self.env.reset())
        return res[0]