"""Contains Basic Environments for Binary-Spin Systems"""
from jax.lax import cond
import jax
from gymnasium import spaces
import numpy as np
from gymnasium import Env
import jax.numpy as jnp
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.utils import (
    assert_config_has_keys,
    assert_config_values_are_even,
)

from components.spaces import (
    binary_actions_space,
    binary_actions_space_half,
    binary_spins_space,
    binary_spins_with_binary_flag,
    one_action_per_site_space,
)
from components.updates import (
    update_many_flip_action,
    update_one_flip_action,
    update_pair_flip_action,
    update_pair_flip_action_even,
    update_pair_flip_action_odd,
)

jax.config.update("jax_enable_x64", True)

class BinarySpinsFewActions(Env):
    """Basic Gym Environment for Binary Spin systems with "few actions" (one per site, plus one for no change)"""

    def check_config(self, config):
        assert_config_has_keys(config, ["L", "render_mode"])
        return config

    def __init__(self, config):
        self.config = self.check_config(config)

        super().__init__()

        self.observation_space = binary_spins_space(config["L"])
        self.action_space = one_action_per_site_space(config["L"])

        self.state = None
        self.render_mode = config["render_mode"]

    def reset(self):
        pass

    def step(self, action):
        pass

    def render(self, mode=None):
        pass


class BinarySpinsSingleFlip(BinarySpinsFewActions):
    """Basic Gym Environment for Binary Spin systems with "single flip" action"""

    def __init__(self, config):
        self.config = self.check_config(config)
        super().__init__(config)
        self.update = update_one_flip_action

    def _reward_func(self, st, at, stp1, config=None):
        """Placeholder to implement the return of a reward given"""
        return 0.0

    def step(self, action):
        state = self.state

        new_state = self.update(state, action)
        terminated = False
        truncated = False
        info = {}
        reward = self._reward_func(state, new_state, action, self.config)

        self.state = new_state

        return new_state, reward, terminated, truncated, info


class BinarySpinsPairFlip(BinarySpinsFewActions):
    """Basic Gym Environment for Binary Spin systems with "pair flip" action"""

    def __init__(self, config):
        self.config = self.check_config(config)
        super().__init__(config)
        self.update = update_pair_flip_action

    def _reward_func(self, st, at, stp1, config=None):
        """Placeholder to implement the return of a reward given"""
        return 0.0

    def step(self, action):
        state = self.state

        new_state = self.update(state, action)
        terminated = False
        truncated = False
        info = {}
        reward = self._reward_func(state, new_state, action, self.config)

        self.state = new_state

        return new_state, reward, terminated, truncated, info


class BinarySpinsAlternatingActions(Env):
    """Basic Gym Environment for Binary Spin systems with "many actions" using an alternating scheme (each site can be flipped)

    This environment has the same action space as the fully parallel case, but with an extra binary flag in the observation space.

    """

    def check_config(self, config):
        assert_config_has_keys(config, ["render_mode"])
        assert_config_values_are_even(config, ["L"])
        return config

    def __init__(self, config):
        self.config = self.check_config(config)

        super().__init__()

        self.observation_space = binary_spins_with_binary_flag(config["L"])
        self.action_space = binary_actions_space(config["L"])

        self.state = None

    def reset(self):
        pass

    def step(self, action):
        pass

    def render(self, mode=None):
        pass


class BinarySpinsSingleFlipAlternating(BinarySpinsAlternatingActions):
    """Basic Gym Environment for Binary Spin systems with single-site flips applied in an even-odd
    alternating fashion

    """

    def __init__(self, config):
        self.config = self.check_config(config)
        super().__init__(config)
        self.update = update_many_flip_action

    def _reward_func(self, st, at, stp1, config=None):
        """Placeholder to implement the return of a reward given"""
        return 0.0

    def step(self, action):
        spin_state = self.state["spin_state"]
        is_even = self.state["is_even"]

        new_spin_state = self.update(spin_state, action)
        terminated = False
        truncated = False
        info = {}
        reward = self._reward_func(spin_state, new_spin_state, action, self.config)

        new_state = {"spin_state": new_spin_state, "is_even": 1 - is_even}

        self.state = new_state

        return new_state, reward, terminated, truncated, info
