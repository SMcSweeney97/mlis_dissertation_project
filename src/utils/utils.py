import jax
import jax.numpy as jnp
from jax import vmap
from jax.lax import cond
from functools import partial
import numpy as np

def get_haiku_parameter_shapes(params):
    """Make printing parameters a little more readable.
    Input the pyTree of params (i.e. weights)
    """
    return jax.tree_util.tree_map(lambda p: p.shape, params)


def available_actions_mask(state, actions, constraint):
    """For a given state, actions and constraint function, return [C(a,s) for a in actions].
    Note this will be inefficent for large number of actions e.g. if actions = action_space and have num_actions ~ exp(num_agent)

    Args:
        state (_type_): _description_
        actions (array): DeviceArray of all actions in the action_space (or all actions for C(a,s) to be evaluated against)
        constraint_fn (function): function implementing C(a,s). This should have the signiture constraint(state, action)-> DeviceArray(Bool)

    Returns:
        mask: DeviceArray() shape=(L). E.g. for L=4, s = [0,1,0,0], DeviceArray([ True,  True, False, False], dtype=bool)
    """

    constraint_batched = vmap(lambda action: constraint(state, action))

    mask = constraint_batched(actions)

    return mask


def assert_config_has_keys(config, keys):
    """checks that env config dict contains the keys in the list"""

    for key in keys:
        assert (
            key in config.keys()
        ), f"arg {key} does not appear in config and is required by environment"


def assert_config_values_are_even(config, keys):
    """checks that the env config dict has the key and its value is even"""

    assert_config_has_keys(config, keys)

    for key in keys:
        is_even = config[key] % 2 == 0
        assert (
            is_even
        ), f"config variable, {key}, must have an even value for this environment"


def get_even(array):
    """Extract the even indices of an array."""
    return array[0::2]


def get_odd(array):
    """Extract the odd indices of an array."""
    return array[1::2]


def get_odd_or_even(array, is_even):
    """Extract the odd or even indices of an array depending on is_even."""
    return cond(is_even, get_even, get_odd, array)


def pad_actions_even(actions):
    """Pad an array of even site actions with 0 actions on the odd sites.
    E.g. a = [1,1] -> [1,0,1,0]
    """
    return jnp.column_stack(
        [actions, jnp.zeros(len(actions), dtype=actions.dtype)]
    ).reshape(-1)


def pad_actions_odd(actions):
    """Pad an array of odd site actions with 0 actions on the even sites.
    E.g. a = [1,1] -> [1,0,1,0]
    """
    return jnp.column_stack(
        [jnp.zeros(len(actions), dtype=actions.dtype), actions]
    ).reshape(-1)


def pad_actions(actions, is_even):
    """Pad an array of site actions on is_even with 0 actions."""
    return cond(is_even, pad_actions_even, pad_actions_odd, actions)


def _create_padding(half_system_size, action_dimension):
    padder = jnp.zeros(action_dimension, dtype=jnp.bool_).at[0].set(True)
    return jnp.repeat(padder.reshape((1, action_dimension)), half_system_size, 0)


def pad_valid_even(valid_actions):
    """Pad an array indicating valid even site actions.

    Pads with additional rows indicating the only valid action is 0 on odd sites.
    """
    half_sys_size, action_dim = valid_actions.shape
    padding = _create_padding(half_sys_size, action_dim)
    return jnp.stack([valid_actions, padding], 1).reshape(
        (2 * half_sys_size, action_dim)
    )


def pad_valid_odd(valid_actions):
    """Pad an array indicating valid odd site actions.

    Pads with additional rows indicating the only valid action is 0 on even sites.
    """
    half_sys_size, action_dim = valid_actions.shape
    padding = _create_padding(half_sys_size, action_dim)
    return jnp.stack([padding, valid_actions], 1).reshape(
        (2 * half_sys_size, action_dim)
    )


def pad_valid(valid_actions, is_even):
    """Pad an array indicating valid actions on sites indicated by is_even.

    Pads with additional rows indicating the only valid action is 0 on remaining sites.
    """
    return cond(is_even, pad_valid_even, pad_valid_odd, valid_actions)

def fix_config_and_jit(func, config):
    """Fix the environment configation and jit the resulting function
       To work, config must be the last positional argument of the func.
    """
    return jax.jit(partial(func, config=config))