"""Contains updates for physical spaces"""
from jax.lax import cond
import jax
import jax.numpy as jnp
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

jax.config.update("jax_enable_x64", True)


# def _flip(state, site):
#     """Flips the spin at the indicated site: s_i -> 1-s_i"""
#     state = state.at[site].set(1 - state[site])
#     return state


def _flip(state, action):
    """Flips the spin at the indicated site: s_i -> 1-s_i where action = (site,) i.e. a shape=(1,) array."""
    state = state.at[action].set(1 - state[action])
    return state


def update_one_flip_action(state, action):
    """Updates the physical-state using the single-spin flip action: s_i -> 1 - s_i

    Args:
        state (array): an array of s_i in {0,1} with shape (L,1). The state of the spin-system.
        action (array): an shape (1,) array with int element indicating the spin to flip (i = 0, 1, ..., L-1)
                        int=L meaning no flip.

    Returns:
        state: the new state
    """
    lattice_size = len(state)

    state = cond(
        jnp.array_equiv(action, lattice_size),
        lambda x: x,
        lambda x: _flip(x, action),
        state,
    )
    return state


def update_many_flip_action(state, action):
    """Updates the physical-state by flipping all spins according to: s_i -> xor(s_i, a_i)

    Args:
        state (array): an array of s_i in {0,1} with shape (L,1). The state of the spin-system.
        action (array): an array of s_i in {0,1} with shape (L,1). A value of 0 indicates no flip on the corresponding site while a value of 1 indicates a flip.

    Returns:
        state: the new state
    """
    return jnp.abs(jnp.subtract(state, action))


def update_pair_flip_action(state, action):
    """Updates the physical-state using the pair flip action: (s_i, s_{i+1}) -> (1-s_i, 1-s_{i+1}).
        PBs are assumed.

    Args:
        state (array): an array of s_i in {0,1} with shape (L,1). The state of the spin-system.
        action (int): an integer indicating which spin should be flipped (i = 0, 1, ..., L-1), with int=L meaning no flip.
        environment_args (dict): a dictionary of environment arguments.

    Returns:
        state: the new state
    """
    lattice_size = len(state)

    state = cond(
        jnp.array_equiv(action, lattice_size),
        lambda x: x,
        lambda x: _flip(_flip(x, action), (action + 1) % lattice_size),
        state,
    )
    return state


def update_pair_flip_action_even(state, action_even):
    """Updates the physical-state by flipping adjacent pairs of spins where the left-site is even, i.e. [0,1], [2,3] ...

    E.g. a = [1,0,1] for L = 6 would flip sites [0,1] and [4,5]

    For this purpose, the odd entries of action are ignored.

    ASSUMES NUMBER OF SITES IS EVEN - THIS FUNCTION DOES NOT ACT OVER BOUNDARIES


    E.g. L = 6.

    0   1   2   3    4   5
    |   |   |   |    |   |
    [ G ]   [ G ]    [ G ]
    |   |   |   |    |   |

    Args:
        state (_type_): _description_
        action_even (array): shape (L/2) array. The even componets [0, 2, 4, ...] indicate whether to flip the corresponding pairs

    Returns:
        _type_: _description_
    """

    flip_action = jnp.column_stack([action_even, action_even]).reshape(
        -1
    )  # flip the pairs
    state = jnp.abs(jnp.subtract(state, flip_action))
    return state


def update_pair_flip_action_odd(state, action_odd):
    """Updates the physical-state by flipping adjacent pairs of spins where the left-site is odd.
    Uses periodic boundary conditions.

    E.g. a = [1,0,1] for L = 6 would flip sites [1,2] and [5,0] (here site 6 = site 0 via PBCs)


    For this purpose, the even entries of action are ignored.

    E.g. L = 6.

       0   1   2   3   4   5
       |   |   |   |   |   |
    ...]   [ G ]   [ G ]   [ G ...
       |   |   |   |   |   |

    Args:
        state (_type_): _description_
        action (array): shape (L) array. The even componets [0, 2, 4, ...] indicate whether to flip the corresponding pairs

    Returns:
        _type_: _description_
    """

    flip_action = jnp.column_stack([action_odd, action_odd]).reshape(
        -1
    )  # flip the pairs
    flip_action = jnp.roll(flip_action, 1)  # for pbc
    state = jnp.abs(jnp.subtract(state, flip_action))
    return state
