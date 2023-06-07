"""Contains Gym Spaces for Stat-Mech-Models"""

import jax
from gymnasium import spaces
import numpy as np
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

jax.config.update("jax_enable_x64", True)


def binary_spins_space(lattice_size):
    """Returns a gym space object representing binary spins. Sampling gives an array of integers, shape (L,)."""
    return spaces.Box(0, 1, (lattice_size,), dtype=np.int64)


def binary_spins_with_binary_flag(lattice_size):
    """Returns a composite gym space of binary spins and a single binary flag.
    E.g., used in spin models with alternating even-odd dynamics where binary flag = is_even
    """
    return spaces.Dict(
        {"is_even": spaces.Discrete(2), "spin_state": binary_spins_space(lattice_size)}
    )


def one_action_per_site_space(lattice_size):
    """Returns a gym space object representing a single (local) action per site. Sampling gives an integer between [0,L].
       By convention, action "L" does nothing to the state.
    E.g. this might represent:
        a) Single spin flips: action "i" flips spin at site "i" and L flips nothing.
        b) Pairs of spin flips: action "i" flips spins at sites "i" and "i+1%L", while "L+1" flips nothing.
    """
    return spaces.Box(0, lattice_size + 1, (1,), dtype=np.int64)
    # return spaces.Discrete(lattice_size + 1)


def binary_actions_space(lattice_size):
    """Returns a gym space object representing an exponential number of actions. a in {0,1}^{L}.
    E.g. for L = 4, there are 16 possible actions: a = [0,0,0,0], [0,0,0,1], [0,0,1,0], [0,0,1,1], ..., [1,1,1,1]#

    This space is identical to that of binary spins (see binary_spin_space)

    By convention, "0" does nothing, while "1" implements the local action. This may be, e.g.,
        a) Single spin flips: e.g. a = [1, 0] flips spin at site 0 and does nothing to site 1.
        b) Pairs of spin flips: e.g. a[i] = 1 flips spins at sites "i" and "i+1%L".

    """
    return spaces.Box(0, 1, (lattice_size,), dtype=np.int64)


def binary_actions_space_half(lattice_size):
    """Returns a gym space object representing many actions that act on either even or odd sites: a in {0,1}^{L/2}.

    Whether these act on the even or odd sites is determined by the environment's state (typically is_even observation)

    E.g. for L = 4, there are 2^(L/2)=4 possible actions: a = [0,0], [0,1], [1,0], [1,1]

    This space is identical to that of binary spins (see binary_spin_space)

    By convention, "0" does nothing, while "1" implements the local action. This may be, e.g.,
        a) Single spin flips: e.g. a = [1, 0] flips spin at site 0 and does nothing to site 2 (in case of even action)

    """
    return spaces.Box(0, 1, (int(lattice_size / 2),), dtype=np.int64)
