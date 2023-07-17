import jax.numpy as jnp
from jax.lax import cond, conv, fori_loop
import numpy as np
import sys, os
import jax

from gymnasium import spaces

jax.config.update("jax_enable_x64", True) # 64 precision helps prevent under and overflow

# pylint: disable=wrong-import-position
# pylint: disable=import-error

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from components.spin_models import BinarySpinsSingleFlip
from components.updates import update_one_flip_action
from scipy.ndimage import convolve, generate_binary_structure

from utils.utils import (
    assert_config_has_keys,
    assert_config_values_are_even,
    get_odd_or_even,
    pad_actions,
    fix_config_and_jit,
)

# pylint: enable=wrong-import-position
# pylint: enable=import-error


# def random_initial_state(key, config):
#     """Generates a random initial state for the environment.
#     State has at least one particle in it, so that the dark state is not created.

#     Args:
#         key (jax.random.PRNGKey): random key
#         config (dict): configuration dictionary

#     Returns:
#         state (array): DeviceArray containing the state of the system
#     """
#     key, subkey = jax.random.split(key)

#     num_particles = jax.random.randint(subkey, [1], 1, config["L"]**config["D"] + 1).item()

#     initial_state = jnp.array([1] * num_particles + [0] * (config["L"]**config["D"] - num_particles))

#     print(config["L"]**config["D"])
#     print(num_particles)
#     print(len(initial_state))

#     key, subkey = jax.random.split(key) # add dtype argument (64)
#     initial_state = jax.random.permutation(subkey, initial_state, independent=True)

#     initial_state = jnp.reshape(initial_state,(config["L"], -1))

#     print(initial_state)


#     return initial_state

def random_initial_state(key, config):
    """Generates a random initial state for the environment.
    State has at least one particle in it, so that the dark state is not created.

    Args:
        key (jax.random.PRNGKey): random key
        config (dict): configuration dictionary

    Returns:
        state (array): DeviceArray containing the state of the system
    """
    # Create an n-dimensional array of zeros
    desired_shape = (config["L"],) * config["D"]
    zeros_array = jnp.zeros(desired_shape)
    
    print(config["L"]**config["D"])
    print(desired_shape)

    # Convert each element to uniformly chosen 0 or 1
    key, subkey = jax.random.split(key)  # Random number generator key
    uniform_array = jax.random.randint(subkey, shape=zeros_array.shape, minval=0, maxval=2, dtype=jnp.uint8)

    print(uniform_array)


    return uniform_array


def constraint(state, action):

    return 1

def activity(s_t, a_t, s_tp1):
    """Counts the number of spin-flips
    For spin-flips \kappa[i] = 1 if s_tp1[i] \neq s_t[i], and zero otherwise).

    Args:
        s_t (array): set of states at time t. shape = (L,).
        a_t (array): Action chosen at time t. shape = (1,).
        s_tp1 (array): set of states at time tp1. shape = (L,).

    Returns:
        k_t (array): the activity for the step t->tp1.
    """

    return jnp.sum(jnp.not_equal(s_t, s_tp1))


def logp_ref(s_t, a_t, config):
    # """Calculate the log-probability of an action in the original (reference) dynamics.

    # Args:
    #     s_t (array): set of states at time t. shape = (L,).
    #     a_t (array): Action chosen at time t. shape = (1,).
    #     config (dict): environment configuration.
    # """
    logp = cond(
        jnp.array_equiv(a_t, config["L"]),  # no spin flip chosen
        lambda x: jnp.log(1 - jnp.sum(x) / config["L"]),  # logp for no-flip
        lambda x: -jnp.log(config["L"]),  # logp for flip
        s_t,
    )
    return logp
    # return 1


def logp_prob(s_t, a_t, config):
    # two objects to calculate prob for,
    return 1

def logp_prop(s_t, a_t, config):
    # probability of the action happening
    return 1


def reward_components(s_t, a_t, s_tp1, config):
    """Return the reward components as a dict"""
    r_bias = -config["bias"] * config["obs_fn"](s_t, a_t, s_tp1)
    r_logp_ref = logp_ref(s_t, a_t, config)
    return {"r_bias": r_bias, "r_logp_ref": r_logp_ref}


def reward(s_t, a_t, s_tp1, config):
    """Return the reward components summed"""

    r_bias = -config["bias"] * config["obs_fn"](s_t, a_t, s_tp1)
    r_logp_ref = logp_ref(s_t, a_t, config)

    # return r_bias + r_logp_ref
    return 5


def step_fn(key, s_t, a_t, config):
    """Performs the alternating EM step to update the current state and release the reward"""

    ## start here for us
    # calc unnormalised prob for state
    # sample from proposal (random spin)
    # calc alpha function (prob of proposal)
    # state, action--- is the flip possible


    s_tp1 = update_one_flip_action(s_t, a_t)
    s_t_energy = get_energy(s_t, config["D"]) # Gets energy for current state
    s_tp1_energy = get_energy(s_tp1, config["D"]) # Gets energy for the next state
    energy_difference = s_tp1_energy-s_t_energy # Compares energy difference

    key, subkey = jax.random.split(key)

    def true_cond(s_t, a_t, s_tp1, config):
        s_t = s_tp1
        r_t = reward(s_t, a_t, s_tp1, config)

        return s_t, r_t

    def false_cond(s_t, a_t, s_tp1, config):
        s_tp1 = s_t
        r_t = reward(s_t, config["L"], s_tp1, config)

        return s_tp1, r_t


    G = ((energy_difference > 0) & (jax.random.uniform(subkey) < jnp.exp(-config["temp"]*energy_difference))) | (energy_difference <= 0)
    s_tp1 = G*s_tp1 -(G-1)*s_t
    r_t = G*reward(s_t, a_t, s_tp1, config) -(G-1)* reward(s_t, config["L"], s_tp1, config)

    # s_tp1, r_t = cond(
    #     True,
    #     true_cond,
    #     false_cond,

    #     s_t, a_t, s_tp1, config
    # )


    return key, s_tp1, r_t

def metropolis():


    return

# %%
def get_energy(lattice, dimensions):

    lattice = (lattice * 2) - 1

    kern = jnp.zeros([3]*dimensions, bool)

    def run_energy(n, init_val):
        kern = init_val
        b = jnp.array([1]*dimensions)
        c = jnp.array([1]*dimensions)

        b = b.at[n].add(-1)
        c = c.at[n].add(1)

        b = tuple(b)
        c = tuple(c)
            
        kern = kern.at[b].set(True)        
        kern = kern.at[c].set(True)

        return kern

    kern = fori_loop(0, dimensions, run_energy, kern)

    arr = -lattice * jax.scipy.signal.convolve(lattice, kern, mode='same', method="direct")
    return jnp.sum(arr)



# %%
# from scipy.ndimage import convolve, generate_binary_structure
# import numpy as np
# import jax.numpy as jnp


# lattice = jnp.array([[1,1,1],[0,1,1],[1,1,0],[1,1,1]])
# # print(lattice)
# get_energy(lattice, 2)


# %%
def policy_ref(key, state, config):
    """Reference policy for the few-action EM.
    A site is selected at random, uniformly.
    If it can flip, it does.
    Otherwise no action is performed
    """

    key, subkey = jax.random.split(key)

    L = config["L"]
    site_idx = jax.random.randint(
        subkey, (1,), 0, L
    )  # choose site at random with equal prob
    has_left = state[
        (site_idx - 1) % L
    ]  # action avail if site has left spin up (i.e. occupied)

    action = jnp.where(has_left, site_idx, L)

    logp = logp_ref(state, action, config)

    return logp, (action, key)


class IsingModel(BinarySpinsSingleFlip):
    """Gym Environment for the East Model.
    This is a Binary spin environment with few actions and a local kinetic constraint.
    An action corresponds to flipping a single spin.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}

    def check_config(self, config):
        """Check config dict for required vars"""
        assert_config_has_keys(config, ["bias", "L", "render_mode", "obs_fn"])
        assert_config_values_are_even(config, ["L"])
        return config

    def __init__(self, config, render_mode=None, seed=123, key=None):
        """_summary_

        Args:
            config (_type_): _description_
            render_mode (_type_, optional): _description_. Defaults to None.
            seed
            key
        """

        self.config = self.check_config(config)
        super().__init__(config)

        self.action_space = spaces.Discrete(config["L"]**config["D"] + 1)
        self.observation_space = spaces.Box(0,1,(config["L"]**config["D"],))


        if key is not None:
            self.key = key
        elif seed is not None:
            self.key = jax.random.PRNGKey(seed)
        else:
            raise ValueError("Must provide a key (preferentially) or a seed on init")

        # JIT THE STEP, REFERENCE DYNAMICS AND PRED_ACTION_IS_AVAILABLE
        self.step_fn_jit = fix_config_and_jit(step_fn, config)
        self.constraint_jit = jax.jit(constraint)
        self.policy_ref_jit = fix_config_and_jit(policy_ref, config)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode


        self.state = None

    def reset(self):
        self.key, subkey = jax.random.split(self.key)
        initial_state = random_initial_state(subkey, self.config)

        self.state = initial_state
        info = {}

        return initial_state, info

    def step(self, action):
        # action = action.item()  # just use the scalar part

        constraint_reward = 0.0
        s_t = self.state

        self.key, s_tp1, r_t = self.step_fn_jit(self.key, s_t, action)
        r_t += constraint_reward

        action_avail = self.constraint_jit(s_t, action)

        terminated = not action_avail
        truncated = False
        info = {"action_available": action_avail}

        self.state = s_tp1

        return s_tp1, r_t, terminated, truncated, info

    def render(self, mode=None):
        if mode is not None:
            self.render_mode = mode