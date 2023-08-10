import jax.numpy as jnp
from jax.lax import cond, conv, fori_loop
import numpy as np
import sys, os
import jax
from jax import vmap

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



def update_one_flip_action_linear(state, action):
    """Applies the update_one_flip action to states that are not in thier linear representation.
    E.g. for 2D, L=4:
    1. input s with shape(4,4)
    2. -> shape(16,)
    3. update -> s' with shape(16,)
    4. -> shape(4,4) for consistency
    """
    state_shape = state.shape
    print(action)
    new_state = update_one_flip_action(jnp.ravel(state), action)
    return jnp.reshape(new_state, state_shape)

def random_initial_state(key, config):
    """Generates a random initial state for the environment, with each site iid s ~ {0,1}.

    Args:
        key (jax.random.PRNGKey): random key
        config (dict): configuration dictionary

    Returns:
        state (array): DeviceArray containing the state of the system
    """
    # Create an n-dimensional array of zeros
    desired_shape = (config["L"],) * config["D"]

    # Convert each element to uniformly chosen 0 or 1
    key, subkey = jax.random.split(key)  # Random number generator key
    uniform_array = jax.random.randint(subkey, shape=desired_shape, minval=0, maxval=2, dtype=jnp.int64)

    return uniform_array

def get_possible_states(state, config):
    """Return array of possible states, corresponding to each action.
    """
    num_sites = config["L"]**config["D"]
    possible_states = jnp.zeros((num_sites+1,)+state.shape)

    def loop_body(index, loop_carry):
        state, possible_states = loop_carry
        new_state = update_one_flip_action_linear(state, jnp.array([index,]))
        possible_states = possible_states.at[index].set(new_state)
        return (state, possible_states)

    carry = (state, possible_states)

    _, possible_states = fori_loop(0, num_sites+2, loop_body, carry)

    return possible_states

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

def magnetisation(s_t, a_t, s_tp1):
    """Computes the average spin-magnetisation for s_t"""
    # counts number of spins
    return jnp.mean(2*(s_t - 0.5))

def logp_state_proposal(s_t, s_tp1, config):
    """phi(s'|s) = the probability of proposing state s_tp1 in state s_t.
    Here, probability is uniform over all sites plus the "no flip" case.

    E.g. for 4x4 grid there are 16+1 = 17 possibilites, with 1/17 prob each.
    """
    L = config["L"]
    D = config["D"]
    return jnp.log(1/(L**D + 1))

def logp_state_proposal_vmapped(s_t, batch_s_tp1, config):
    """vmapped version of logp_state_proposal over s_tp1 -> batch_s_tp1
    Use to compute phi(s'|s) over a batch of s' (e.g. all possible s -> s')
    """
    fun_vmapped = vmap(lambda new_state: logp_state_proposal(s_t, new_state, config)) # maps the function to a batch of new states
    res = fun_vmapped(batch_s_tp1)
    return res

def log_ratio_target_dist(s_t, s_tp1, config):
    """Log-ratio of unnormalised target prob-dist:
    = log[tilde{q}(s_tp1)/tilde{q}(s_t)] where tilde{q}(s) is the target dist (steady-state)
     of the Markov Chain produced by MH-MCMC

    For the Ising Model (exponential distribution with fixed energy) then:

    log[R] = -beta*delta_eps ~,

    where:
        R: the ratio of (unnormed) prob-dists
        beta: the inverse temp (i.e. the parameter for the energy)
        delta_eps: the energy difference
    """
    s_t_energy = get_energy(s_t, config["kern"]) # Gets energy for current state = log[tilde{q}(s)]
    s_tp1_energy = get_energy(s_tp1, config["kern"]) # Gets energy for the next state = log[tilde{q}(s')]
    energy_difference = s_tp1_energy-s_t_energy # Compares energy difference
    return jnp.array(-config["temp"]*energy_difference, np.float64) #multiply by facotor -beta

def log_alpha(s_t, s_tp1, config):
    """Computes log[alpha] where alpha = tilde{q}(s')/tilde{q}(s) * phi(s|s')/phi(s'|s) is the
    Metropolis Hastings Factor (used in computing the acceptance probability)

    Note: if s=s' then alpha = 1.

    Args:
        s_t (_type_): _description_
        s_tp1 (_type_): _description_
        config (_type_): _description_

    Returns:
        _type_: _description_
    """
    log_proposal_forward = logp_state_proposal(s_t, s_tp1, config)
    log_proposal_backward = logp_state_proposal(s_tp1, s_t, config)
    log_ratio = log_ratio_target_dist(s_t, s_tp1, config)
    return log_ratio + log_proposal_backward - log_proposal_forward

def logp_acceptance(s_t, s_tp1, config):
    """Comutes the MH-MCMC acceptance log-prob, defined via,
    A = min[1, alpha] -> log[A] = min[0, log[alpha]]
    Note: A proposal of s'=s is always accepted

    """
    return jnp.amin(jnp.append(log_alpha(s_t, s_tp1, config), 0.0))

def logp_acceptance_vmapped(s_t, batch_s_tp1, config):
    """Batched version of logp_acceptance over argument s_tp1"""
    fun_vmapped = vmap(lambda new_state: logp_acceptance(s_t, new_state, config))
    res = fun_vmapped(batch_s_tp1)
    return res

def logp_ref(s_t, s_tp1, config):
    """Computes the log probability of the reference dynamics.
    Here q(s'|s) = MH-MCMC -> q*(s) where q*(s) is the Ising Model.
    The proposal used, phi(a|s), is a uniform single-site spin flip (including a single "no-flip")
    """

    pred = 1 - jnp.array_equiv(s_t, s_tp1) # the predicate: s' != s
    logp_accept = logp_acceptance(s_t, s_tp1, config)
    logp_prop = logp_state_proposal(s_t, s_tp1, config)

    # compute probs for all possible proposed changes to (effectively) normalise
    possible_states = get_possible_states(s_t, config)
    acceptance_probs = jnp.exp(logp_acceptance_vmapped(s_t, possible_states, config))
    rejection_probs = 1 - acceptance_probs
    proposal_probs = jnp.exp(logp_state_proposal_vmapped(s_t, possible_states, config))
    proposed_and_rejected_tot = jnp.sum(rejection_probs*proposal_probs) # includes phi(s|s) but has zero rejection-weight

    def true_fun(args): # s' != s
        # Only way to have s' != s is an accepted proposed change
        return logp_prop + logp_accept

    def false_fun(args): # s' == s
        # s'== s could be either from:
        #    1. rejection of a proposed state, tilde{s}.
        #    2. no change was proposed
        prob_proposal_no_change = proposal_probs[-1]
        return jnp.log(prob_proposal_no_change + proposed_and_rejected_tot)

    args = ()

    logp = cond(pred, true_fun, false_fun, args)

    return logp

def reward_components(s_t, a_t, s_tp1, config):
    """Return the reward components as a dict"""
    r_bias = -config["bias"] * config["obs_fn"](s_t, a_t, s_tp1)
    r_logp_ref = logp_ref(s_t, s_tp1, config)
    return {"r_bias": r_bias, "r_logp_ref": r_logp_ref}


def reward(s_t, a_t, s_tp1, config):
    """Return the reward components summed"""

    r_bias = -config["bias"] * config["obs_fn"](s_t, a_t, s_tp1)
    r_logp_ref = logp_ref(s_t, s_tp1, config)

    return r_bias + r_logp_ref

def step_fn(key, s_t, a_t, config):
    """This determinisitc step_fn simply applies the indicated single-site spin-flip.
    The reward_fn computes the log-prob of this action under a Metropolis Hastings Dynamics in the reward.
    For bias=0, subtracting the entropy of the policy from the reward gives the KL-divergence between the policy
    and the MH-MCMC reference dynamics.
    """

    s_tp1 = update_one_flip_action_linear(s_t, a_t)
    r_t = reward(s_t, a_t, s_tp1, config)

    return key, s_tp1, r_t

def get_kern_filter(dimensions):
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
    
    return kern

# %%
def alt_get_energy(state, dimensions):

    interaction_strength = 1 #J

    tmp  = 2*state - 1
    roll_rows = jnp.roll(tmp, 1, axis=0)
    roll_cols = jnp.roll(tmp, 1, axis=1)

    tmp1 = tmp*roll_rows #gives the col interactions
    tmp2 = tmp*roll_cols #gives the row interactions

    energy = -interaction_strength*jnp.sum(tmp1 + tmp2)

    return energy


# %%
def get_energy(lattice, kern):
    """Computes energy level for a given lattice of n dimensions

    Args:
        lattice array: Lattice of (0 and 1) or (-1 and 1)
        dimension int: Integer relating to the number of dimensions of the lattice

    Returns:
        int: Sum of the energy within the given lattice divided by 2 for twiddled
    """
    lattice = (lattice * 2) - 1
    extended_lattice = jnp.pad(lattice, 1, mode="wrap")

    # print(kern)
    # print(lattice)
    # print(extended_lattice)

    arr = -lattice * jax.scipy.signal.convolve(extended_lattice, kern, mode='valid', method="direct")
    # print(arr)
    return jnp.sum(arr)/2

# %%


# %%
import jax
from scipy.ndimage import convolve, generate_binary_structure
import numpy as np
import jax.numpy as jnp
from jax.lax import cond, fori_loop

lattice = jnp.array([[1,0,1],[0,1,1],[1,1,0],[0,1,0]])
# lattice = jnp.array([[1,1],[1,1]])
# lattice = jnp.array([[[1,1,1],[0,1,1],[1,1,0],[1,1,1]],[[1,1,1],[0,1,1],[1,1,0],[1,1,1]],[[1,1,1],[0,1,1],[1,1,0],[1,1,1]]])

# print(period_boundary_get_energy(lattice, 2))
# print(get_energy(lattice))

# %%
def policy_ref(key, state, config):
    """Reference Policy for MH-MCMC with uniform proposal.
    This performs MH-MCMC such that the steady-state/limiting distribution, q*(s) is the Ising model.
    """

    lattice_size = config["L"]**config["D"]

    key, subkey = jax.random.split(key)
    a_t = jax.random.randint(key=subkey, minval=0, maxval=config["L"]*config["D"]+2, shape=(1,), dtype=np.int64)
    proposed_state = update_one_flip_action_linear(state, a_t)

    prob_accept = jnp.exp(logp_acceptance(state, proposed_state, config))

    key, subkey = jax.random.split(key)
    pred = jax.random.uniform(subkey) <= prob_accept

    def true_cond(variables):
        """Proposal Accepted"""
        s_t, a_t, s_tp1 = variables
        return s_tp1, a_t

    def false_cond(variables):
        """Proposed Rejected"""
        s_t, a_t, s_tp1 = variables
        return s_t, jnp.array([lattice_size])

    new_state, action = cond(
        pred,
        true_cond,
        false_cond,
        (state, a_t, proposed_state)
    )

    logp = logp_ref(state, new_state, config)

    return logp, (action, key)


class IsingModel(BinarySpinsSingleFlip):
    """Gym Environment for the Ising Model.
    This is a Binary spin environment with few actions .
    This is an unconstrained model.
    An action corresponds to flipping a single spin, or flipping no spins.
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

        self.state = jnp.array(s_tp1)

        return s_tp1, r_t, terminated, truncated, info

    def render(self, mode=None):
        if mode is not None:
            self.render_mode = mode