# %%
import os, sys

from fix_pathing import root_dir

from src.utils.animation import make_animation_vertical, make_animation_horizontal
from src.envs.ising_model_1d.ising_model import IsingModel, activity, magnetisation, get_possible_states, logp_state_proposal_vmapped, log_ratio_target_dist, logp_acceptance_vmapped, get_energy

from src.utils.plotting import render_spin_trajectory, plot_learning_curve
import haiku as hk
import matplotlib.pyplot as plt
import numpy as np
import time
from jax import jit
import jax.numpy as jnp
from functools import partial
import jax.random as jr
from jax import vmap
import itertools

from src.utils.evaluation import (
    evaluate_policy_biases,
)
import seaborn as sns
import pandas as pd
# %% INIT ENV
rng = hk.PRNGSequence(456)
env_seed = 123
config = {"L": 2, "bias": 0, "d": 2, "D":2, "temp":0.5, "render_mode": None, "obs_fn": activity, "mean": 0}
env = IsingModel(config, seed=env_seed)
# %% EXACT PROBS - only run this for small systems...

def get_all_states(config):

    all_states = []

    site = [0,1]
    sites = [site for _ in range(config["L"]**config["D"])]
    for state in itertools.product(*sites):
        all_states.append(np.reshape(state, (config["L"], config["L"])))

    all_states = np.array(all_states)

    return all_states

def get_energies(states, config):

    return [get_energy(state, config["D"]) for state in states]

def get_probs(energies, config):

    probs = [np.exp(-config["temp"]*energy) for energy in energies]

    probs = np.array(probs)/np.sum(probs)

    return probs

def get_probs_batch(temps, states, config):

    config = dict(config) # will modify so copy this

    energies = get_energies(states, config)

    probs_batch = []

    for i in range(len(temps)):

        temp = temps[i]

        if temp == 0:
            temp += 1e-1

            temps[0] = temp

        config["temp"] = 1/temp

        probs_batch.append(get_probs(energies, config))

    return temps, np.array(probs_batch)

def get_abs_mags(states):
    return [np.abs(magnetisation(state, None, None)) for state in states]

def expected_mag(mags, probs):
    return np.sum(mags*probs)

def expected_mag_batch(mags, probs_batch):
    return np.sum(mags*probs_batch, axis=1)

states = get_all_states(config)
energies = get_energies(states, config)
probs = get_probs(energies, config)
mags = get_abs_mags(states)
temps, probs_batch = get_probs_batch(np.arange(0, 6), states, config)
expected_mags = expected_mag(mags, probs)
expected_mags_batch = expected_mag_batch(mags, probs_batch)

fig, ax = plt.subplots(1, 1)

ax.plot(temps, expected_mags_batch, "b^--")

mag_df = pd.DataFrame({"T":temps, "M":expected_mags_batch})

# %% ISING - EXPLICIT TEST OF MCMC STEP FOR DIAGNOSTICS

NUM_STEPS = 1
start_time = time.time()

(
    _,
    _,
) = env.reset()   #STATE IS INITIALISED
states_cache = [env.state]
env_rewards_cache = []
rewards_cache = []
activity_cache = []
magnetisation_cache = []
actions_cache = []
current_cache = []
logp_cache = []
s_t = env.state

activity_jit = jit(activity)  #activity counts flips
magnetisation_jit = jit(magnetisation)

for t in range(NUM_STEPS):

    # print(t)

    # TO CHECK - ENERGY DIFFERENCE SEEM VERY LARGE?
    possible_states = get_possible_states(s_t, config)
    energy_diffs = vmap(lambda proposed_state: log_ratio_target_dist(s_t, proposed_state, config))(possible_states)
    proposal_probs = jnp.exp(logp_state_proposal_vmapped(s_t, possible_states, config))
    acceptance_probs = jnp.exp(logp_acceptance_vmapped(s_t, possible_states, config))


    logp, (a_t, _) = env.policy_ref_jit(next(rng), s_t)

    actions_cache.append(a_t)
    logp_cache.append(logp)  # entropy

    s_tp1, r_t, terminated, truncated, info = env.step(a_t)

    # print("...", logp, r_t, a_t[0])

    env_rewards_cache.append(r_t) #log[q] + r_obs
    rewards_cache.append(r_t - logp)  # subtract entropy term, logp

    states_cache.append(s_tp1)
    activity_cache.append(activity_jit(s_t, a_t, s_tp1))
    magnetisation_cache.append(magnetisation_jit(s_t, a_t, s_tp1))

    # s_t = s_tp1

loop_time = time.time() - start_time
time_per_step = loop_time / NUM_STEPS
est_time_full_run = time_per_step * 10**6 / 3600  # time for 1 million steps in hours
print(f"{loop_time*1000:.2f}", "ms")
# %% ISING

NUM_STEPS = 10000
start_time = time.time()

(
    _,
    _,
) = env.reset()   #STATE IS INITIALISED
states_cache = [env.state]
env_rewards_cache = []
rewards_cache = []
activity_cache = []
magnetisation_cache = []
actions_cache = []
current_cache = []
logp_cache = []
s_t = env.state

activity_jit = jit(activity)  #activity counts flips
magnetisation_jit = jit(magnetisation)

for t in range(NUM_STEPS):

    # print(t)

    logp, (a_t, _) = env.policy_ref_jit(next(rng), s_t)

    actions_cache.append(a_t)
    logp_cache.append(logp)  # entropy

    s_tp1, r_t, terminated, truncated, info = env.step(a_t)

    # print("...", logp, r_t, a_t[0])

    env_rewards_cache.append(r_t) #log[q] + r_obs
    rewards_cache.append(r_t - logp)  # subtract entropy term, logp

    states_cache.append(s_tp1)
    activity_cache.append(activity_jit(s_t, a_t, s_tp1))
    magnetisation_cache.append(magnetisation_jit(s_t, a_t, s_tp1))

    s_t = s_tp1

loop_time = time.time() - start_time
time_per_step = loop_time / NUM_STEPS
est_time_full_run = time_per_step * 10**6 / 3600  # time for 1 million steps in hours
print(f"{loop_time*1000:.2f}", "ms")
# %% Some Snapshots of the state
num_snapshots = 4
dt_snapshots = 3
fig, axs = plt.subplots(2, num_snapshots, squeeze=False)
for i in range(0, num_snapshots):
    t_idx = i*dt_snapshots
    axs[0][i].imshow(states_cache[t_idx], cmap="gray_r")
    if i > 0:
        diff = np.array(states_cache[t_idx] - states_cache[t_idx-1])
        axs[1][i].imshow(diff)
    else:
        axs[1][i].imshow(np.zeros_like(states_cache[0]))
# %% Policy Magnetisation. 2D Ising model has phase transition around T = 2.
# For infinite size system, below this mag = 1, above this mag = 0
exact_result = (mag_df[mag_df["T"]==1/config["temp"]])["M"]
plot_learning_curve(np.abs(magnetisation_cache))
plt.plot([], [], "r-", linewidth=4, label="MCMC")
plt.title(f"T = {1/config['temp']:.2f}, M = {np.mean(np.absolute(magnetisation_cache)[600::])}", fontsize=20)
plt.hlines(exact_result, 0, len(magnetisation_cache), label="Exact", linewidth=4, linestyle="dashed")
plt.legend(fontsize=20)
plt.show()
# %%

fig, ax = plt.subplots(1, 1)

ax.plot(temps, expected_mags_batch, "b^--", label="Exact")
ax.plot([1/config["temp"]], [np.mean(np.absolute(magnetisation_cache)[600::])], "ks", label="MCMC")
ax.legend()
# %%
