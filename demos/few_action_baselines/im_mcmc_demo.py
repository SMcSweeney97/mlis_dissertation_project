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

from src.utils.evaluation import (
    evaluate_policy_biases,
)
import seaborn as sns
# %% INIT ENV
rng = hk.PRNGSequence(456)
env_seed = 123
config = {"L": 4, "bias": 0, "d": 2, "D":2,"temp":0.2, "render_mode": None, "obs_fn": activity, "mean": 0}
env = IsingModel(config, seed=env_seed)
# %% TEST ENERGY FUNCTION

state, _ = env.reset()

print(state)
print(get_energy(state, 2))

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

NUM_STEPS = 100000
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
plot_learning_curve(np.abs(magnetisation_cache))
plt.title(f"T = {1/config['temp']:.2f}, M = {np.mean(np.absolute(magnetisation_cache)[600::])}", fontsize=20)
plt.show()

# %% PLOT CUMULATIVE MEAN OF REWARDS
fig_entropy = plot_learning_curve(
    -np.array(logp_cache), ylabel=r"$<\mathrm{Entropy}>_{1:t}$"
)
fig_rewards = plot_learning_curve(
    rewards_cache, ylabel=r"$<\mathrm{env-rewards}>_{1:t}$"
)
fig_rewards = plot_learning_curve(rewards_cache)
fig_current = plot_learning_curve(activity_cache, ylabel=r"$<\mathrm{activity}>_{1:t}$")

print("rb = ", np.mean(rewards_cache) - np.mean(logp_cache))
print(r"<\kappa> = ", np.mean(activity_cache))
# %% Histogram of Actions
actions_cache = np.ravel(actions_cache)
sns.displot(actions_cache, discrete=True)
# %% Frac of no-flip-actions
no_flip_actions = np.cumsum(np.array(actions_cache) == config["L"]**config["D"])
frac_no_flip = no_flip_actions/np.arange(1,len(no_flip_actions)+1)
plt.plot(frac_no_flip)
num_no_flip_actions = np.sum(np.array(actions_cache) == config["L"]**config["D"])
plt.title(f"Long-time no change frac = {num_no_flip_actions}/{len(actions_cache)} = {num_no_flip_actions/len(actions_cache):.2f}")
plt.ylim(0,1)
# %%
