# %%
import os, sys

from fix_pathing import root_dir

from src.utils.animation import make_animation_vertical, make_animation_horizontal
from src.envs.ising_model_1d.ising_model_1d import IsingModel, activity

from src.utils.plotting import render_spin_trajectory, plot_learning_curve
import haiku as hk
import matplotlib.pyplot as plt
import numpy as np
import time
from jax import jit
import jax.numpy as jnp
from functools import partial
import jax.random as jr

from src.utils.evaluation import (
    evaluate_policy_biases,
)

# %% INIT ENV
rng = hk.PRNGSequence(123)
config = {"L": 20, "bias": 0, "d": 2, "D":2,"temp":0.2, "render_mode": None, "obs_fn": activity, "mean": 0}
env = IsingModel(config)


# %% ISING ATTEMPT 1

def ising_initial_policy(key,state,config):
    #chance_of_choice = 0
    #chosen_change= jnp.array(config["L"], dtype=jnp.int64)
    #return jnp.log(chance_of_choice), (chosen_change, key)
    key, subkey = jr.split(key)
    return jnp.log(1 / config["L"]), (
        jr.randint(subkey, [1], 1, config["L"] + 1, dtype=jnp.int64),
        key,
    )


ising_initial_policy_jit = jit(partial(ising_initial_policy, config=config))


NUM_STEPS = 1000
start_time = time.time()

(
    _,
    _,
) = env.reset()   #STATE IS INITIALISED
states_cache = [env.state]
env_rewards_cache = []
rewards_cache = []
activity_cache = []
current_cache = []
logp_cache = []
s_t = env.state

activity_jit = jit(activity)  #activity counts flips

for _ in range(NUM_STEPS):

    logp, (a_t, _) = ising_initial_policy_jit(next(rng), s_t)

    logp_cache.append(logp)  # entropy

    s_tp1, r_t, terminated, truncated, info = env.step(a_t)

    env_rewards_cache.append(r_t)
    rewards_cache.append(r_t - logp)  # subtract entropy term, logp

    states_cache.append(s_tp1)
    rewards_cache.append(r_t)

    activity_cache.append(activity_jit(s_t, a_t, s_tp1))

    s_t = s_tp1

loop_time = time.time() - start_time
time_per_step = loop_time / NUM_STEPS
est_time_full_run = time_per_step * 10**6 / 3600  # time for 1 million steps in hours
print(f"{loop_time*1000:.2f}", "ms")




# %% EXAMPLE OF TAKING NO ACTIONS EVERY TIME
def policy_no_action(key, state, config):
    """Policy that choosens "no update" action deterministically"""
    return jnp.log(1.0), (jnp.array(config["L"], dtype=jnp.int64), key)


policy_no_action_jit = jit(partial(policy_no_action, config=config))


NUM_STEPS = 1000
start_time = time.time()

(
    _,
    _,
) = env.reset()
states_cache = [env.state]
rewards_cache = []
current_cache = []
logp_cache = []
s_t = env.state

activity_jit = jit(activity)


for t in range(NUM_STEPS):
    logp, (a_t, _) = policy_no_action_jit(None, None)

    logp_cache.append(logp)

    s_tp1, r_t, terminated, truncated, info = env.step(a_t)

    states_cache.append(s_tp1)

    s_t = s_tp1

    if terminated or truncated:
        print(f"Terminated = {terminated} after {t+1} steps")
        break

loop_time = time.time() - start_time
time_per_step = loop_time / NUM_STEPS
est_time_full_run = time_per_step * 10**6 / 3600  # time for 1 million steps in hours
print(f"{loop_time*1000:.2f}", "ms")
# %% VIZUALISE TRAJECTORY
# fig_traj = render_spin_trajectory(states_cache)


# %% EXAMPLE OF TAKING COMPLETELY RANDOM ACTIONS
def policy_random_action(key, state, config):
    """Policy that choosens random action"""
    key, subkey = jr.split(key)
    return jnp.log(1 / config["L"]), (
        jr.randint(subkey, [1], 1, config["L"] + 1, dtype=jnp.int64),
        key,
    )
        

policy_random_action_jit = jit(partial(policy_random_action, config=config))


NUM_STEPS = 1000
start_time = time.time()

(
    _,
    _,
) = env.reset()
states_cache = [env.state]
rewards_cache = []
current_cache = []
logp_cache = []
s_t = env.state

activity_jit = jit(activity)

for t in range(NUM_STEPS):
    logp, (a_t, _) = policy_random_action_jit(next(rng), None)

    logp_cache.append(logp)

    s_tp1, r_t, terminated, truncated, info = env.step(a_t)

    states_cache.append(s_tp1)

    s_t = s_tp1

    if terminated or truncated:
        print(f"Terminated = {terminated} after {t+1} steps")
        break

loop_time = time.time() - start_time
time_per_step = loop_time / NUM_STEPS
est_time_full_run = time_per_step * 10**6 / 3600  # time for 1 million steps in hours
print(f"{loop_time*1000:.2f}", "ms")
# %% VIZUALISE TRAJECTORY
# fig_traj = render_spin_trajectory(states_cache)
# %% THE REFERENCE DYNAMICS
NUM_STEPS = 1000
start_time = time.time()

(
    _,
    _,
) = env.reset()
states_cache = [env.state]
env_rewards_cache = []
rewards_cache = []
activity_cache = []
logp_cache = []
s_t = env.state

activity_jit = jit(activity)

for _ in range(NUM_STEPS):
    logp, (a_t, _) = env.policy_ref_jit(next(rng), s_t)

    logp_cache.append(logp)  # entropy

    s_tp1, r_t, terminated, truncated, info = env.step(a_t)

    env_rewards_cache.append(r_t)
    rewards_cache.append(r_t - logp)  # subtract entropy term, logp

    states_cache.append(s_tp1)
    rewards_cache.append(r_t)

    activity_cache.append(activity_jit(s_t, a_t, s_tp1))

    s_t = s_tp1

loop_time = time.time() - start_time
time_per_step = loop_time / NUM_STEPS
est_time_full_run = time_per_step * 10**6 / 3600  # time for 1 million steps in hours
print(f"{loop_time*1000:.2f}", "ms")
# %% PLOT A TRAJ GENERATED BY THE REFERENCE DYNAMICS
# fig_traj = render_spin_trajectory(states_cache)
# %%
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

# %% USE THE EVALUATION UTILITY
biases = np.arange(-1, 1.1, 0.1)

results_ref_policy = evaluate_policy_biases(
    env.policy_ref_jit, env, 1000, biases, verbose=True
)

results_no_action_policy = evaluate_policy_biases(
    policy_no_action_jit, env, 1000, biases, verbose=False
)
# %%
scgfs_ref = [np.mean(result["r_t"]) for result in results_ref_policy]
scgfs_no_action = [np.mean(result["r_t"]) for result in results_no_action_policy]

plt.figure()
plt.plot(biases, scgfs_ref, "b^--", label="Reference Dynamics")
plt.plot(biases, scgfs_no_action, "o--", label="No Action Dynamics")
plt.grid()
plt.xlabel("Bias")
plt.ylabel("SCGF")
# %% PLOT ANIMATION OF THE REFERENCE DYNAMICS (SLOW)
# # %%
# anim = make_animation_horizontal(
#     np.array(states_cache), 150, 1000, "../data/animations/", "em_ref_dynamics", "10"
# )
# %%
