# %% IMPORTS
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', "..", "stat-mech-gym", "src")) #gym

import jax.numpy as jnp
import haiku as hk
import matplotlib.pyplot as plt
import numpy as np
import coax
from optax import sgd
import time
import optax

import os, sys

from fix_pathing import root_dir

from src.envs.ising_model_1d.ising_model import IsingModel, activity, magnetisation,get_kern_filter

from src.utils.plotting import plot_learning_curve
import haiku as hk
import matplotlib.pyplot as plt
import numpy as np
import time
from jax import jit
import jax.numpy as jnp
import jax.random as jr

from src.utils.evaluation import (
    evaluate_policy_biases,
)

import seaborn as sns

import jax
# %% INIT RNG AND CONFIG
seed_env = 567
seed_pi = 234
seed_vf = 123
# %% INIT ENV
# config = {"L": 4, "bias": 0, "d": 2, "D": 2, "temp":0.2, "render_mode": None, "obs_fn": activity}
config = {"L": 20, "bias": 1, "d": 2, "D":2, "temp":1/2.269, "render_mode": None, "obs_fn": activity, "mean": 0, "kern":get_kern_filter(2)}

env = IsingModel(config, seed=seed_env)
# %%
def func_v(S, is_training):
    value = hk.nets.MLP(output_sizes=[4,1], w_init=jnp.zeros, b_init=jnp.zeros)
    return jnp.ravel(value(S))

def func_pi(S, is_training, config):
    logits = hk.nets.MLP(output_sizes=[4,config["L"]**config["D"]+1], w_init=jnp.zeros, b_init=jnp.zeros)
    return {'logits': logits(S)}
# %% DEFINE THE HYPER-PARAMS
BOOTSTRAP_N = 1
DISCOUNT = 1
LR_PI = 1e-2
LR_VF = 1e-2
LR_R = 1e-3
tracer = coax.reward_tracing.NStep(n=1, gamma=DISCOUNT) #stores (R_t^n, I_t^n, S_t+n, R_t+n)
# %% RUN A TEST LOOP WITH ASSERTIONS ETC.
NUM_STEPS = 50000
OBS_FREQ = 1
s_0, _ = env.reset()
tracer.reset()
s_0 = env.state
rewards = []
reward_diff = []
states = [s_0]
s_t = s_0
rb_t = 0.0 #Estimate of the average-reward per time-step
avg_rew_ests = []
actions = []
env_rewards = []
entropies = []
mags = []
activities = []

pi = coax.Policy(lambda S, is_training: func_pi(S, is_training, config), env, proba_dist=coax.proba_dists.CategoricalDist(env.action_space, gumbel_softmax_tau=0.2), random_seed=seed_pi)
vf = coax.V(func_v, env, random_seed=seed_vf)

simple_td = coax.td_learning.SimpleTD(vf, vf,  optimizer=optax.adam(learning_rate=LR_VF), loss_function=coax.value_losses.mse) #TD UPDATER
vanilla_pg = coax.policy_objectives.VanillaPG(pi, optimizer=optax.adam(learning_rate=LR_PI)) # POLICY-GRAD UPDATER

#store r_t, rb_t and states for comparisons
rewards = []
avg_reward_ests = [rb_t]
states = [s_0]

start_time = time.time()

for t in range(NUM_STEPS):

    a_t, logp_t = pi(s_t, return_logp=True)
    s_tp1, r_t, terminated, truncated, info = env.step(a_t)

    actions.append(a_t)
    env_rewards.append(r_t)
    entropies.append(logp_t)
    mags.append(magnetisation(s_t, a_t, s_tp1))
    activities.append(activity(s_t, a_t, s_tp1))

    r_t = r_t - logp_t #entropy term
    rd_t = r_t - rb_t #differential reward

    tracer.add(s_t, a_t, rd_t, terminated, logp=logp_t)

    while tracer:

        transition_batch = tracer.pop()

        metrics_v, td_error = simple_td.update(transition_batch, return_td_error=True)
        metrics_pi = vanilla_pg.update(transition_batch, td_error) #advantage factor is the td_error for A2C

        rb_t = rb_t + LR_R*td_error.item() #update rb_t

    s_t = s_tp1


    if t%OBS_FREQ==0:

        states.append(s_tp1)
        rewards.append(r_t)
        avg_reward_ests.append(rb_t)

loop_time = time.time() - start_time
time_per_step = loop_time/NUM_STEPS
print(f"{loop_time:.2f}s for {NUM_STEPS} steps")

# %%

rewards = np.ravel(rewards)
avg_rew_ests = np.ravel(avg_reward_ests)

# %% Plot ESTIMATES OF AVERAGE REWARD
fig, axs = plt.subplots(1,1, figsize=(10,5))
axs.plot(np.arange(0, NUM_STEPS + OBS_FREQ, OBS_FREQ), avg_reward_ests, "bs--", linewidth=4, label="rb_t")
axs.set_xlim(0, NUM_STEPS)
axs.legend()
# %% Plot Magnetisation LC
# for L=4, T = 0.5 (beta = 0.2), |M| approx 0.35 for PBC
fig = plot_learning_curve(np.ravel(mags), ylabel="Mag")

# %% Histogram of Actions
sns.displot(actions, discrete=True)
# %% Frac of no-flip-actions
no_flip_actions = np.cumsum(np.array(actions) == config["L"]**config["D"])
frac_no_flip = no_flip_actions/np.arange(1,len(no_flip_actions)+1)
plt.plot(frac_no_flip)
num_no_flip_actions = np.sum(np.array(actions) == config["L"]**config["D"])
plt.title(f"Long-time no change frac = {num_no_flip_actions}/{len(actions)} = {num_no_flip_actions/len(actions):.2f}")
plt.ylim(0,1)
# %% Activity
fig_activity = plot_learning_curve(activities, hlines=[np.mean(activities)])
plt.ylim(0,1)
plt.title(f"Long-time no average activity = {np.mean(activities):.2f}")
# %% Actions
plt.plot(actions)
# %% KL-div
kl_div = -np.ravel(rewards) - config["bias"]*np.ravel(activities)
fig_kl= plot_learning_curve(kl_div, hlines=[np.mean(kl_div)])
plt.title(f"MC Approx to KL-div in Steady State = {np.mean(kl_div):.2f}")
# %% COMPARE WITH TD-EST
kl_div_est = -np.ravel(avg_reward_ests[1::]) - np.cumsum(config["bias"]*np.ravel(activities))/np.arange(1,len(np.ravel(activities))+1)
plt.plot(kl_div_est)
plt.title(f"MC Approx to Estimate of KL-div in Steady State = {np.mean(kl_div_est[6000::]):.2f}")
# %% EVALUATE THE POLICY
# %% RUN A TEST LOOP WITH ASSERTIONS ETC.
NUM_STEPS = 10000
OBS_FREQ = 1
s_0, _ = env.reset()
tracer.reset()
s_0 = env.state
rewards = []
reward_diff = []
states = [s_0]
s_t = s_0
eval_results = {}
eval_results["avg_rew_ests"] = []
actions = []
env_rewards = []
entropies = []
mags = []
activities = []

#store r_t, rb_t and states for comparisons
rewards = []
avg_reward_ests = [rb_t]
states = [s_0]

start_time = time.time()

for t in range(NUM_STEPS):

    a_t, logp_t = pi(s_t, return_logp=True)
    s_tp1, r_t, terminated, truncated, info = env.step(a_t)

    actions.append(a_t)
    env_rewards.append(r_t)
    entropies.append(logp_t)
    mags.append(magnetisation(s_t, a_t, s_tp1))
    activities.append(activity(s_t, a_t, s_tp1))

    r_t = r_t - logp_t #entropy term

    s_t = s_tp1

    if t%OBS_FREQ==0:

        states.append(s_tp1)
        rewards.append(r_t)

loop_time = time.time() - start_time
time_per_step = loop_time/NUM_STEPS
print(f"{loop_time:.2f}s for {NUM_STEPS} steps")
# %% Plot ESTIMATES OF AVERAGE REWARD
fig = plot_learning_curve(rewards)
# %%
sns.displot(actions, discrete=True)
# %%
