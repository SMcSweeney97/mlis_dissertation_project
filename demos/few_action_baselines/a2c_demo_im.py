# %% IMPORTS
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', "..", "stat-mech-gym", "src")) #gym

import jax.numpy as jnp
import jax
import haiku as hk
import matplotlib.pyplot as plt
import numpy as np
import coax
from optax import sgd
import time
import optax

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

# %% INIT RNG AND CONFIG
seed_env = 456
seed_pi = 786
seed_vf = 123
# %% INIT ENV
rng = hk.PRNGSequence(seed_env)
config = {"L": 20, "bias": 0, "d": 2, "D": 2, "temp":0.2, "render_mode": None, "obs_fn": activity, "mean": 0}
env = IsingModel(config)
# %%
def func_v(S, is_training): 
    value = hk.Sequential((hk.Linear(1, w_init=jnp.zeros), jnp.ravel))
    return value(S)
#if gaussian instead of sequential should be a dict of mew and sigma

def func_pi(S, is_training, config):
    logvar = hk.Sequential((
        hk.Flatten(),
        hk.Linear(8), jax.nn.relu,
        hk.Linear(8), jax.nn.relu,
        hk.Linear(8), jax.nn.relu,
        hk.Linear(config["L"]**config["D"], w_init=jnp.zeros), jax.nn.softmax))
    mu = hk.Sequential((
        hk.Flatten(),
        hk.Linear(8), jax.nn.relu,
        hk.Linear(8), jax.nn.relu,
        hk.Linear(8), jax.nn.relu,
        hk.Linear(config["L"]**config["D"], w_init=jnp.zeros), jax.nn.softmax))
    return {'logvar': logvar(S),'mu': mu(S)}

# %% function approximators
pi = coax.Policy(lambda S, is_training: func_pi(S, is_training, config), env, proba_dist=coax.proba_dists.NormalDist(env.observation_space, clip_box=(-256.0, 256.0), clip_reals=(-30.0, 30.0), clip_logvar=(-20.0, 20.0)))

vf = coax.V(func_v, env)
# %% DEFINE THE HYPER-PARAMS
BOOTSTRAP_N = 1
DISCOUNT = 1
LR_PI = 1e-3
LR_VF = 1e-4
LR_R = 1e-4
tracer = coax.reward_tracing.NStep(n=1, gamma=DISCOUNT) #stores (R_t^n, I_t^n, S_t+n, R_t+n)
# %% RUN A TEST LOOP WITH ASSERTIONS ETC.
NUM_STEPS=1000
s_0, _ = env.reset()
tracer.reset()
s_0 = env.state
rewards = []
reward_diff = []
states = [s_0]
s_t = s_0
rb_t = 0.0 #Estimate of the average-reward per time-step
avg_rew_ests = []

#pi = coax.Policy(lambda S, is_training: func_pi(S, is_training, config), env, proba_dist=coax.proba_dists.CategoricalDist(env.action_space, gumbel_softmax_tau=0.2))
pi = coax.Policy(lambda S, is_training: func_pi(S, is_training, config), env, proba_dist=coax.proba_dists.NormalDist(env.observation_space, clip_box=(-256.0, 256.0), clip_reals=(-30.0, 30.0), clip_logvar=(-20.0, 20.0)))

vf = coax.V(func_v, env)

simple_td = coax.td_learning.SimpleTD(vf, vf,  optimizer=optax.sgd(LR_VF), loss_function=coax.value_losses.mse) #TD UPDATER
vanilla_pg = coax.policy_objectives.VanillaPG(pi, optimizer=optax.sgd(LR_PI)) # POLICY-GRAD UPDATER

start_time = time.time()

for t in range(NUM_STEPS):

    a_t, logp_t = pi(s_t, return_logp=True)
    
    a_t= jnp.floor(a_t+0.5)
    a_t=a_t.astype(int)
    # a_t = jax.nn.sigmoid(a_t)

    # a_t = jnp.where(a_t.astype(bool))
    print("HERE")
    print(a_t)
    print(type(a_t))
    s_tp1, r_t, terminated, truncated, info = env.step(a_t)

    r_t = r_t - logp_t #entropy term
    rd_t = r_t - rb_t #differential reward
    
    tracer.add(s_t, a_t, rd_t, terminated, logp=logp_t, w=6)

    while tracer:
        transition_batch = tracer.pop()

        # tracer cache i, provides t content, and is ready at t+1
        assert np.allclose(transition_batch.S[0,:,:], states[t-1])
        assert np.allclose(transition_batch.S_next[0,:,:], states[t])
        assert np.allclose(transition_batch.Rn, reward_diff[t-1])


        td_error_ref_batch = transition_batch.Rn + vf(transition_batch.S_next) - vf(transition_batch.S) #actor-critic explains and should estimate rewards from now onwards
        
        metrics_v, td_error = simple_td.update(transition_batch, return_td_error=True)
        
        assert np.allclose(td_error, td_error_ref_batch)

        metrics_pi = vanilla_pg.update(transition_batch, td_error) #advantage factor is the td_error

        rb_t = rb_t + LR_R*td_error.item() #update avg_rew_est

    states.append(s_tp1)
    rewards.append(r_t)
    reward_diff.append(rd_t)
    avg_rew_ests.append(rb_t)

    avg_rew_est = np.mean(rewards)

    s_t = s_tp1

loop_time = time.time() - start_time
time_per_step = loop_time/NUM_STEPS
print(f"{loop_time:.2f}s for {NUM_STEPS} steps")

# %% RUN A FULL LOOP.
NUM_STEPS=1000
OBS_FREQ = 1 #How often to store rewards etc for visualisation
s_0, _ = env.reset()
tracer.reset()
s_0 = env.state
s_t = s_0
rb_t = 0.0 #Estimate of the average-reward per time-step


pi = coax.Policy(lambda S, is_training: func_pi(S, is_training, config), env, proba_dist=coax.proba_dists.NormalDist(env.observation_space, clip_box=(-256.0, 256.0), clip_reals=(-30.0, 30.0), clip_logvar=(-20.0, 20.0)))



#pi = coax.Policy(lambda S, is_training: func_pi(S, is_training, config), env, proba_dist=coax.proba_dists.CategoricalDist(env.action_space, gumbel_softmax_tau=0.2))
vf = coax.V(func_v, env)

# simple_td = coax.td_learning.SimpleTD(vf, None,  optimizer=sgd(LR_VF), loss_function=coax.value_losses.mse) #TD UPDATER
# vanilla_pg = coax.policy_objectives.VanillaPG(pi, optimizer=sgd(LR_PI)) # POLICY-GRAD UPDATER
simple_td = coax.td_learning.SimpleTD(vf, None,  optimizer=optax.sgd(learning_rate=LR_VF), loss_function=coax.value_losses.mse) #TD UPDATER
vanilla_pg = coax.policy_objectives.VanillaPG(pi, optimizer=optax.sgd(learning_rate=LR_PI)) # POLICY-GRAD UPDATER

#store r_t, rb_t and states for comparisons
rewards = []
avg_reward_ests = [rb_t]
states = [s_0]

start_time = time.time()

for t in range(NUM_STEPS):

    a_t, logp_t = pi(s_t, return_logp=True)
    a_t= jnp.floor(a_t+0.5)
    a_t=a_t.astype(int)
    s_tp1, r_t, terminated, truncated, info = env.step(a_t)

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

# %% PLOT ESTIMATES OF AVERAGE REWARD
import pickle
data_path = os.path.join(os.path.dirname(__file__), '..', "data")
file_path = os.path.join(data_path, f"learing_curve_L{8}_s{config['s']}.pkl")

with open(file_path, 'rb') as file_obj:
    ref_lc = pickle.load(file_obj)

acten_results = env.get_baseline_rewards("acten")
ed_results = env.get_baseline_rewards("exact")

fig_rewards = plot_learning_curve(rewards, method="mean")
plt.plot(np.arange(0, NUM_STEPS + OBS_FREQ, OBS_FREQ), avg_reward_ests, "b-", linewidth=4, label="rb_t")
plt.plot(np.arange(0, 100000, 100), ref_lc, "y-",linewidth=4, label="pure-func acten lc")
plt.xlim(0, NUM_STEPS)
plt.hlines(acten_results, 0, NUM_STEPS, color="k", linewidth=4, label="acten")
plt.hlines(ed_results, 0, NUM_STEPS, color="g", linestyles="dashed", linewidth=4, label="exact")
plt.legend()
plt.ylim(bottom=-0.15, top = -0.07)
plt.savefig('coax_L8_s-2.png')
# plt.xscale('log')
# %%
# %% PLOT TRAJ
fig_traj = render_spin_trajectory(states)
# %%
