import numpy as np
import jax.random as jr


def make_bias_set(config, biases):
    """Construct a set of environment configs with the chosen biases
    The current bias in config is overwritten, while other variables are kept.
    """

    configs = []
    for bias in biases:
        tmp = dict(config)
        tmp["bias"] = bias
        configs.append(tmp)

    return configs


def evaluate_policy_biases(
    policy,
    env,
    n_steps,
    biases,
    seed=101,
    save_states=False,
    save_actions=False,
    verbose=False,
):
    config = env.config
    configs = make_bias_set(config, biases)

    results = []

    for counter, config in enumerate(configs):
        if verbose:
            print(f"Evaluating Policy in Environment {counter+1} of {len(configs)}")
            print(f".... Bias set to {config['bias']:.4f}")

        env.__init__(config, render_mode=None, seed=seed, key=None)

        res = evaluate_policy(
            policy, env, n_steps, save_states=save_states, save_actions=save_actions
        )
        results.append(res)

    return results


def evaluate_policies_biases(
    policies,
    env,
    n_steps,
    biases,
    seed=101,
    save_states=False,
    save_actions=False,
    verbose=False,
):
    config = env.config
    configs = make_bias_set(config, biases)

    results = []

    for counter, config in enumerate(configs):
        if verbose:
            print(f"Evaluating Policy in Environment {counter+1} of {len(configs)}")
            print(f".... Bias set to {config['bias']:.4f}")

        env.__init__(config, render_mode=None, seed=seed, key=None)

        res = evaluate_policy(
            policies[counter],
            env,
            n_steps,
            save_states=save_states,
            save_actions=save_actions,
        )
        results.append(res)

    return results


def evaluate_policy(
    policy, env, n_steps, seed=101, save_states=False, save_actions=False
):
    states = []
    actions = []

    key = jr.PRNGKey(seed)

    s_0, info = env.reset()
    rewards = []

    res = {}

    s_t = s_0

    if save_states is True:
        states.append(s_0)

    for step in range(n_steps):
        key, subkey = jr.split(key)

        logp, (a_t, _) = policy(subkey, s_t)

        ent = -logp

        if save_actions is True:
            actions.append(a_t)

        s_tp1, r_t, terminated, truncated, info = env.step(a_t)

        rewards.append(r_t + ent)

        if terminated or truncated:
            break

        if save_states is True:
            states.append(s_tp1)

        s_t = s_tp1

    res["r_t"] = np.ravel(rewards)
    res["terminated"] = terminated
    res["num_steps"] = step + 1

    if save_states is True:
        res["states"] = states
    if save_actions is True:
        res["actions"] = actions

    return res
