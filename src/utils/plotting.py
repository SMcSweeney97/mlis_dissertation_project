import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as c
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator


def get_physical_state_traj(states_cache, physical_state_name="spin_state"):
    """Extract the physical-state trajectory from a cache of compostite states (including aux-states)"""
    traj = [states_cache[i][physical_state_name] for i in range(len(states_cache))]
    return np.array(traj)


def plot_learning_curve(rewards_cache, method="mean", hlines=None, ylabel=None):
    """Plot a learning curve based on the rewards"""

    LINEWIDTH = 4
    FONTSIZE = 20
    MARKERSIZE = 12
    FIGX = 10
    FIGY = 8

    methods = ["mean"]

    assert (
        method in methods
    ), f"chosen method {method} not in currently implemented methods {methods}"

    fig, ax = plt.subplots(1, 1, figsize=(FIGX, FIGY))

    ax.tick_params(axis="both", which="major", labelsize=FONTSIZE)
    ax.set_xlabel("t", fontsize=FONTSIZE)

    if not (hlines is None):
        for val in hlines:
            ax.hlines(val, 0, len(rewards_cache), linewidth=LINEWIDTH)

    if method == "mean":
        ax.plot(
            np.cumsum(rewards_cache) / np.arange(1, len(rewards_cache) + 1),
            "r-",
            linewidth=LINEWIDTH,
        )

        if ylabel is None:
            ax.set_ylabel(r"$\langle r \rangle_{1:t}$", fontsize=FONTSIZE)

        else:
            ax.set_ylabel(ylabel, fontsize=FONTSIZE)

        return fig


def render_spin_trajectory(states, save_path=None, plot_tracer=False):
    """Create a trajectory image from a set of spin states"""

    LINEDWITH = 4
    FONTSIZE = 20
    MARKERSIZE = 12

    states = np.array(states)
    traj_length, lattice_size = states.shape

    fig = plt.figure(figsize=(20, 8))

    trajs_to_keep = []
    tracers_to_keep = []

    gs = gridspec.GridSpec(1, 1, wspace=0.25)
    ax1 = fig.add_subplot(gs[0, 0])

    y = np.arange(0, lattice_size)
    x = np.arange(1, traj_length + 1)
    xmesh, ymesh = np.meshgrid(x, y)

    traj = states.T

    ax1.pcolormesh(xmesh, ymesh, traj, cmap="gray_r")

    if plot_tracer:
        cMaps = [c.ListedColormap(["b"]), c.ListedColormap(["y"])]

        initial_slice = traj[:, 0]
        initial_positions = np.where(initial_slice == 1)[0]

        tracers_to_plot = [np.amin(initial_positions)]

        for idx in range(len(tracers_to_plot)):
            traj_array = get_tracer_array(tracers_to_plot[idx], traj)

            tracers_to_keep.append(traj_array)

            # traj_array = get_tracer(6,traj)
            traj_array = np.ma.masked_array(traj_array, traj_array != 1)

            ax1.pcolormesh(xmesh, ymesh, traj_array, cmap=cMaps[idx])

    ax1.tick_params(axis="both", which="major", labelsize=FONTSIZE)
    ax1.set_ylabel(r"$x$", fontsize=FONTSIZE)
    ax1.set_xlabel(r"$t$", fontsize=FONTSIZE)

    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))

    if save_path is not None:
        fig.savefig(
            save_path,
            bbox_inches="tight",
            transparent=True,
            facecolor="w",
            edgecolor="w",
            orientation="landscape",
        )

    return fig


def get_tracer_array(starting_position, traj):
    def get_action(current_state, next_state):
        current_state = np.array(current_state)
        next_state = np.array(next_state)

        if np.allclose(current_state, next_state):
            action = len(current_state)
        else:
            flip_idxs = np.where(current_state != next_state)[0]
            assert (
                len(flip_idxs) == 2
            ), f"More than one flip found: flip_idxs = {flip_idxs}"
            if np.allclose(flip_idxs, [0, len(current_state) - 1]):  # boundary case
                action = len(current_state) - 1
            else:
                action = np.amin(flip_idxs)

        return action

    def get_actions_traj(traj):
        actions = []

        for current_time in range(traj.shape[1] - 1):
            current_state = traj[:, current_time]
            next_state = traj[:, current_time + 1]
            action = get_action(current_state, next_state)
            actions.append(action)

        return actions

    def get_particle_evolution(initial_position, actions, lattice_size):
        positions = [initial_position]
        current_position = positions[-1]
        for action in actions:
            if action == current_position:  # particle moves to right
                next_position = (current_position + 1) % lattice_size
            elif action == (current_position - 1) % lattice_size:  # left move
                next_position = (current_position - 1) % lattice_size
            else:
                next_position = current_position

            positions.append(next_position)
            current_position = next_position

        return positions

    traj_array = np.zeros((traj.shape[0], traj.shape[1]))
    actions = get_actions_traj(traj)

    assert (
        traj[starting_position, 0] == 1
    ), f"Starting position doesn't have a particle in."
    positions = get_particle_evolution(starting_position, actions, traj.shape[0])

    for counter, position in enumerate(positions):
        traj_array[position, counter] = 1

    return traj_array
