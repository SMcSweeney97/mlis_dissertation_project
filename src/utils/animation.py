import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

def make_animation(images, fps, save_path=None, file_name=None):
    """Puts together a set of images into an animation via imshow

        file_name: name of file with or without .mp4 file extension (only works for mp4)

        Example usage:
        anim = make_animation(np.reshape(x, (1001, 8, 1)), 100, "./", "test")


    """

    if len(images.shape) == 2: #add a fake colour channel
        images = np.reshape(images, (images.shape[0], images.shape[1], 1))

    num_frames = len(images)
    fig = plt.figure(figsize=(8,8))

    a = images[0]
    im = plt.imshow(a)

    def animate_func(i):
        # if i % fps == 0:
        #     print( '.', end ='' )

        im.set_array(images[i])
        return [im]

    anim = animation.FuncAnimation(
                                fig,
                                animate_func,
                                frames = num_frames,
                                interval = 1000 / fps, # in ms
                                )

    if save_path is not None:
        if file_name[-4::] != ".mp4":
            save_path = save_path + file_name + ".mp4"
        anim.save(save_path, fps=fps, extra_args=['-vcodec', 'libx264'])

    return anim

def make_animation_vertical(images, fps, steps_to_show, save_path=None, file_name=None, aspect_ratio=None):
    """Puts together a set of images into an animation via imshow

        This animation places time "vertically" with a sliding window

        file_name: name of file with or without .mp4 file extension (only works for mp4)

        Example usage:
        anim = make_animation(np.reshape(x, (1001, 8, 1)), 100, "./", "test")

    Adapted from https://github.com/IlievskiV/Amusive-Blogging-N-Coding/blob/master/Cellular%20Automata/cellular_automata.ipynb


    """

    size = images.shape[1]
    iterations_per_frame = 1

    # if len(images.shape) == 2: #add a fake colour channel
    #     images = np.reshape(images, (images.shape[0], images.shape[1], 1))

    num_frames = len(images)

    fig = plt.figure(figsize=(10, 10))

    ax = plt.axes()
    ax.set_axis_off()

    def animate_func(i):
        ax.clear()  # clear the plot
        ax.set_axis_off()  # disable axis

        Y = np.zeros((steps_to_show, size), dtype=np.int8)  # initialize with all zeros
        upper_boundary = (i + 1) * iterations_per_frame  # window upper boundary
        lower_boundary = 0 if upper_boundary <= steps_to_show else upper_boundary - steps_to_show  # window lower bound.
        for t in range(lower_boundary, upper_boundary):  # assign the values
            Y[t - lower_boundary, :] = images[t, :]

        img = ax.imshow(Y, interpolation='none',cmap='RdPu', aspect=aspect_ratio)
        return [img]

    anim = animation.FuncAnimation(
                                fig,
                                animate_func,
                                frames = num_frames,
                                interval = 1000 / fps, # in ms
                                )

    if save_path is not None:
        if file_name[-4::] != ".mp4":
            save_path = save_path + file_name + ".mp4"
        anim.save(save_path, fps=fps, extra_args=['-vcodec', 'libx264'])

    return anim

def make_animation_horizontal(images, fps, steps_to_show, save_path=None, file_name=None, aspect_ratio=None):
    """Puts together a set of images into an animation via imshow

        This animation places time "vertically" with a sliding window

        file_name: name of file with or without .mp4 file extension (only works for mp4)

        Example usage:
        anim = make_animation(np.reshape(x, (1001, 8, 1)), 100, "./", "test")

    Adapted from https://github.com/IlievskiV/Amusive-Blogging-N-Coding/blob/master/Cellular%20Automata/cellular_automata.ipynb


    """

    size = images.shape[1]
    iterations_per_frame = 1

    # if len(images.shape) == 2: #add a fake colour channel
    #     images = np.reshape(images, (images.shape[0], images.shape[1], 1))

    num_frames = len(images)

    fig = plt.figure(figsize=(10, 10))

    ax = plt.axes()
    ax.set_axis_off()

    def animate_func(i):
        ax.clear()  # clear the plot
        ax.set_axis_off()  # disable axis

        Y = np.zeros((steps_to_show, size), dtype=np.int8)  # initialize with all zeros
        upper_boundary = (i + 1) * iterations_per_frame  # window upper boundary
        lower_boundary = 0 if upper_boundary <= steps_to_show else upper_boundary - steps_to_show  # window lower bound.
        for t in range(lower_boundary, upper_boundary):  # assign the values
            Y[t - lower_boundary, :] = images[t, :]

        img = ax.imshow(np.transpose(Y), interpolation='none',cmap='RdPu', aspect=aspect_ratio)
        return [img]

    anim = animation.FuncAnimation(
                                fig,
                                animate_func,
                                frames = num_frames,
                                interval = 1000 / fps, # in ms
                                )

    if save_path is not None:
        if file_name[-4::] != ".mp4":
            save_path = save_path + file_name + ".mp4"
        anim.save(save_path, fps=fps, extra_args=['-vcodec', 'libx264'])

    return anim