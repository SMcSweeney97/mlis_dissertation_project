from gymnasium import Wrapper


class SoftenConstraint(Wrapper):
    """Softens the Constraint.

    In Default (Hard-constraint) choosing an unavailable action terminates the episode.
    When Softened, choosing an unavailable action incurs a penalty instead.

    Note, this only works if info contains "action_available"

    """

    def __init__(self, env, constraint_penalty):
        super().__init__(env)
        self.constraint_penalty = constraint_penalty

    def step(self, action):
        res = list(self.env.step(action))
        terminated = res[2]  # this will be "done" in the old gym style
        info = res[-1]
        action_avail = info["action_available"].item()
        reward = res[1]

        if action_avail is False:
            terminated = False  # reset truncated=True from hard constraint
            reward = reward + self.constraint_penalty

        res[1] = reward
        res[2] = terminated

        return tuple(res)
