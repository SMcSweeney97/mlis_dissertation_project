import jax.numpy as jnp
import jax

jax.config.update("jax_enable_x64", True)

def activity(state, action, next_state):
    """Counts the number of spin-flips.

    Args:
        state (array): state before the action. shape = (L,).
        action (array): shape L array indicated the bonds on which to swap spins.
        new_state (array): state after the action. shape = (L,).

    Returns:
        activity (int): number of spin-flips.
    """
    spin_state = state["spin_state"]
    next_spin_state = next_state["spin_state"]
    return jnp.sum(jnp.not_equal(spin_state, next_spin_state))