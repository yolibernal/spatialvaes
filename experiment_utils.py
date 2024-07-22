import numpy as np


def exponential_scheduler(step, total_steps, initial, final):
    """Exponential scheduler"""

    if step >= total_steps:
        return final
    if step <= 0:
        return initial
    if total_steps <= 1:
        return final

    t = step / (total_steps - 1)
    log_value = (1.0 - t) * np.log(initial) + t * np.log(final)
    return np.exp(log_value)


def linear_scheduler(step, total_steps, initial, final):
    """Linear scheduler"""

    if step >= total_steps:
        return final
    if step <= 0:
        return initial
    if total_steps <= 1:
        return final

    t = step / (total_steps - 1)
    return (1.0 - t) * initial + t * final


def cyclical_annealing_scheduler(step, total_steps, initial, final, M, R=0.5):
    if step >= total_steps:
        return final
    if step <= 0:
        return initial
    cycle_length = total_steps // M
    tau = (step % (cycle_length)) / (cycle_length)

    if tau <= R:
        return linear_scheduler(int(tau * cycle_length), cycle_length * R, initial, final)
    else:
        return final
