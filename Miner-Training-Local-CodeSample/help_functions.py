from collections import OrderedDict, deque
import random
from typing import Deque, Dict, List, Tuple

import gym
import numpy as np
import torch
import torch.nn as nn

def get_n_step_info_from_demo(
    demo: List, n_step: int, gamma: float
) -> Tuple[List, List]:
    """Return 1 step and n step demos."""
    assert n_step > 1

    demos_1_step = list()
    demos_n_step = list()
    n_step_buffer: Deque = deque(maxlen=n_step)

    for transition in demo:
        n_step_buffer.append(transition)

        if len(n_step_buffer) == n_step:
            # add a single step transition
            demos_1_step.append(n_step_buffer[0])

            # add a multi step transition
            curr_state, action = n_step_buffer[0][:2]
            reward, next_state, done = get_n_step_info(n_step_buffer, gamma)
            transition = (curr_state, action, reward, next_state, done)
            demos_n_step.append(transition)

    return demos_1_step, demos_n_step

def get_n_step_info(
    n_step_buffer: Deque, gamma: float
) -> Tuple[np.int64, np.ndarray, bool]:
    """Return n step reward, next state, and done."""
    # info of the last transition
    reward, next_state, done = n_step_buffer[-1][-3:]

    for transition in reversed(list(n_step_buffer)[:-1]):
        r, n_s, d = transition[-3:]

        reward = r + gamma * reward * (1 - d)
        next_state, done = (n_s, d) if d else (next_state, done)

    return reward, next_state, done

def numpy2floattensor(arrays: Tuple[np.ndarray]) -> Tuple[np.ndarray]:
    """Convert numpy arrays to torch float tensor."""
    tensors = []
    for array in arrays:
        tensor = torch.FloatTensor(array).to(device, non_blocking=True)
        tensors.append(tensor)
    return tuple(tensors)

