import copy
import math
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn


def convert(input):
    """
    Converts input to a PyTorch tensor if it is a NumPy array.
    If the input is already a PyTorch tensor, it returns the input unchanged.
    """
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output

def get_shape_from_space(space):
    """
    Returns the shape of the action space.
    """
    if isinstance(space, gym.spaces.Discrete):
        return (1,)
    elif isinstance(space, gym.spaces.Box) \
            or isinstance(space, gym.spaces.MultiDiscrete) \
            or isinstance(space, gym.spaces.MultiBinary):
        return space.shape
    elif isinstance(space,gym.spaces.Tuple) and \
           isinstance(space[0], gym.spaces.MultiDiscrete) and \
               isinstance(space[1], gym.spaces.Discrete):
        return (space[0].shape[0] + 1,)
    else:
        raise NotImplementedError(f"Unsupported action space type: {type(space)}!")

def get_grad_norm(it):
    """
    Calculates the gradient norm of a list of tensors.
    It sums the squares of the gradients and returns the square root of the sum.
    Args:
        it (list): A list of tensors.
    Returns:
        float: The gradient norm.
    """
    sum_grad = 0
    for x in it:
        if x.grad is None:
            continue
        sum_grad += x.grad.norm() ** 2
    return math.sqrt(sum_grad)

def init(module: nn.Module, weight_init, bias_init, gain=1):
    """
    Initializes the weights and biases of a PyTorch module.
    Args:
        module (nn.Module): The PyTorch module to initialize.
        weight_init (callable): A function to initialize the weights.
        bias_init (callable): A function to initialize the biases.
        gain (float, optional): Gain factor for weight initialization. Default is 1.
    """
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

def get_clones(module, N):
    """
    Creates N clones of a given PyTorch module.
    Args:
        module (nn.Module): The PyTorch module to clone.
        N (int): The number of clones to create.
    Returns:
        nn.ModuleList: A list of cloned modules.
    """
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])