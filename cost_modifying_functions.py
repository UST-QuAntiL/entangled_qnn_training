import torch


power = 2

def identity(x):
    return x


def power_func(x):
    return torch.float_power(torch.abs(x), power)


def root_func(x):
    return x-x if x < 1e-2 else torch.float_power(torch.abs(x), 1/power)


def funky_func(x):
    return 1-(torch.exp(-x*10))
