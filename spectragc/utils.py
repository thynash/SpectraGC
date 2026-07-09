"""
Utility functions.
"""

import random
import numpy as np
import torch


def set_seed(seed: int = 42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)


def get_device():

    if torch.cuda.is_available():

        return torch.device("cuda")

    return torch.device("cpu")


def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
