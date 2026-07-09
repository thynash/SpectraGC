"""
Dataset loading utilities.
"""

import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T


def load_cora(device=None):

    dataset = Planetoid(
        root="data/Cora",
        name="Cora",
        transform=T.NormalizeFeatures()
    )

    data = dataset[0]

    if device is not None:

        data = data.to(device)

    return data
