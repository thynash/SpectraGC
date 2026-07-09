"""
Graph utility functions.
"""

import torch
from torch_geometric.utils import to_dense_adj


def edge_index_to_dense(edge_index):

    return to_dense_adj(edge_index).squeeze(0)


def number_of_nodes(data):

    return data.num_nodes


def number_of_edges(data):

    return data.edge_index.size(1)
