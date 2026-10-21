"""A relation-constrained MLP built from extract_structure.py's symbol matrices.

Each original weight is a fixed sign (+1/-1/0) times one trainable cluster
center.  This implements the symbol-matrix parameter sharing in SI Eq. (8.1).
It does not perform the paper's subsequent manual block merging or pruning.
"""
import json
import math
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F

from Module.PINN import Net


class ClusteredLinear(nn.Module):
    def __init__(self, weight_info, bias_info, init):
        super().__init__()
        self.out_features, self.in_features = weight_info['shape']
        self.weight_centers = nn.Parameter(self._centers(weight_info, init, self.in_features))
        weight_index, weight_sign = self._relation(weight_info)
        self.register_buffer('weight_index', weight_index)
        self.register_buffer('weight_sign', weight_sign)

        self.bias_centers = nn.Parameter(self._centers(bias_info, init, self.in_features))
        bias_index, bias_sign = self._relation(bias_info)
        self.register_buffer('bias_index', bias_index)
        self.register_buffer('bias_sign', bias_sign)

    @staticmethod
    def _centers(info, init, fan_in):
        values = torch.tensor(list(info['centers'].values()), dtype=torch.float32)
        if init == 'random':
            values.uniform_(0, 1 / math.sqrt(fan_in))
        elif init != 'centers':
            raise ValueError(f'Unknown initialization: {init}')
        return values

    @staticmethod
    def _relation(info):
        letters = list(info['centers'])
        index_of = {letter: i for i, letter in enumerate(letters)}
        def flatten(items):
            for item in items:
                if isinstance(item, list):
                    yield from flatten(item)
                else:
                    yield item
        values = list(flatten(info['symbols']))
        indices = [0 if value == '0' else index_of[value.lstrip('-')] for value in values]
        signs = [0.0 if value == '0' else (-1.0 if value.startswith('-') else 1.0) for value in values]
        return (torch.tensor(indices, dtype=torch.long).reshape(info['shape']),
                torch.tensor(signs, dtype=torch.float32).reshape(info['shape']))

    def forward(self, x):
        weight = self.weight_centers[self.weight_index] * self.weight_sign
        bias = self.bias_centers[self.bias_index] * self.bias_sign
        return F.linear(x, weight, bias)


class ClusteredBurgers(Net):
    def __init__(self, structure_path, init='random'):
        with open(structure_path, encoding='utf-8') as stream:
            structure = json.load(stream)
        weights = []
        layer_number = 0
        while f'layers.layer_{layer_number}.weight' in structure:
            weights.append(structure[f'layers.layer_{layer_number}.weight'])
            layer_number += 1
        layers = [weights[0]['shape'][1]] + [info['shape'][0] for info in weights]
        super().__init__(layers)
        ordered = OrderedDict()
        for i, weight_info in enumerate(weights):
            bias_info = structure[f'layers.layer_{i}.bias']
            ordered[f'layer_{i}'] = ClusteredLinear(weight_info, bias_info, init)
            if i < len(weights) - 1:
                ordered[f'activation_{i}'] = nn.Tanh()
        self.layers = nn.Sequential(ordered)

    @property
    def dense_parameter_count(self):
        return sum(layer.weight_index.numel() + layer.bias_index.numel()
                   for layer in self.layers if isinstance(layer, ClusteredLinear))
