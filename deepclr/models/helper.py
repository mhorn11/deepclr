from typing import List, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F


_size_1_t = Union[int, Tuple[int]]


class Conv1d(nn.Module):
    """Conv1d with Relu."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_1_t, stride: _size_1_t = 1,
                 padding: _size_1_t = 0, dilation: _size_1_t = 1, groups: int = 1, bias: bool = True,
                 batch_norm: bool = False):
        super().__init__()

        # conv layer
        conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

        # init weights
        nn.init.xavier_uniform_(conv.weight)
        if conv.bias is not None:
            conv.bias.data.fill_(0.0)

        # sequential
        if batch_norm:
            self._sequential = nn.Sequential(conv, nn.BatchNorm1d(out_channels))
        else:
            self._sequential = nn.Sequential(conv)
        self._output_dim = out_channels

    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self._sequential(x))


class Linear(nn.Module):
    """Linear layer with Relu."""
    def __init__(self, in_features: int, out_features: int, bias: bool = True, batch_norm: bool = False):
        super().__init__()

        # conv layer
        lin = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)

        # init weights
        nn.init.xavier_uniform_(lin.weight)
        if bias:
            lin.bias.data.fill_(0.0)

        # sequential
        if batch_norm:
            self._sequential = nn.Sequential(lin, nn.BatchNorm1d(out_features))
        else:
            self._sequential = nn.Sequential(lin)
        self._output_dim = out_features

    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self._sequential(x))


class Conv1dMultiLayer(nn.Module):
    """Multiple conv1d layers with Relu and Dropout."""
    def __init__(self, layer_sizes: List[int], batch_norm: bool = False, dropout_keep: float = 1.0,
                 dropout_last: bool = False):
        super().__init__()
        layers: List[nn.Module] = list()

        # input and hidden layers
        for size_in, size_out in zip(layer_sizes[:-2], layer_sizes[1:-1]):
            layers.append(Conv1d(size_in, size_out, 1, bias=True, batch_norm=batch_norm))
            if dropout_keep < 1.0:
                layers.append(nn.Dropout(1.0 - dropout_keep))

        # output layer
        layers.append(Conv1d(layer_sizes[-2], layer_sizes[-1], 1, bias=True, batch_norm=batch_norm))
        if dropout_last and dropout_keep < 1.0:
            layers.append(nn.Dropout(1.0 - dropout_keep))

        # create sequential
        self._sequential = nn.Sequential(*layers)
        self._output_dim = layer_sizes[-1]

    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._sequential(x)


class LinearMultiLayer(nn.Module):
    """Multiple linear layers with Relu and Dropout."""
    def __init__(self, layer_sizes: List[int], batch_norm: bool = False, dropout_keep: float = 1.0,
                 dropout_last: bool = False):
        super().__init__()
        layers: List[nn.Module] = list()

        # input and hidden layers
        for size_in, size_out in zip(layer_sizes[:-2], layer_sizes[1:-1]):
            layers.append(Linear(size_in, size_out, bias=True, batch_norm=batch_norm))
            if dropout_keep < 1.0:
                layers.append(nn.Dropout(1.0 - dropout_keep))

        # output layer
        layers.append(Linear(layer_sizes[-2], layer_sizes[-1], bias=True, batch_norm=batch_norm))
        if dropout_last and dropout_keep < 1.0:
            layers.append(nn.Dropout(1.0 - dropout_keep))

        # create sequential
        self._sequential = nn.Sequential(*layers)
        self._output_dim = layer_sizes[-1]

    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._sequential(x)
