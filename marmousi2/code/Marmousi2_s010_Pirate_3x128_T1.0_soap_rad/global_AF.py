import torch.nn as nn
import torch
import os
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
torch.manual_seed(42)



class Sine(nn.Module):
    def __init__(self, w0: float = 1.0):
        """Sine activation function with w0 scaling support.

        Example:
            >>> w = torch.tensor([3.14, 1.57])
            >>> Sine(w0=1)(w)
            torch.Tensor([0, 1])

        :param w0: w0 in the activation step `act(x; w0) = sin(w0 * x)`.
            defaults to 1.0
        :type w0: float, optional
        """
        super(Sine, self).__init__()
        self.w0 = w0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input(x)
        return torch.sin(self.w0 * x)

    @staticmethod
    def _check_input(x):
        if not isinstance(x, torch.Tensor):
            raise TypeError(
                'input to forward() must be torch.xTensor')



def make_activation(act: str):
    """
    Build an activation module by name.

    NOTE: this must match the activation the checkpoint was trained with. Parameter
    shapes are identical for 'sine' and 'tanh', so load_state_dict() succeeds either
    way and a mismatch fails silently -- the network simply computes a different
    function. The curriculum checkpoints under experiments/curriculum/ were trained
    with 'sine'.
    """
    if act == 'sine':
        return Sine(w0=1.0)
    elif act == 'tanh':
        return nn.Tanh()
    raise ValueError(f"unknown activation {act!r}, expected 'sine' or 'tanh'")


class RWFLinear(nn.Module):
    """
    Dense layer with random weight factorization, as in jaxpi (Dense with reparam "weight_fact"):
    weight = diag(g) v, with g = exp(mean + stddev * N(0, 1)) drawn per output neuron and
    v = w / g for a Glorot-uniform w (no gain). g, v and the bias are trained.
    At initialization the layer computes the same function as a plain layer with weight w.
    """
    def __init__(self, in_features: int, out_features: int, mean: float = 1.0, stddev: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mean = mean
        self.stddev = stddev
        self.g = nn.Parameter(torch.ones(out_features))
        self.v = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        w = torch.empty(self.out_features, self.in_features)
        nn.init.xavier_uniform_(w)
        g = torch.exp(self.mean + self.stddev * torch.randn(self.out_features))
        with torch.no_grad():
            self.g.copy_(g)
            self.v.copy_(w / g[:, None])
            self.bias.zero_()

    @property
    def weight(self) -> torch.Tensor:
        return self.g[:, None] * self.v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class PirateNetBlock(nn.Module):
    """
    Three-layer residual block that blends two inputs (u, v) and
    applies the activation after each dense layer.
    """
    def __init__(self, hidden_dim: int, act: str = 'sine', rwf_mean: float = 1.0, rwf_stddev: float = 0.1):
        super().__init__()
        self.dense1 = RWFLinear(hidden_dim, hidden_dim, rwf_mean, rwf_stddev)
        self.dense2 = RWFLinear(hidden_dim, hidden_dim, rwf_mean, rwf_stddev)
        self.dense3 = RWFLinear(hidden_dim, hidden_dim, rwf_mean, rwf_stddev)

        # learnable blend factor for residual connection
        self.alpha = nn.Parameter(torch.zeros(1))

        # single activation module reused at every layer
        self.act = make_activation(act)

    def forward(self, x: torch.Tensor, v: torch.Tensor, u_minus_v: torch.Tensor) -> torch.Tensor:
        # Same function as f * u + (1 - f) * v and alpha * h + (1 - alpha) * x, written with
        # fewer elementwise ops on the hidden-width tensors. u - v is the same in both blends
        # and in every block, so PirateNet.forward computes it once and passes it in. The
        # second-order autograd graph traverses these tensors on every grad call, so the op
        # count here is multiplied by roughly the number of backward passes per residual.
        f  = self.act(self.dense1(x))
        z1 = v + f * u_minus_v

        g  = self.act(self.dense2(z1))
        z2 = v + g * u_minus_v

        h  = self.act(self.dense3(z2))
        return x + self.alpha * (h - x)

class PirateNet(nn.Module):
    def __init__(self, input_dim, output_dim, num_blocks, hidden_dim=256, s=10.0, act='sine',
                 rwf_mean=1.0, rwf_stddev=0.1):
        super(PirateNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_blocks = num_blocks
        self.hidden_dim = hidden_dim
        self.s = s
        self.act_name = act
        self.activation = make_activation(act)  # Activation function for the network

        # Fourier embedding as in jaxpi: [cos(x B), sin(x B)], B ~ N(0, s^2), trainable.
        # No factor pi: s is the standard deviation of the angular frequencies.
        self.B = nn.Parameter(torch.randn(input_dim, hidden_dim // 2) * self.s)
        self.embedding = lambda x: torch.cat(
            [torch.cos(torch.matmul(x, self.B)), torch.sin(torch.matmul(x, self.B))], dim=-1
        )

        # List of PirateNetBlock layers
        self.blocks = nn.ModuleList([PirateNetBlock(hidden_dim, act=act, rwf_mean=rwf_mean, rwf_stddev=rwf_stddev)
                                     for _ in range(num_blocks)])
        self.U = RWFLinear(hidden_dim, hidden_dim, rwf_mean, rwf_stddev)  # U layer for transformations
        self.V = RWFLinear(hidden_dim, hidden_dim, rwf_mean, rwf_stddev)  # V layer for transformations
        
        # Final output layer
        self.final_layer = RWFLinear(hidden_dim, output_dim, rwf_mean, rwf_stddev)
        self.retrain_seed = torch.manual_seed(42)
        self.init_weights()  # Draw all dense layers again from seed 42
    
    def init_weights(self):
        # Glorot uniform without gain, zero bias, random weight factorization (see RWFLinear).
        for m in self.modules():
            if isinstance(m, RWFLinear):
                m.reset_parameters()


    # def initialize_weights(self):
    #     for module in self.modules():
    #         if isinstance(module, nn.Linear):
    #             torch.nn.init.xavier_uniform_(module.weight)  # Xavier initialization for weights
    #             if module.bias is not None:
    #                 module.bias.data.zero_()  # Initialize biases to zero

    def forward(self, x):
        x = self.embedding(x)  # Apply embedding transformation
        u = self.activation(self.U(x))  # Apply transformation for u
        v = self.activation(self.V(x))  # Apply transformation for v
        u_minus_v = u - v  # hoisted: the same in both blends of every block
        for block in self.blocks:
            x = block(x, v, u_minus_v)  # Pass through blocks
        return self.final_layer(x)  # Return the final output



