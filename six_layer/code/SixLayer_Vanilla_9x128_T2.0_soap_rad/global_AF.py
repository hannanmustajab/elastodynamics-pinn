import torch.nn as nn
import torch
import os
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


class NeuralNet(nn.Module):

    def __init__(self, input_dimension, output_dimension, n_hidden_layers, neurons, regularization_param=None, regularization_exp=None, retrain_seed=42):
        super(NeuralNet, self).__init__()
        # Number of input dimensions n
        self.input_dimension = input_dimension
        # Number of output dimensions m
        self.output_dimension = output_dimension
        # Number of neurons per layer
        self.neurons = neurons
        # Number of hidden layers
        self.n_hidden_layers = n_hidden_layers
        # Activation function
        self.activation = make_activation('sine') #CustomActivation()
        self.regularization_param = regularization_param
        # Regularization exponent
        self.regularization_exp = regularization_exp
        # Random seed for weight initialization
        # self.A = nn.Parameter(torch.tensor(0.1))
        # self.A = (torch.tensor(0.1))

        self.input_layer = nn.Linear(self.input_dimension, self.neurons)
        self.hidden_layers = nn.ModuleList([nn.Linear(self.neurons, self.neurons) for _ in range(n_hidden_layers - 1)])
        self.output_layer = nn.Linear(self.neurons, self.output_dimension)
        self.retrain_seed = retrain_seed
        self.device = device
        # Random Seed for weight initialization
        self.init_xavier()

    def forward(self, x):
        # The forward function performs the set of affine and non-linear transformations defining the network
        # (see equation above)
        x = self.activation(self.input_layer(x))
        for k, l in enumerate(self.hidden_layers):
            x = self.activation(l(x))
        return self.output_layer(x)

    def init_xavier(self):
        torch.manual_seed(self.retrain_seed)

        def init_weights(m):
            if type(m) == nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
                g = nn.init.calculate_gain('tanh')
                torch.nn.init.xavier_uniform_(m.weight, gain=g)
                # torch.nn.init.xavier_normal_(m.weight, gain=g)
                m.bias.data.fill_(0)

        self.apply(init_weights)

    def regularization(self):
        reg_loss = 0
        for name, param in self.named_parameters():
            if 'weight' in name:
                reg_loss = reg_loss + torch.norm(param, self.regularization_exp)
        return self.regularization_param * reg_loss


# Fourier feature mapping
class FourierFeatureMapping(nn.Module):
    def __init__(self, input_dim, mapping_size, scale=1.0):
        super(FourierFeatureMapping, self).__init__()
        self.device = device
        self.B = nn.Parameter(torch.randn((input_dim, mapping_size), device=self.device) * scale)  # Random Fourier coefficients
        # self.register("B", self.B)
 
    def forward(self, x):
        # Apply the Fourier mapping
        x_proj = np.pi * x @ self.B
        # x_proj = x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class FourierNeuralNet(nn.Module):

    def __init__(self, input_dimension, output_dimension, n_hidden_layers, neurons, mapping_size, scale, activation=nn.Tanh(), A_init=0.1, 
                 regularization_param=None, regularization_exp=None, retrain_seed=42):
        super(FourierNeuralNet, self).__init__()

        # Fourier Mapping
        self.fourier_mapping = FourierFeatureMapping(input_dimension, mapping_size, scale)

        # Input dimension is now the size after Fourier mapping
        self.input_dimension = mapping_size * 2  # After Fourier feature mapping
        self.output_dimension = output_dimension
        self.neurons = neurons
        self.n_hidden_layers = n_hidden_layers
        self.activation = make_activation('sine') #nn.Tanh()
        self.regularization_param = regularization_param
        self.regularization_exp = regularization_exp
        # self.A = (torch.tensor(A_init))  # Learnable scaling factor

        # Define layers
        self.input_layer = nn.Linear(self.input_dimension, self.neurons)
        self.hidden_layers = nn.ModuleList([nn.Linear(self.neurons, self.neurons) for _ in range(n_hidden_layers - 1)])
        self.output_layer = nn.Linear(self.neurons, self.output_dimension)
        self.retrain_seed = retrain_seed

        # Random Seed for weight initialization
        self.init_xavier()

    def forward(self, x):
        # Apply Fourier feature mapping to the input
        x_mapped = self.fourier_mapping(x)
        
        # Apply scaling with learnable parameter A
        x = self.activation(self.input_layer(x_mapped))
        for l in self.hidden_layers:
            x = self.activation(l(x))
        return self.output_layer(x)

    def init_xavier(self):
        torch.manual_seed(self.retrain_seed)

        def init_weights(m):
            if isinstance(m, nn.Linear):
                g = nn.init.calculate_gain('tanh')
                torch.nn.init.xavier_uniform_(m.weight, gain=g)
                m.bias.data.fill_(0.1)

        self.apply(init_weights)

    def regularization(self):
        if self.regularization_param is not None and self.regularization_exp is not None:
            reg_loss = 0
            for name, param in self.named_parameters():
                if 'weight' in name:
                    reg_loss += torch.norm(param, self.regularization_exp)
            return self.regularization_param * reg_loss
        return 0