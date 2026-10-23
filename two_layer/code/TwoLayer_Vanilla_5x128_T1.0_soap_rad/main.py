import numpy as np
import torch.optim as optim
import torch
from torch.utils.data import DataLoader, Dataset
from global_AF import NeuralNet
from soap import SOAP
import time
torch.manual_seed(128)
import os
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn as nn
from psutil import virtual_memory
from matplotlib.colors import LogNorm
import argparse
import pickle
import json

import matplotlib.pyplot as plt

import scienceplots
plt.style.use(['science','nature','no-latex'])

plt.rcParams.update({'font.size': 8})

################################################################################################
# CONFIGURATION
################################################################################################
print("################################################################################################")
print("Configuration started...")
# Access the t1 value

# MUST match the checkpoint we warm-start from (curriculum run used t1 = 0.2).
# t1 sets the hard-constraint ansatz time-scale; changing it invalidates the
# learned perturbation.
t1_value = 0.2

print(f"Running simulation with short time t1 = {t1_value}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# TF32 is off. With it on, the loss value barely changes but its gradient is off by about 8%,
# and L-BFGS with the strong-Wolfe line search stops moving (measured on 2026-10-11 on the
# PirateNet 2 x 64 run, which froze at a loss of 8,238). TF32 was worth about 3-4% wall clock.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# Model configuration
n_layers = 5   # hidden layers
n_neurons = 128
mapping_size = 32
scale = 1   # not used by the Vanilla network
# Set per run (see the RUN CONFIG block printed before training).
activation = 'sine'

# Training configuration
# 100,000 collocation points in the first set: 30,000 for t <= source_t_max, most of them where the
# wave is, and 70,000 Sobol points spread evenly over the rest of the window [0, 1]. Every
# resampling event draws 100,000 new points.
n_int = 70000              # Sobol points for t in (source_t_max, tmax], whole spatial domain
n_early_uniform = 4000      # Sobol points for t in [0, source_t_max], whole spatial domain
n_t0_uniform = 0            # uniform Sobol points at t = 0
n_source = 22000            # Sobol points with t <= source_t_max and |x|, |y| <= source_half_width
n_source_t0 = 4000          # Sobol points at t = 0 with |x|, |y| <= source_t0_half_width
source_t_max = 0.3
source_half_width = 0.75    # the wave stays inside this square until t = source_t_max
source_t0_half_width = 0.4  # the initial displacement is below 0.3% of its peak outside this square

# Absorbing boundary term (Lysmer-Kuhlemeyer): on each edge, traction + rho*c_p*(normal velocity)
# + rho*c_s*(tangential velocity) = 0. A wave leaving at normal incidence satisfies it; a wave
# entering through the edge or a field drifting at the edge does not.
n_boundary_per_edge = 2000   # Sobol points in (t, position along the edge) on each edge: 2,000 per time unit
# Weight of the mean squared boundary residual. 1/sigma^2 with sigma = 0.1, the source width, puts
# the boundary residual (a stress) on the scale of the PDE residual (a stress per length).
bc_weight = 100.0
n_sb = 200
n_tb = 2000
tmax = 3

# Optimizer configuration
# Set per run (see the RUN CONFIG block printed before training).
lr = 2.0
max_iter = 200
max_eval = 200
history_size = 100
# Loss evaluations per training window: 400 optimizer steps of up to max_eval evaluations each.
evaluation_budget = 400 * max_eval
# Largest number of collocation points in one backward pass (see compute_loss). It limits the GPU
# memory in use; the loss and its gradient do not depend on it.
# 68250 = 273000 / 4, the fastest arrangement measured on the 32 GB card: 4 passes run at
# 0.73 s per loss evaluation and peak at 11.3 GB, against 1.57 s for 14 passes of 20000.
# Larger passes are slower, not faster -- 2 passes of 136500 peak at 22.9 GB and the allocator
# thrashes; a single pass over all 273000 points needs about 46 GB and runs out of memory.
max_points_per_pass = 68250

# SOAP with mini-batches and residual-based resampling (this run). The L-BFGS settings above
# are not used.
soap_epochs = 10000              # one epoch = one pass over the collocation set and the boundary points
# Batches per epoch. 0: found at the start of training as the smallest number of equal batches
# whose loss and gradient stay within soap_memory_fraction of the memory this process can use on
# the card (author's rule: reduce the batch size until it fits). A continued run keeps its number.
soap_batches_per_epoch = 0
soap_memory_fraction = 0.8       # PyTorch reserves about 17% more memory than it allocates
soap_lr = 1e-3                   # learning rate after the warm-up
soap_betas = (0.99, 0.999)       # Wang, Bhartari, Li, Perdikaris 2025, Appendix G
soap_precondition_frequency = 1  # preconditioner updated at every step
soap_warmup_epochs = 1000        # linear warm-up from 0 over this many epochs
soap_decay_rate = 0.9            # factor per soap_decay_epochs epochs after the warm-up
soap_decay_epochs = 500          # 2,000 steps at 4 batches per epoch, the interval of the first SOAP run
# The same two intervals in optimizer steps; fit_soap sets them once the number of batches is known.
soap_warmup_steps = soap_warmup_epochs * soap_batches_per_epoch
soap_decay_steps = soap_decay_epochs * soap_batches_per_epoch
soap_shuffle_seed = 128          # order of the points within an epoch
soap_print_every = 50            # epochs between loss lines
soap_monitor_every = 500         # epochs between error evaluations and checkpoints
soap_test_points = 200000        # fixed uniform points on which the PDE term is reported
soap_test_seed = 2026
# Residual-based resampling (RAD, Wu et al. 2023): every rad_interval epochs the whole collocation
# set is drawn anew from rad_candidates uniform points with probability proportional to
# r^rad_power / mean(r^rad_power) + rad_c, r = residual magnitude clipped at its 99th percentile.
rad_interval = 1000
rad_candidates = 2000000
rad_power = 2.0
rad_c = 1.0
rad_seed = 4000                  # the event at epoch e uses the seed rad_seed + e
rad_eval_points = 34125          # points per residual evaluation pass (half the memory of a training batch)
rad_map_bins = 100               # bins per direction of the residual and point-density maps


def soap_lr_factor(step):
    """Learning rate of SOAP step `step` divided by soap_lr: linear warm-up from 0, then exponential decay."""
    if step < soap_warmup_steps:
        return step / soap_warmup_steps
    return soap_decay_rate ** ((step - soap_warmup_steps) / soap_decay_steps)

# Logging configuration
print(f"Model Configuration:\n"
    f"  Number of layers: {n_layers}\n"
    f"  Number of neurons: {n_neurons}\n"
    f"  Scale: {scale}\n"
    f"  Mapping size: {mapping_size}\n"
    f"  Number of interior points: {n_int}\n")

print(f"Optimizer Configuration:\n"
    f"  Learning rate (LBFGS): {lr}\n"
    f"  Max iterations: {max_iter}\n"
    f"  Max evaluations: {max_eval}\n"
    f"  History size: {history_size}\n")

# Check available RAM
ram_gb = virtual_memory().total / 1e9
print(f"Your runtime has {ram_gb:.1f} gigabytes of available RAM")

if ram_gb < 20:
    print("Not using a high-RAM runtime")
else:
    print("You are using a high-RAM runtime!")


print("################################################################################################")

################################################################################################
# CONFIGURATION
################################################################################################


class SeismicSimulation:
    def __init__(self, lambda_vals, mu_vals, rho_vals, sharpness=80):
        """
        Initializes the SeismicSimulation with multiple layers.

        Args:
            lambda_vals (list or array): List of Lamé's first parameter (λ) for each layer.
            mu_vals (list or array): List of Lamé's second parameter (μ) for each layer.
            rho_vals (list or array): List of densities (ρ) for each layer.
            sharpness (float, optional): Controls the steepness of each transition. Default is 80.
        """
        # Validate input lengths
        if not (len(lambda_vals) == len(mu_vals) == len(rho_vals)):
            raise ValueError("lambda_vals, mu_vals, and rho_vals must have the same length.")

        self.lambda_vals = lambda_vals
        self.mu_vals = mu_vals
        self.rho_vals = rho_vals
        self.n_layers = len(lambda_vals)
        self.sharpness = sharpness

        if self.n_layers < 1:
            raise ValueError("There must be at least one layer.")

    def smooth_step(self, y, center=0.0, sharpness=80):
        """
        Smooth step function based on tanh for a smooth transition.

        Args:
            y (torch.Tensor): Spatial coordinate array.
            center (float): The y-coordinate where the transition occurs.
            sharpness (float): Controls the steepness of the transition.

        Returns:
            torch.Tensor: A smooth transition factor between 0 and 1.
        """
        return (torch.tanh(sharpness * (y - center)) + 1) / 2

    def get_material_properties(self, y, y_min=None, y_max=None):
        """
        Returns smoothly varying material properties (lambda, mu, rho) based on spatial coordinate y.

        Args:
            y (torch.Tensor): Spatial coordinate array.

        Returns:
            tuple: (lambda_solid, mu_solid, rho_solid)
        """
        if self.n_layers == 1:
            # Single layer, no transitions
            lambda_solid = torch.full_like(y, self.lambda_vals[0], dtype=torch.float32)
            mu_solid = torch.full_like(y, self.mu_vals[0], dtype=torch.float32)
            rho_solid = torch.full_like(y, self.rho_vals[0], dtype=torch.float32)
            return lambda_solid, mu_solid, rho_solid

        # Determine transition centers based on y's range
        # With bounds given, the centers do not depend on the batch and stay outside the
        # autograd graph. Without bounds, fall back to the range of y.
        if y_min is None:
            y_min = y.min().item()
        if y_max is None:
            y_max = y.max().item()
        total_transitions = self.n_layers - 1
        transition_centers = torch.linspace(y_min, y_max, steps=self.n_layers + 1)[1:-1]  # Exclude first and last

        # Initialize material properties with the first layer's properties
        lambda_solid = torch.full_like(y, self.lambda_vals[0], dtype=torch.float32)
        mu_solid = torch.full_like(y, self.mu_vals[0], dtype=torch.float32)
        rho_solid = torch.full_like(y, self.rho_vals[0], dtype=torch.float32)

        # Iterate over each transition and update material properties
        for i in range(total_transitions):
            # Compute the smooth step for the current transition
            transition = self.smooth_step(y, center=float(transition_centers[i]), sharpness=self.sharpness)

            # Linearly interpolate between the current layer and the next layer
            lambda_solid = lambda_solid * (1 - transition) + self.lambda_vals[i + 1] * transition
            mu_solid = mu_solid * (1 - transition) + self.mu_vals[i + 1] * transition
            rho_solid = rho_solid * (1 - transition) + self.rho_vals[i + 1] * transition

        return lambda_solid, mu_solid, rho_solid

    def get_number_of_layers(self):
        """
        Returns the number of layers in the simulation.

        Returns:
            int: Number of layers.
        """
        return self.n_layers


def load_all_ground_truth_wavefields(input_directory_x, input_directory_y, num_time_steps):
    """
    Load all ground truth wavefields for Ux and Uy from the given directories.

    Parameters:
        input_directory_x (str): Directory path for Ux snapshots.
        input_directory_y (str): Directory path for Uy snapshots.
        num_time_steps (int): Total number of time steps to load.

    Returns:
        tuple: Two lists containing the loaded wavefields for Ux and Uy respectively.
    """
    loaded_res_list_x = []
    loaded_res_list_y = []
    loaded_res_list_u = []

    for t in range(num_time_steps):
        # Construct file names for the given time step
        ux_filename = os.path.join(input_directory_x, f"snapshot_{t:04d}.npy")
        uy_filename = os.path.join(input_directory_y, f"snapshot_{t:04d}.npy")
        # u = np.sqrt(ux_pinn ** 2 + uy_pinn ** 2)
        # Load the Ux and Uy snapshots and append them to the respective lists
        loaded_res_list_x.append(np.load(ux_filename))
        loaded_res_list_y.append(np.load(uy_filename))
        loaded_res_list_u.append(np.sqrt(np.load(ux_filename) ** 2 + np.load(uy_filename) ** 2))

    return loaded_res_list_x, loaded_res_list_y, loaded_res_list_u


# Example usage:
input_directory_x = '/gxfs_work/cau/sungw910/PINN/pinn-nesh/paper_exp/simple_media_corrected/reference_two_layer/snapshots_x'
input_directory_y = '/gxfs_work/cau/sungw910/PINN/pinn-nesh/paper_exp/simple_media_corrected/reference_two_layer/snapshots_y'
num_time_steps = 101  # snapshot_0000 .. snapshot_0100
# Spacing between ground-truth snapshots, in the same time units the PINN uses.
# Verified empirically: sweeping dt and scoring the trained tmax=1 checkpoint against
# these snapshots gives a sharp minimum at exactly 0.0100 (0.0100 -> 0.134 rel-L2,
# 0.0102 -> 0.208, 0.0110 -> 0.683). So frame i is at t = i * 0.01.
dt_frame = 0.01
# The training window ends at t = 1.00.
GT_MAX_INDEX = num_time_steps - 1
GT_MAX_TIME = GT_MAX_INDEX * dt_frame

ux_wavefields, uy_wavefields, u_wavefields = load_all_ground_truth_wavefields(input_directory_x, input_directory_y, num_time_steps)

print(ux_wavefields[0].shape, uy_wavefields[0].shape, u_wavefields[0].shape)

class Pinns:
    def __init__(self, n_int_,t1,tmax):
        self.n_int = n_int_
        self.tmax = tmax
        #iteration number   
        self.iter = 0
    
        # Define the path for saving experiments (kept separate per tmax leg so we
        # never overwrite checkpoints/plots from earlier legs, e.g. the tmax=1 run)
        self.PATH = f'experiments/full_time_ext_tmax_{tmax}/'
        if not os.path.exists(self.PATH):
            os.makedirs(self.PATH)
        print(f"Experiment path: {self.PATH}")
        # Update domain extrema to include two spatial dimensions
        
        self.domain_extrema = torch.tensor([[0.0,  tmax],  # Time dimension
                                            [-1.5, 1.5],    # Space dimension x
                                            [-1.5, 1.5]])   # Space dimension y

        print(self.domain_extrema)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
          # Generator of Sobol sequences
        self.soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])
        # Separate sequence for the extra points in the source region
        self.soboleng_source = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])

        # F Dense NN to approximate the solution of the underlying heat equation
        self.network_u = NeuralNet(input_dimension=3, output_dimension=2,
                                   n_hidden_layers=n_layers, neurons=n_neurons).to(self.device)
        
        # Training sets S_sb, S_tb, S_int as torch dataloader
        self.training_set_int = self.assemble_datasets()
   
        self.network_u = self.network_u.to(device)

        print(self.network_u)
       
        # training parameters         
        self.t1 = t1
        print(self.t1)
        self.sigma_quake = float(0.1)
        self.sigma_quake = torch.tensor(self.sigma_quake)
        self.sigma_quake = self.sigma_quake.to(device)
        self.mu_quake = torch.tensor([2, 2])
        self.mu_quake = self.mu_quake.to(device)
        self.lambda_solid = 20
        self.mu_solid = 30
        self.rho_solid = 100

        # Training sets S_sb, S_tb, S_int as torch dataloader
        self.training_set_int = self.assemble_datasets()
        print(len(self.training_set_int))

        # Points and outward normals on the four edges for the absorbing-boundary term
        boundary_points, boundary_normals = self.assemble_boundary_points()
        self.boundary_points = boundary_points.to(self.device).requires_grad_(True)
        self.boundary_normals = boundary_normals.to(self.device)
        print(f"Boundary points: {len(self.boundary_points)} ({n_boundary_per_edge} per edge)")
        # lambda_values, mu_values, rho_values = simulation.get_material_properties(y)
        # lbfgs_params = [0.1, 500, None, -1, -1, 50] 
        # # Optimization Hyperparameters for LBFGS [learning rate, max iterations, max evaluations, tolerance gradient, tolerance change, history size]

        
        self.lbfgs = optim.LBFGS(list(self.network_u.parameters()),
                              lr=lr,
                              max_iter=max_iter,
                              max_eval=max_eval,
                              history_size=history_size,
                              line_search_fn="strong_wolfe",
                              tolerance_grad=1e-8, #previous 1e-8
                              tolerance_change=1.0 * np.finfo(float).eps
                            #   tolerance_change=1.0 * np.finfo(float).eps
                              )

        
        self.adam = optim.Adam(list(self.network_u.parameters()),
                            lr=float(0.0009)
                            # previous best 0.001
                            )
        
        self.gamma = 0.5  # Desired decay factor after 20,000 steps
        self.scheduler = ExponentialLR(self.adam, gamma=self.gamma)
        self.decay_steps = 20000

        # Optimizer of this run. The scheduler is stepped once per optimizer step.
        self.soap = SOAP(list(self.network_u.parameters()), lr=soap_lr, betas=soap_betas, weight_decay=0.0,
                         precondition_frequency=soap_precondition_frequency)
        self.soap_scheduler = None  # built in fit_soap, once the number of batches per epoch is known

        self.collocation_loss = []
        self.af_loss = []
        self.training_loss = []

    ################################################################################################
    # Function to linearly transform a tensor whose value are between 0 and 1
    # to a tensor whose values are between the domain extrema
    def convert(self, tens):
        assert (tens.shape[1] == self.domain_extrema.shape[0])
        # return tens * (self.domain_extrema[:, 1] - self.domain_extrema[:, 0]) + self.domain_extrema[:, 0]
        return tens * (self.domain_extrema[:, 1] - self.domain_extrema[:, 0]) + self.domain_extrema[:, 0]
    
    # Function to add 10,000 points at t=0
    def add_points_at_t0(self, n_points=20000):
        """
        Add n_points at t=0 using Sobol sampling for the spatial components.

        Parameters:
        n_points (int): Number of points to add (default: 10,000).
        
        Returns:
        Tensor: Input tensor with points at t=0.
        """
        # Generate Sobol points (n_points) in the spatial dimensions (excluding time)
        sobol_points = self.soboleng.draw(n_points)

        # Convert the Sobol points to fit within the domain extrema
        input_points = self.convert(sobol_points)

        # Set the time component (first column) to 0
        input_points[:, 0] = 0

        return input_points
    
    
    #  Function returning the input-output tensor required to assemble the training set S_int corresponding to the interior domain where the PDE is enforced
    def add_source_points(self, n_points, at_t0=False):
        """
        Sobol points in the source region: t in [0, min(source_t_max, tmax)], x and y within
        source_half_width of the source. With at_t0=True the time is set to 0 and the points
        lie within source_t0_half_width.
        """
        t_hi = min(source_t_max, float(self.tmax))
        half = source_t0_half_width if at_t0 else source_half_width
        lo = torch.tensor([0.0, -half, -half])
        hi = torch.tensor([t_hi, half, half])
        points = self.soboleng_source.draw(n_points) * (hi - lo) + lo
        if at_t0:
            points[:, 0] = 0
        return points

    def add_interior_points(self):
        # torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])
        # input_s = self.convert(soboleng.draw(int(self.n_collocation_points)))
        input_int = self.convert(self.soboleng.draw(self.n_int))
        # Keep these points in t in (source_t_max, tmax]; earlier times get their own points.
        t_max = float(self.tmax)
        if t_max > source_t_max:
            input_int[:, 0] = source_t_max + input_int[:, 0] * (t_max - source_t_max) / t_max
        # output_int = torch.zeros((input_int.shape[0], 1))
        return input_int

    def add_early_points(self, n_points):
        """
        Sobol points for t in [0, min(source_t_max, tmax)] over the whole spatial domain.
        """
        points = self.convert(self.soboleng.draw(n_points))
        t_max = float(self.tmax)
        points[:, 0] = points[:, 0] * min(source_t_max, t_max) / t_max
        return points

    # Function returning the training sets S_sb, S_tb, S_int as dataloader
    def assemble_datasets(self):
    
        # Uniform Sobol points over the whole space-time box, plus extra points where the wave
        # starts. The counts are set in the configuration block.
        input_int = self.add_interior_points()
        empty = torch.empty(0, self.domain_extrema.shape[0])
        input_t0 = self.add_points_at_t0(n_t0_uniform) if n_t0_uniform > 0 else empty
        input_src = self.add_source_points(n_source) if n_source > 0 else empty
        input_src_t0 = self.add_source_points(n_source_t0, at_t0=True) if n_source_t0 > 0 else empty
        input_early = self.add_early_points(n_early_uniform) if n_early_uniform > 0 else empty
        training_set = torch.cat((input_int, input_early, input_t0, input_src, input_src_t0), dim=0)
        print(f"Collocation points: {len(input_int)} for t > {source_t_max}, {len(input_early)} uniform for t <= {source_t_max}, "
              f"{len(input_t0)} uniform at t=0, {len(input_src)} source region, {len(input_src_t0)} source region at t=0, "
              f"total {len(training_set)}")
        
        training_set = training_set.to(self.device)
        
        # Two-layer medium, listed from y_min upward: lambda = 25, mu = 35 below the interface
        # at y = 0 and lambda = 20, mu = 30 above it.
        lambda_vals = [25, 20]
        mu_vals     = [35, 30]

        rho_vals    = [100, 100]

        # Instantiate the simulation with two layers
        simulation = SeismicSimulation(lambda_vals, mu_vals, rho_vals, sharpness=90)
        # Kept for compute_pde_residual, which evaluates lambda and mu from the differentiable input.
        self.simulation = simulation
        self.y_bounds = (float(self.domain_extrema[2, 0]), float(self.domain_extrema[2, 1]))

        # Material values at the training points, used only for the velocity-model plot.
        self.lambda_values, self.mu_values, self.rho_values = simulation.get_material_properties(
            training_set[:,2], y_min=self.y_bounds[0], y_max=self.y_bounds[1])


        # self.lambda_values, self.mu_values, self.rho_values = simulation.get_material_properties(training_set[:,2])
        self.plot_and_save_velocity_model(self.lambda_values, self.mu_values, self.rho_values, training_set)

    
        training_set_int = DataLoader(torch.utils.data.TensorDataset(training_set), batch_size=len(training_set), shuffle=False)
        
        
        return training_set_int
    
    def initial_condition_ansatz(self, input_tensor):
        """
        Applies the initial condition and returns the perturbations in both x and y directions.

        Args:
            input_tensor (torch.Tensor): The input tensor, typically containing the spatial and temporal coordinates.

        Returns:
            torch.Tensor: A tensor U containing perturbations in both x and y directions.
        """
        # Evaluate the models for displacement fields in x and y directions
        U_perturbation = self.network_u(input_tensor)  # Output from NN1

        # Get the initial condition for both x and y directions
        u0x, u0y = self.initial_condition_gaussian(input_tensor)

        # Same ansatz as before, with the ramp and the envelope hoisted out of the two
        # components (they do not depend on the component) and the result assembled with
        # torch.stack. Writing into a zeros tensor with two slice assignments added copy
        # nodes that the second-order graph traversed on every grad call.
        t = input_tensor[:, 0]
        ramp = torch.tanh(2.5 * t / self.t1) ** 2
        envelope = torch.exp(-0.5 * (1.5 * t / self.t1) ** 2)

        return torch.stack((U_perturbation[:, 0] * ramp + u0x * envelope,
                            U_perturbation[:, 1] * ramp + u0y * envelope), dim=1)


    # Gaussian Initial Condition
    def initial_condition_gaussian(self, input_tensor):
        """_summary_

        Args:
            input_tensor (_type_): _description_

        Returns:
            _type_: _description_
        """
        # ( x - mu ) ** 2
        # x = input_tensor[:, 1] - 0
        # y = input_tensor[:, 2] - 0
        sigma = 0.1
        x = input_tensor[:, 1] - 0
        y = input_tensor[:, 2] - 0
        gauss = torch.exp(- (x**2 + y**2) / (2*sigma**2))
        grad_x = -2 * x * gauss / (2*sigma**2)
        grad_y = -2 * y * gauss / (2*sigma**2)
        # Analytic maximum of |grad|, exp(-1/2) / sigma, reached at distance sigma from the source.
        # A maximum over the batch depends on the sampled points and, taken inside the autograd
        # graph, adds batch-wide terms to the residual.
        peak = float(np.exp(-0.5)) / sigma
        u0x = -grad_x / peak
        u0y = -grad_y / peak

        return u0x, u0y

    ################################################################################################
    # Function to compute the terms required in the definition of the TEMPORAL boundary res
    # idual
    
    def assemble_boundary_points(self):
        """
        Sobol points on the four edges of the spatial domain over the whole time window,
        with the outward unit normal of each point.
        """
        engine = torch.quasirandom.SobolEngine(dimension=2)
        t_hi = float(self.domain_extrema[0, 1])
        x_lo, x_hi = float(self.domain_extrema[1, 0]), float(self.domain_extrema[1, 1])
        y_lo, y_hi = float(self.domain_extrema[2, 0]), float(self.domain_extrema[2, 1])
        points, normals = [], []
        for axis, value, sign in ((1, x_lo, -1.0), (1, x_hi, 1.0), (2, y_lo, -1.0), (2, y_hi, 1.0)):
            draw = engine.draw(n_boundary_per_edge)
            p = torch.zeros(n_boundary_per_edge, 3)
            p[:, 0] = draw[:, 0] * t_hi
            if axis == 1:
                p[:, 1] = value
                p[:, 2] = y_lo + draw[:, 1] * (y_hi - y_lo)
            else:
                p[:, 1] = x_lo + draw[:, 1] * (x_hi - x_lo)
                p[:, 2] = value
            n = torch.zeros(n_boundary_per_edge, 2)
            n[:, axis - 1] = sign
            points.append(p)
            normals.append(n)
        return torch.cat(points, dim=0), torch.cat(normals, dim=0)

    def compute_boundary_residual(self, input_b, normals):
        """
        Residual of the absorbing condition sigma n + rho c_p (v.n) n + rho c_s (v - (v.n) n) = 0
        at boundary points input_b with outward normals; v is the velocity du/dt.

        Returns:
        Tensor of shape (N, 2): the x and y components of the residual.
        """
        U = self.initial_condition_ansatz(input_b)
        gradient_x = torch.autograd.grad(U[:, 0].sum(), input_b, create_graph=True)[0]
        gradient_y = torch.autograd.grad(U[:, 1].sum(), input_b, create_graph=True)[0]

        lam, mu, _ = self.simulation.get_material_properties(
            input_b[:, 2], y_min=self.y_bounds[0], y_max=self.y_bounds[1])

        exx, eyy = gradient_x[:, 1], gradient_y[:, 2]
        exy = 0.5 * (gradient_x[:, 2] + gradient_y[:, 1])
        sxx = lam * (exx + eyy) + 2.0 * mu * exx
        syy = lam * (exx + eyy) + 2.0 * mu * eyy
        sxy = 2.0 * mu * exy

        nx, ny = normals[:, 0], normals[:, 1]
        traction_x = sxx * nx + sxy * ny
        traction_y = sxy * nx + syy * ny

        vx, vy = gradient_x[:, 0], gradient_y[:, 0]
        v_normal = vx * nx + vy * ny
        z_p = torch.sqrt(self.rho_solid * (lam + 2.0 * mu))   # rho * c_p
        z_s = torch.sqrt(self.rho_solid * mu)                 # rho * c_s

        res_x = traction_x + z_p * v_normal * nx + z_s * (vx - v_normal * nx)
        res_y = traction_y + z_p * v_normal * ny + z_s * (vy - v_normal * ny)
        return torch.stack((res_x, res_y), dim=1)

    def compute_pde_residual(self, input_s):
        """
        Compute the PDE residual for the solid's governing equations.
        
        Args:
        input_s (Tensor): Input tensor containing spatial and temporal coordinates.
        
        Returns:
        Tensor: The PDE residual computed as rho * (d^2u/dt^2) - Div(stress tensor).
        """

        U = self.initial_condition_ansatz(input_s)
        # Calculate displacements u_x and u_y
        u_x = U[:, 0].unsqueeze(1)
        u_y = U[:, 1].unsqueeze(1)

        # Calculate gradients
        gradient_x = torch.autograd.grad(u_x.sum(), input_s, create_graph=True)[0]
        gradient_y = torch.autograd.grad(u_y.sum(), input_s, create_graph=True)[0]
        
        # Compute second time derivatives
        dt2_x = torch.autograd.grad(gradient_x[:, 0].sum(), input_s, create_graph=True)[0][:, 0]
        dt2_y = torch.autograd.grad(gradient_y[:, 0].sum(), input_s, create_graph=True)[0][:, 0]

        # Strain tensor calculation
        eps = 0.5 * torch.stack(
            (torch.stack((2.0 * gradient_x[:, 1], gradient_x[:, 2] + gradient_y[:, 1])),
            torch.stack((gradient_x[:, 2] + gradient_y[:, 1], 2.0 * gradient_y[:, 2]))),
            dim=1
        )

        # Lame parameters as functions of the differentiable input, so that the divergence of the
        # stress below includes the terms with d(lambda)/dy and d(mu)/dy.
        lam, mu, _ = self.simulation.get_material_properties(
            input_s[:, 2], y_min=self.y_bounds[0], y_max=self.y_bounds[1])

        # Stress tensor components
        stress_tensor_00 = lam * (eps[0, 0] + eps[1, 1]) + 2.0 * mu * eps[0, 0]
        stress_tensor_off_diag = 2.0 * mu * eps[0, 1]
        stress_tensor_11 = lam * (eps[0, 0] + eps[1, 1]) + 2.0 * mu * eps[1, 1]

        # Compute gradients of stress tensor components
        off_diag_grad = torch.autograd.grad(stress_tensor_off_diag.sum(), input_s, create_graph=True)[0]
        
        # Divergence of the stress tensor
        div_stress = torch.zeros(2, input_s.size(0), dtype=torch.float32, device=input_s.device)
        div_stress[0, :] = torch.autograd.grad(stress_tensor_00.sum(), input_s, create_graph=True)[0][:, 1] + \
                        off_diag_grad[:, 2]
        div_stress[1, :] = off_diag_grad[:, 1] + \
                        torch.autograd.grad(stress_tensor_11.sum(), input_s, create_graph=True)[0][:, 2]

        # Combine second time derivatives
        dt2_combined = torch.stack((dt2_x, dt2_y), dim=0)
        
        # PDE residual
        residual_solid = self.rho_solid * dt2_combined - div_stress
        
        return residual_solid.reshape(-1, 1)
    
    def get_time_steps(self,tmax: float):
        """
        Return a tensor of time steps starting at 0, up to tmax, drawn from the
        predefined 'master' list plus points near the current leg's tmax so
        periodic diagnostics actually cover the time range being trained on.
        """
        # Master list of "interesting" time points
        master_times = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 0.9, 1.0, 1.2, 1.5, 1.8]

        filtered = [t for t in master_times if t <= tmax]

        if tmax > 0.9:
            # Also sample near the current leg's tmax so we can actually see
            # PINN vs. ground-truth performance at the extended horizon.
            extra = [t for t in [tmax - 0.4, tmax - 0.2, tmax - 0.1, tmax] if t > 0.9]
            filtered = sorted(set(filtered + extra))

        # Snap to the ground-truth frame grid so each requested time lands exactly on a
        # snapshot rather than between two of them.
        filtered = sorted({round(round(t / dt_frame) * dt_frame, 6) for t in filtered})

        return torch.tensor(filtered)


    def compute_loss(self, input_s, verbose=True):
        """
        Total loss: mean squared PDE residual plus the weighted mean squared absorbing-boundary
        residual. The gradient is accumulated in the network parameters by this function.

        The PDE residual is evaluated in passes of at most max_points_per_pass collocation
        points. Each pass contributes sum(residual^2) / (number of residual values of the whole
        set) and calls backward(), so the accumulated gradient is that of the full-batch loss
        while the autograd graph in memory covers one pass only.
        
        Args:
        input_s (Tensor): Input tensor containing spatial and temporal coordinates.
        verbose (bool): If True, prints detailed loss information periodically.
        
        Returns:
        Tensor: The loss value, detached. Its gradient is already in the parameters, so the
        caller must not call backward() on it.
        """
        # Mean squared PDE residual, one pass at a time. There are two residual values per point.
        n_points = input_s.shape[0]
        loss_pde = torch.zeros((), device=input_s.device)
        for start in range(0, n_points, max_points_per_pass):
            residual_solid = self.compute_pde_residual(input_s[start:start + max_points_per_pass])
            loss_part = torch.sum(torch.square(residual_solid)) / (2 * n_points)
            loss_part.backward()
            loss_pde += loss_part.detach()
            del residual_solid, loss_part

        # Weighted mean squared absorbing-boundary residual, in a pass of its own.
        residual_boundary = self.compute_boundary_residual(self.boundary_points, self.boundary_normals)
        loss_boundary = torch.mean(torch.square(residual_boundary))
        (bc_weight * loss_boundary).backward()
        loss_boundary = loss_boundary.detach()
        del residual_boundary

        loss = loss_pde + bc_weight * loss_boundary
        
        # Initialize bookkeeping if not already set
        if not hasattr(self, 'loss_history'):
            self.loss_history = []
            self.iter = 0  # Initialize iteration counter

        # Record the loss history
        self.collocation_loss.append(loss.item())
        
        # Print detailed loss information at specified intervals
        if verbose and self.iter % 500 == 0:
            print(f'Iteration {self.iter} - Total loss: {round(loss.item(), 5)}')
            print(f'  PDE term: {loss_pde.item():.5f}  boundary term (unweighted): {loss_boundary.item():.5f}')
            if torch.cuda.is_available():
                print(f'  GPU memory peak so far: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB')
            # print('Ux amplitude: ', (self.network_ux.A).item())
            # print('Uy amplitude: ', (self.network_uy.A).item())
        
        if self.iter % 2000 == 0:
            timesteps = self.get_time_steps(self.tmax)
            wavefields = self.generate_pinn_wavefields(timesteps, device=self.device)
            self.compare_wavefields(wavefields, ux_wavefields, uy_wavefields,u_wavefields, timesteps)
            self.compare_wavefields_cross_section(wavefields, ux_wavefields, uy_wavefields,u_wavefields, timesteps)
            self.save_model(self.iter + 1)

        # Increment iteration count
        self.iter += 1

        return loss



    def minibatch_loss(self, input_s, input_b, normals_b):
        """
        Loss of one mini-batch with the definition of compute_loss: mean squared PDE residual
        (two values per point) plus bc_weight times the mean squared absorbing-boundary residual.

        Returns:
        (loss with its graph, PDE term detached, boundary term detached and unweighted)
        """
        residual_solid = self.compute_pde_residual(input_s)
        loss_pde = torch.sum(torch.square(residual_solid)) / (2 * input_s.shape[0])
        loss_boundary = torch.mean(torch.square(self.compute_boundary_residual(input_b, normals_b)))
        return loss_pde + bc_weight * loss_boundary, loss_pde.detach(), loss_boundary.detach()

    def uniform_points(self, n_points, generator):
        """n_points uniformly distributed points of the space-time domain, on the training device."""
        lo, hi = self.domain_extrema[:, 0], self.domain_extrema[:, 1]
        return (torch.rand(n_points, 3, generator=generator) * (hi - lo) + lo).to(self.device)

    def pde_residual_magnitude(self, points):
        """Norm of the two-component PDE residual at each point, without parameter gradients."""
        magnitudes = []
        for start in range(0, len(points), rad_eval_points):
            inputs = points[start:start + rad_eval_points].detach().clone().requires_grad_(True)
            residual = self.compute_pde_residual(inputs).detach().reshape(2, -1)
            magnitudes.append(torch.sqrt(residual[0] ** 2 + residual[1] ** 2))
            del residual, inputs
        return torch.cat(magnitudes)

    def pde_term(self, points):
        """PDE term of the loss on a point set: mean over the points of half the squared residual norm."""
        return float(torch.mean(self.pde_residual_magnitude(points) ** 2) / 2)

    def interface_distance(self, points):
        """Distance in y of each point to the nearest layer interface."""
        y_lo, y_hi = self.y_bounds
        interfaces = torch.linspace(y_lo, y_hi, self.simulation.get_number_of_layers() + 1, device=points.device)[1:-1]
        if len(interfaces) == 0:
            return torch.full((len(points),), float('inf'), device=points.device)
        return torch.min(torch.abs(points[:, 2:3] - interfaces[None, :]), dim=1).values

    def resample_points(self, epoch, n_points):
        """
        Residual-based adaptive distribution (RAD, Wu et al. 2023): a new collocation set of
        n_points points, drawn without replacement from rad_candidates uniform candidates with
        probability proportional to r^rad_power / mean(r^rad_power) + rad_c. r is the residual
        magnitude of the current network, clipped at its 99th percentile so that a few extreme
        points do not take all the weight; rad_c keeps the whole domain covered.
        """
        generator = torch.Generator().manual_seed(rad_seed + epoch)
        candidates = self.uniform_points(rad_candidates, generator)
        magnitude = self.pde_residual_magnitude(candidates)
        percentile_99 = torch.quantile(magnitude, 0.99)
        powered = torch.clamp(magnitude, max=percentile_99) ** rad_power
        probabilities = (powered / powered.mean() + rad_c).cpu()
        chosen = torch.multinomial(probabilities, n_points, replacement=False, generator=generator).to(self.device)
        new_points = candidates[chosen]

        def shares(points):
            early = points[:, 0] <= source_t_max
            near = self.interface_distance(points) < 0.02
            return [100 * float(mask.float().mean()) for mask in (early, near, early & near)]

        new, uniform = shares(new_points), shares(candidates)
        print(f"Resampling at epoch {epoch}: residual magnitude on {rad_candidates} uniform candidates: "
              f"median {float(magnitude.median()):.3f}, mean {float(magnitude.mean()):.3f}, "
              f"99th percentile {float(percentile_99):.3f}, max {float(magnitude.max()):.3f}")
        print(f"  New collocation set of {len(new_points)} points (uniform candidates in brackets): "
              f"{new[0]:.1f}% ({uniform[0]:.1f}%) with t <= {source_t_max}, "
              f"{new[1]:.1f}% ({uniform[1]:.1f}%) within 0.02 of an interface, "
              f"{new[2]:.2f}% ({uniform[2]:.2f}%) both")
        self.plot_resampling(epoch, candidates, magnitude, new_points)
        return new_points

    def plot_resampling(self, epoch, candidates, magnitude, new_points):
        """
        Figure of one resampling event. For six time windows: the mean residual magnitude of the
        uniform candidates on an x-y grid (top row) and the density of the new collocation
        points relative to a uniform set (middle row; 1 = as many points as a uniform set would
        have there). Bottom: distribution of the new points over time. Dashed lines: interfaces.
        """
        cand, mag, new = candidates.cpu().numpy(), magnitude.cpu().numpy(), new_points.cpu().numpy()
        x_lo, x_hi = float(self.domain_extrema[1, 0]), float(self.domain_extrema[1, 1])
        y_lo, y_hi = self.y_bounds
        edges = [0.0, source_t_max] + [float(v) for v in np.linspace(source_t_max, self.tmax, 6)[1:]]
        interfaces = np.linspace(y_lo, y_hi, self.simulation.get_number_of_layers() + 1)[1:-1]
        box = [[x_lo, x_hi], [y_lo, y_hi]]
        residual_maps, density_maps = [], []
        for t_lo, t_hi in zip(edges[:-1], edges[1:]):
            in_c = (cand[:, 0] >= t_lo) & (cand[:, 0] <= t_hi)
            in_n = (new[:, 0] >= t_lo) & (new[:, 0] <= t_hi)
            total, _, _ = np.histogram2d(cand[in_c, 1], cand[in_c, 2], bins=rad_map_bins, range=box, weights=mag[in_c])
            count, _, _ = np.histogram2d(cand[in_c, 1], cand[in_c, 2], bins=rad_map_bins, range=box)
            points, _, _ = np.histogram2d(new[in_n, 1], new[in_n, 2], bins=rad_map_bins, range=box)
            residual_maps.append(total / np.maximum(count, 1))
            density_maps.append(points / (len(new) * (t_hi - t_lo) / self.tmax / rad_map_bins ** 2))
        r_hi = max(float(m.max()) for m in residual_maps)
        r_lo = max(r_hi * 1e-4, min(float(m[m > 0].min()) for m in residual_maps))
        d_hi = max(2.0, float(np.percentile(np.concatenate([m.ravel() for m in density_maps]), 99.5)))

        fig = plt.figure(figsize=(3.0 * (len(edges) - 1), 8.0))
        grid = fig.add_gridspec(3, len(edges) - 1, height_ratios=[1, 1, 0.55])
        rows = ([], [])
        for k, (t_lo, t_hi) in enumerate(zip(edges[:-1], edges[1:])):
            for row, (field, kwargs) in enumerate(((residual_maps[k], dict(norm=LogNorm(vmin=r_lo, vmax=r_hi), cmap='magma')),
                                                   (density_maps[k], dict(vmin=0.0, vmax=d_hi, cmap='viridis')))):
                ax = fig.add_subplot(grid[row, k])
                image = ax.imshow(np.maximum(field, r_lo).T if row == 0 else field.T, origin='lower',
                                  extent=[x_lo, x_hi, y_lo, y_hi], aspect='equal', **kwargs)
                for y_c in interfaces:
                    ax.axhline(y_c, color='white', lw=0.4, ls='--', alpha=0.7)
                ax.set_title(f"{t_lo:.2f} <= t <= {t_hi:.2f}" if row == 0 else f"{int(((new[:, 0] >= t_lo) & (new[:, 0] <= t_hi)).sum())} points")
                ax.set_xticks([]); ax.set_yticks([])
                if k == 0:
                    ax.set_ylabel("mean residual magnitude" if row == 0 else "new points / uniform")
                rows[row].append(ax)
                if k == len(edges) - 2:
                    fig.colorbar(image, ax=rows[row], fraction=0.012, pad=0.01)
        ax = fig.add_subplot(grid[2, :])
        ax.hist(new[:, 0], bins=100, range=(0.0, self.tmax), density=True, color='tab:blue', label='new collocation points')
        ax.axhline(1.0 / self.tmax, color='tab:orange', lw=1.2, label='uniform')
        ax.set_xlabel("t"); ax.set_ylabel("density of points"); ax.set_xlim(0.0, self.tmax); ax.legend(frameon=False)
        fig.suptitle(f"Resampling at epoch {epoch}: residual of {len(cand)} uniform candidates and the {len(new)} points drawn from them")
        directory = os.path.join(self.PATH, 'plots', 'rad')
        os.makedirs(directory, exist_ok=True)
        fig.savefig(os.path.join(directory, f'rad_epoch_{epoch}.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Residual and point maps: {os.path.join(directory, f'rad_epoch_{epoch}.png')}")

    def soap_state_path(self):
        return os.path.join(self.PATH, 'model', 'soap_state_latest.pth')

    def save_soap_state(self, epoch, points, shuffle):
        """Everything besides the network that a continued run needs, written next to the checkpoints."""
        state = {'epoch': epoch, 'optimizer': self.soap.state_dict(), 'scheduler': self.soap_scheduler.state_dict(),
                 'points': points.cpu(), 'shuffle': shuffle.get_state(), 'batches': soap_batches_per_epoch}
        torch.save(state, self.soap_state_path() + '.tmp')
        os.replace(self.soap_state_path() + '.tmp', self.soap_state_path())

    def load_soap_state(self, epoch, shuffle):
        """Restore optimizer, schedule and shuffling of a run that stopped after `epoch` epochs; returns its collocation set."""
        if not os.path.exists(self.soap_state_path()):
            raise RuntimeError(f"No SOAP state file {self.soap_state_path()}: this window cannot be continued.")
        state = torch.load(self.soap_state_path(), map_location=self.device)
        if state['epoch'] != epoch:
            raise RuntimeError(f"SOAP state is from epoch {state['epoch']}, the newest checkpoint from epoch {epoch}.")
        self.soap.load_state_dict(state['optimizer'])
        self.soap_scheduler.load_state_dict(state['scheduler'])
        shuffle.set_state(state['shuffle'].cpu())
        return state['points'].to(self.device)

    def monitor_soap(self, epoch, points, test_points, shuffle):
        """Error against the reference, PDE term on the current and on the fixed test points, checkpoint."""
        self.iter = epoch  # names the plot folder and the checkpoint, like the evaluation count of the L-BFGS runs
        print(f"Monitoring after epoch {epoch}: PDE term on the current collocation points {self.pde_term(points):.5f}, "
              f"on {len(test_points)} fixed uniform test points {self.pde_term(test_points):.5f}")
        timesteps = self.get_time_steps(self.tmax)
        wavefields = self.generate_pinn_wavefields(timesteps, device=self.device)
        self.compare_wavefields(wavefields, ux_wavefields, uy_wavefields, u_wavefields, timesteps)
        self.compare_wavefields_cross_section(wavefields, ux_wavefields, uy_wavefields, u_wavefields, timesteps)
        self.save_model(epoch + 1)
        self.save_soap_state(epoch, points, shuffle)

    def batches_that_fit(self, points, boundary_points):
        """
        Smallest number of equal batches per epoch for which the loss and its gradient of one batch
        stay within soap_memory_fraction of the memory this process can use on the card (what is
        free now plus what it already holds). The margin leaves room for the memory PyTorch
        reserves beyond what it allocates.
        """
        import gc
        if not torch.cuda.is_available():
            return 1  # a test on a CPU: one batch
        torch.cuda.empty_cache()
        free, _ = torch.cuda.mem_get_info()
        budget = soap_memory_fraction * (free + torch.cuda.memory_reserved())
        for n in range(1, 257):
            size, size_b = -(-len(points) // n), -(-len(boundary_points) // n)
            torch.cuda.reset_peak_memory_stats()
            outcome = "fits"
            try:
                self.soap.zero_grad(set_to_none=True)
                loss = self.minibatch_loss(points[:size].clone().requires_grad_(True),
                                           boundary_points[:size_b].clone().requires_grad_(True),
                                           self.boundary_normals[:size_b])[0]
                loss.backward()  # trial
                if torch.cuda.max_memory_allocated() > budget:
                    outcome = "too large"
            except RuntimeError as error:
                # Some CUDA libraries report a failed allocation as a plain RuntimeError.
                if not isinstance(error, torch.cuda.OutOfMemoryError) and "out of memory" not in str(error).lower():
                    raise
                outcome = "out of memory"
            peak = torch.cuda.max_memory_allocated()
            loss = None
            self.soap.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            print(f"Batch test: {n} batch(es) of {size} collocation points: peak {peak / 1e9:.1f} GB, "
                  f"allowed {budget / 1e9:.1f} GB -> {outcome}")
            if outcome == "fits":
                torch.cuda.reset_peak_memory_stats()
                return n
        raise RuntimeError("No batch size fits on the card.")

    def fit_soap(self):
        """
        SOAP on mini-batches, with the loop of jaxpi: one optimizer step and one step of the
        learning-rate schedule per batch. One epoch is one pass over the collocation set and
        the boundary points in soap_batches_per_epoch shuffled batches. Every rad_interval
        epochs the collocation set is replaced by residual-based resampling.

        Returns:
        list: mean batch loss of each epoch.
        """
        global soap_batches_per_epoch, soap_warmup_steps, soap_decay_steps
        points = next(iter(self.training_set_int))[0].detach()
        boundary_points = self.boundary_points.detach()
        shuffle = torch.Generator().manual_seed(soap_shuffle_seed)
        start_epoch = self.iter
        if start_epoch > 0:
            if not os.path.exists(self.soap_state_path()):
                raise RuntimeError(f"No SOAP state file {self.soap_state_path()}: this window cannot be continued.")
            soap_batches_per_epoch = torch.load(self.soap_state_path(), map_location='cpu')['batches']
        elif soap_batches_per_epoch == 0:
            soap_batches_per_epoch = self.batches_that_fit(points, boundary_points)
        soap_warmup_steps = soap_warmup_epochs * soap_batches_per_epoch
        soap_decay_steps = soap_decay_epochs * soap_batches_per_epoch
        self.soap_scheduler = torch.optim.lr_scheduler.LambdaLR(self.soap, lr_lambda=soap_lr_factor)
        if start_epoch > 0:
            points = self.load_soap_state(start_epoch, shuffle)
            print(f"Continuing SOAP after epoch {start_epoch} with {len(points)} collocation points")
        n_points, n_boundary = len(points), len(boundary_points)
        batch = -(-n_points // soap_batches_per_epoch)
        batch_b = -(-n_boundary // soap_batches_per_epoch)
        test_points = self.uniform_points(soap_test_points, torch.Generator().manual_seed(soap_test_seed))
        print(f"SOAP: {soap_epochs} epochs of {soap_batches_per_epoch} steps, {batch} collocation and {batch_b} boundary points per step")

        history = []
        start_time = time.time()
        for epoch in range(start_epoch, soap_epochs):
            if epoch % soap_monitor_every == 0 and (epoch == 0 or epoch != start_epoch):
                self.monitor_soap(epoch, points, test_points, shuffle)
            if epoch > 0 and epoch % rad_interval == 0:
                points = self.resample_points(epoch, n_points)

            order = torch.randperm(n_points, generator=shuffle).to(self.device)
            order_b = torch.randperm(n_boundary, generator=shuffle).to(self.device)
            sums = torch.zeros(3, device=self.device)
            for k in range(soap_batches_per_epoch):
                input_s = points[order[k * batch:(k + 1) * batch]].requires_grad_(True)
                index_b = order_b[k * batch_b:(k + 1) * batch_b]
                input_b = boundary_points[index_b].requires_grad_(True)
                self.soap.zero_grad(set_to_none=True)
                loss, loss_pde, loss_boundary = self.minibatch_loss(input_s, input_b, self.boundary_normals[index_b])
                loss.backward()
                self.soap.step()
                self.soap_scheduler.step()
                sums += torch.stack((loss.detach(), loss_pde, loss_boundary))
            mean_loss, mean_pde, mean_boundary = (sums / soap_batches_per_epoch).tolist()
            history.append(mean_loss)
            if not np.isfinite(mean_loss):
                raise RuntimeError(f"Loss is not finite in epoch {epoch}.")
            if epoch % soap_print_every == 0:
                print(f'Epoch {epoch} - Total loss: {round(mean_loss, 5)}')
                print(f'  PDE term: {mean_pde:.5f}  boundary term (unweighted): {mean_boundary:.5f}')
                print(f'  Means over the {soap_batches_per_epoch} batches of the epoch. Learning rate '
                      f'{self.soap.param_groups[0]["lr"]:.3e}, {time.time() - start_time:.0f} s in this job, '
                      f'GPU memory peak so far: {(torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0) / 1e9:.1f} GB')

        self.monitor_soap(soap_epochs, points, test_points, shuffle)
        return history

    def build_gt_aligned_timesteps(self):
        """
        Build an evaluation time grid that lands exactly on ground-truth snapshot times,
        covering [0, min(tmax, GT_MAX_TIME)].

        Returns:
            (torch.Tensor, list): times t_k = k * dt_frame, and the matching ground-truth
            |u| snapshots, so that element k of each corresponds to the same instant.
        """
        n_frames = min(int(round(self.tmax / dt_frame)), GT_MAX_INDEX) + 1
        timesteps = torch.tensor(np.arange(n_frames) * dt_frame, dtype=torch.float32)
        return timesteps, u_wavefields[:n_frames]

    def validate_warm_start(self, tol=0.25):
        """
        Verify that a freshly loaded checkpoint actually reproduces the wavefield it was
        trained to produce.

        This exists because load_state_dict() cannot detect the failure mode that broke
        the previous run: the checkpoint was trained with Sine activations but loaded
        into a Tanh network. Parameter shapes are identical, so every key "matched
        successfully" while the network computed an entirely different function.

        Compares |u| at t=1.0 against the corresponding ground-truth frame. A correctly
        warm-started tmax=1 model scores ~0.14; the broken Tanh load scored ~0.61.
        """
        t_check = 1.0
        if self.tmax < t_check:
            # t=1.0 lies outside this leg's training window [0, self.tmax]; the network
            # was never trained there, so the check would be meaningless.
            print(f"validate_warm_start: t=1.0 outside this leg's domain "
                  f"[0, {self.tmax}], skipping check.")
            return None
        idx = int(round(t_check / dt_frame))
        if idx > GT_MAX_INDEX:
            print("validate_warm_start: t=1.0 beyond ground truth, skipping check.")
            return None

        wavefields = self.generate_pinn_wavefields(torch.tensor([t_check]), device=self.device)
        ux, uy = wavefields[0]
        u_pinn = np.sqrt(ux ** 2 + uy ** 2)
        u_gt = u_wavefields[idx]

        rel_l2 = np.linalg.norm(u_pinn - u_gt) / np.linalg.norm(u_gt)
        print(f"Warm-start check: rel-L2 of |u| at t={t_check} vs frame {idx} = {rel_l2:.4f}")

        if rel_l2 > tol:
            raise RuntimeError(
                f"Warm start looks broken: rel-L2 {rel_l2:.4f} > tol {tol}. "
                f"The checkpoint loaded but does not reproduce the trained solution. "
                f"Check that the activation ('{getattr(self.network_u, 'act_name', '?')}') "
                f"and t1 ({self.t1}) match the run that produced the checkpoint."
            )
        return rel_l2

    def load_model(self, checkpoint_path):
        """
        Load the state of the neural network from a checkpoint.
        
        Parameters:
        checkpoint_path (str): Path to the saved checkpoint.
        
        Returns:
        int: The epoch to resume training from.
        """
        # map_location keeps this loadable on a CPU-only node (the checkpoint stores
        # CUDA tensors, which otherwise raises on deserialize).
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.network_u.load_state_dict(checkpoint['model_u_state_dict'])
        
        print(f"Model loaded from {checkpoint_path}, resuming from epoch {checkpoint['epoch']}.")
        
        return checkpoint['epoch']


    def save_model(self, epoch):
        """
        Save the state of the neural network and scheduler (if any).

        Parameters:
        epoch (int): The current epoch number.
        """
        checkpoint = {
            'epoch': epoch,
            'model_u_state_dict': self.network_u.state_dict(),
        }
        
        # If using a scheduler, include its state:
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        model_dir = os.path.join(self.PATH, 'model')
        os.makedirs(model_dir, exist_ok=True)

        checkpoint_path = os.path.join(model_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        print(f"Model and scheduler state saved at epoch {epoch}.")

        self.checkpoint_path_latest = checkpoint_path

        return checkpoint_path


    def fit(self, num_epochs, optimizer, verbose=True):
        """
        Train the model for a given number of epochs.

        Parameters:
        num_epochs (int): The number of epochs to train for.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        verbose (bool): Flag for printing epoch information.

        Returns:
        list: History of loss values.
        """
        history = []

        inp_train_int = next(iter(self.training_set_int))[0]
        inp_train_int.requires_grad = True
        print(len(inp_train_int))
      

        for epoch in range(num_epochs):
            if verbose and (epoch % 500 == 0):
                print(f"################################ {epoch} ################################")

            def closure():
                optimizer.zero_grad()
                # compute_loss evaluates the loss in passes and accumulates the gradient itself.
                loss = self.compute_loss(inp_train_int, verbose=verbose)

                # Save the model every 1000 epochs
                if (epoch + 1) % 1000 == 0:
                    self.save_model(epoch + 1)

                return loss


            optimizer.step(closure=closure)

            
            if (epoch + 1) % self.decay_steps == 0:
                self.scheduler.step()
                print(f"Step {epoch + 1}, Learning Rate: {self.scheduler.get_last_lr()[0]}")

        # print('Final Loss:', history[-1])
        
        return history
        

    def train(self, optimizer, num_epochs=5, resume_from=None, verbose=True):
        """
        Train the model using Adam and L-BFGS optimizers, with the ability to resume from a checkpoint.

        Parameters:
        optimizer (str): Optimizer type ('adam' or 'lbfgs').
        num_epochs (int): Number of epochs for training with the L-BFGS optimizer.
        resume_from (str): Path to the checkpoint to resume from (if any).
        verbose (bool): Flag for verbosity during training.

        Returns:
        list: Training loss history.
        """
        print("Training started...")

        if resume_from:
            start_epoch = self.load_model(resume_from)
            # Warn, don't abort: in a from-scratch curriculum the previous leg was
            # trained on a shorter window, so a large error at t=1.0 is expected.
            try:
                self.validate_warm_start()
            except RuntimeError as e:
                print(f"WARNING: {e}\nContinuing anyway (non-fatal for this run).")
        else:
            start_epoch = 0
        
        optimizer_dispatcher = {
            'adam': self.adam,
            'lbfgs': self.lbfgs
        }

        # Evaluate on the ground-truth frame grid itself, so that PINN sample k and
        # ground-truth frame k refer to the same physical time by construction.
        # (compute_energy_difference_over_time pairs them positionally via zip.)
        timesteps, gt_u = self.build_gt_aligned_timesteps()
        wavefields = self.generate_pinn_wavefields(timesteps, device=self.device)
        save_path = f'{self.PATH}energy_difference_{n_layers}_{n_neurons}_mapping_{mapping_size}_0.pkl'
        energy_difference = self.compute_energy_difference_over_time(wavefields, gt_u, timesteps,save_path,verbose=False)
        self.plot_energy_difference(timesteps,  energy_difference, save_path=f'{self.PATH}before.png')


        if optimizer == 'adam':
            # Train with Adam optimizer
            self.training_loss.extend(self.fit(num_epochs=num_epochs, optimizer=self.adam, verbose=verbose))
            print("Adam Optimization Completed...")
        elif optimizer == 'lbfgs':
            # Train with L-BFGS optimizer
            lbfgs_loss = self.fit(num_epochs=num_epochs, optimizer=self.lbfgs, verbose=verbose)
            self.training_loss.extend(lbfgs_loss)
        elif optimizer == 'soap':
            self.training_loss.extend(self.fit_soap())

        
        wavefields = self.generate_pinn_wavefields(timesteps, device=self.device)
        # self.compare_wavefields(wavefields, ux_wavefields, uy_wavefields,u_wavefields, timesteps)
        save_path = f'{self.PATH}energy_difference_{self.tmax}s.pkl'
        energy_difference_new = self.compute_energy_difference_over_time(wavefields, gt_u, timesteps,save_path,verbose=True)
        self.plot_energy_difference(timesteps,  energy_difference_new, save_path=f'{self.PATH}final.png')

        print('Plotting Loss...')

        return self.training_loss

    
    def predict(self, input_tensor):
        """
        Predicts the displacement field (Ux, Uy) using both neural networks (for x and y directions).

        Args:
            input_tensor (torch.Tensor): The input tensor containing spatial and temporal coordinates.

        Returns:
            torch.Tensor: A tensor with two columns: predicted displacements in x and y directions.
        """
        # Run the input through both networks
        U_pred = self.initial_condition_ansatz(input_tensor)  # Displacement prediction in the x-direction (shape: [N, 1])
        # Uy_pred = self.network_uy(input_tensor)  # Displacement prediction in the y-direction (shape: [N, 1])

        #  # Reshape to [N, 1] if necessary
        # if Ux_pred.dim() == 1:
        #     Ux_pred = Ux_pred.unsqueeze(1)
        # if Uy_pred.dim() == 1:
        #     Uy_pred = Uy_pred.unsqueeze(1)

        # # Combine the two predictions into a single tensor (shape: [N, 2])
        # U_pred = torch.cat((Ux_pred, Uy_pred), dim=1)

        return U_pred


    def generate_pinn_wavefields(self, time_steps, numpoints_sqrt=512, device='cpu'):
        """
        Generate PINN wavefields for the specified time steps.

        Parameters:
            pinn: The PINN model to use for predictions.
            time_steps (list): List of time steps to generate predictions for.
            numpoints_sqrt (int): Number of points per spatial dimension (256x256 grid by default).
            device (str): Device to perform computations ('cpu' or 'cuda').

        Returns:
            list: List of wavefields as (ux, uy) tuples for each time step.
        """
        wavefields = []
        soboleng = torch.quasirandom.SobolEngine(dimension=3)
        
        # Generate grid points
        grid_x, grid_y = torch.meshgrid(
            torch.linspace(-1.5, 1.5, numpoints_sqrt),
            torch.linspace(-1.5, 1.5, numpoints_sqrt)
        )
        grid_x = grid_x.reshape(-1).to(device)
        grid_y = grid_y.reshape(-1).to(device)
        inputs = soboleng.draw(int(numpoints_sqrt ** 2)).to(self.device)

        # Loop through each specified time step and generate wavefields
        for t in time_steps:
            
            inputs[:, 1] = grid_x
            inputs[:, 2] = grid_y
            inputs[:, 0] = t

            # Predict using the PINN model
            U = self.predict(inputs)
            ux = U[:, 0].detach().cpu().numpy().reshape(numpoints_sqrt, numpoints_sqrt)
            uy = U[:, 1].detach().cpu().numpy().reshape(numpoints_sqrt, numpoints_sqrt)

            wavefields.append((ux, uy))
        
        return wavefields
    
    def compare_wavefields(self, wavefields, ground_truth_x, ground_truth_y, ground_truth_u, time_steps):
        """
        Compare PINN wavefields with ground truth data for Ux, Uy, and U at the specified time steps.

        Parameters:
            wavefields (list): List of wavefields (ux, uy) for each time step.
            ground_truth_x (list): Ground truth data for ux.
            ground_truth_y (list): Ground truth data for uy.
            ground_truth_u (list): Ground truth data for u = sqrt(ux^2 + uy^2).
            time_steps (list): List of time steps corresponding to the wavefields.
        """
        save_directory = f'{self.PATH}/plots/{self.iter}/'
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        total_l2_ux = 0
        total_l2_uy = 0
        total_l2_u = 0
        num_time_steps = len(time_steps)
        numpoints_sqrt = 512
        RMSE_total = 0
        RRMSE_total = 0
        l2_relative_error_u_running = 0 
        total_points = 0

        # Set font sizes to 8pt as required by Springer Nature
        title_fontsize = 8
        label_fontsize = 8

        # Define the extent for all plots
        extent_val = [-1.5, 1.5, -1.5, 1.5]

        for i, t in enumerate(time_steps):
            # Create a subdirectory for each time step
            plot_directory = os.path.join(save_directory, f"plot_{t}")
            if not os.path.exists(plot_directory):
                os.makedirs(plot_directory)

            # Extract wavefields for the current time step
            ux_pinn, uy_pinn = wavefields[i]
            u_pinn = np.sqrt(ux_pinn ** 2 + uy_pinn ** 2)
            gt_idx = int(round(float(t) / dt_frame))

            if gt_idx > GT_MAX_INDEX:
                # No ground truth beyond t = GT_MAX_TIME (2.20). Plot the PINN field on
                # its own so the extended-horizon legs remain inspectable, and skip the
                # difference panels and error metrics.
                self._plot_pinn_only(u_pinn, t, plot_directory, extent_val, title_fontsize)
                continue

            ux_ground_truth = ground_truth_x[gt_idx]
            uy_ground_truth = ground_truth_y[gt_idx]
            u_ground_truth = ground_truth_u[gt_idx]

            # Calculate L2 errors
            l2_relative_error_ux = np.linalg.norm(uy_pinn - ux_ground_truth) / np.linalg.norm(ux_ground_truth) * 100
            l2_relative_error_uy = np.linalg.norm(ux_pinn - uy_ground_truth) / np.linalg.norm(uy_ground_truth) * 100
            l2_relative_error_u = np.linalg.norm(u_pinn - u_ground_truth) / np.linalg.norm(u_ground_truth) * 100

            # Accumulate errors
            total_l2_ux += l2_relative_error_ux
            total_l2_uy += l2_relative_error_uy
            total_l2_u += l2_relative_error_u

            # Plot configurations
            cmap = 'seismic'
            dpi = 600

            # Create figure for Ux and Uy comparisons
            fig, axes = plt.subplots(2, 3 , gridspec_kw={'wspace': 0.1, 'hspace': 0.1})
            
            # Labels for columns
            for ax, label in zip(axes[0], ["PINN", "Numerical", "Difference"]):
                ax.set_title(label, fontsize=title_fontsize)

            # Remove x and y ticks
            for ax in axes.flatten():
                ax.set_xticks([])
                ax.set_yticks([])

            # Row 1: Ux comparison
            im1 = axes[0, 0].imshow(ux_pinn.T, vmin=-0.2, vmax=0.2, cmap=cmap, extent=extent_val)
            im2 = axes[0, 1].imshow(uy_ground_truth.T, vmin=-0.2, vmax=0.2, cmap=cmap, extent=extent_val)
            diffx = ux_pinn - uy_ground_truth
            diffx[0:9, :] = 0
            diffx[:, 0:9] = 0
            diffx[:, numpoints_sqrt-9:numpoints_sqrt] = 0
            diffx[numpoints_sqrt-9:numpoints_sqrt, :] = 0
            im3 = axes[0, 2].imshow(diffx.T, vmin=-0.2, vmax=0.2, cmap=cmap, extent=extent_val)
        
            # Row 2: Uy comparison
            im4 = axes[1, 0].imshow(uy_pinn.T, vmin=-0.2, vmax=0.2, cmap=cmap, extent=extent_val)
            im5 = axes[1, 1].imshow(ux_ground_truth.T, vmin=-0.2, vmax=0.2, cmap=cmap, extent=extent_val)
            diffy = uy_pinn - ux_ground_truth
            diffy[0:9, :] = 0
            diffy[:, 0:9] = 0
            diffy[:, numpoints_sqrt-9:numpoints_sqrt]  = 0
            diffy[numpoints_sqrt-9:numpoints_sqrt, :] = 0
            im6 = axes[1, 2].imshow(diffy.T, vmin=-0.2, vmax=0.2, cmap=cmap, extent=extent_val)
            
            # Add single colorbar at the bottom
            cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.03])
            cbar = fig.colorbar(im1, cax=cbar_ax, orientation='horizontal')

            # plt.tight_layout()
            plt.savefig(os.path.join(plot_directory, "Wavefield_Comparison_Ux_Uy.png"), dpi=dpi)
            plt.close()

            # Separate figure for U comparisons
            fig, axes = plt.subplots(1, 3, gridspec_kw={'wspace': 0.1})
            
            # Labels for columns
            for ax, label in zip(axes, ["PINN", "Numerical", "Difference"]):
                ax.set_title(label, fontsize=title_fontsize)

            # Remove x and y ticks
            for ax in axes:
                ax.set_xticks([])
                ax.set_yticks([])

            # U comparison
            im7 = axes[0].imshow(u_pinn, vmin=0, vmax=0.2, cmap=cmap, extent=extent_val)
            im8 = axes[1].imshow(u_ground_truth, vmin=0, vmax=0.2, cmap=cmap, extent=extent_val)
            diffu = u_pinn - u_ground_truth
            diffu[0:9, :] = 0
            diffu[:, 0:9] = 0
            diffu[:, numpoints_sqrt-9:numpoints_sqrt]  = 0
            diffu[numpoints_sqrt-9:numpoints_sqrt, :] = 0
            
            mse_u = np.mean(diffu**2)
            rmse_u = np.sqrt(mse_u)
            ground_truth_rms = np.sqrt(np.mean(u_ground_truth**2))
            rrmse_u = rmse_u / ground_truth_rms
            
            # Accumulate RMSE and RRMSE
            RMSE_total += rmse_u
            RRMSE_total += rrmse_u
            total_points += 1

            # Optional: Compute Relative L2 Error
            l2_relative_error_u = np.linalg.norm(diffu) / np.linalg.norm(u_ground_truth)
            l2_relative_error_u_running += l2_relative_error_u
            
            im9 = axes[2].imshow(diffu, vmin=0, vmax=0.2, cmap=cmap, extent=extent_val)
            
            # Add relative L2 error as annotation
            l2_error_text = f"L2 Error: {l2_relative_error_u:.2f}"
            axes[0].annotate(l2_error_text, xy=(0.05, 0.95), xycoords='axes fraction', fontsize=label_fontsize, color='white', ha='left', va='top', bbox=dict(facecolor='black', alpha=0.7, boxstyle="round,pad=0.5"))

            # Add single colorbar at the bottom
            cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.03])
            cbar = fig.colorbar(im7, cax=cbar_ax, orientation='horizontal')

            # plt.tight_layout()
            plt.savefig(os.path.join(plot_directory, "Wavefield_Comparison_U.png"), dpi=dpi)
            plt.close()

        # Average over the time steps that actually had ground truth
        if total_points == 0:
            print(f"No ground truth available for any requested time step "
                  f"(all beyond t={GT_MAX_TIME:.2f}); skipped error metrics.")
        else:
            avg_RMSE = RMSE_total / total_points
            avg_RRMSE = RRMSE_total / total_points
            avg_l2_relative_error_u = l2_relative_error_u_running / total_points

            print(f"Average RMSE: {avg_RMSE:.4f}")
            print(f"Average RRMSE: {avg_RRMSE:.4f}")
            print(f"Average Relative L2: {avg_l2_relative_error_u:.4f} "
                  f"(over {total_points}/{len(time_steps)} time steps with ground truth)")

    def _plot_pinn_only(self, u_pinn, t, plot_directory, extent_val, title_fontsize):
        """
        Plot the PINN |u| field alone, for time steps past the end of the ground-truth
        record (t > GT_MAX_TIME). Keeps the extended-horizon legs visually inspectable
        even though no quantitative comparison is possible.
        """
        fig, ax = plt.subplots(1, 1)
        im = ax.imshow(u_pinn, vmin=0, vmax=0.2, cmap='seismic', extent=extent_val)
        ax.set_title(f"PINN $|u|$ at t={float(t):.2f} (no ground truth)",
                     fontsize=title_fontsize)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.046, pad=0.04)
        plt.savefig(os.path.join(plot_directory, "Wavefield_PINN_only_U.png"), dpi=600)
        plt.close()

    def compare_wavefields_cross_section(self, wavefields, ground_truth_x, ground_truth_y, ground_truth_u, time_steps):
        """
        Compare PINN wavefields with ground truth data for Ux, Uy, and U at the specified time steps.

        Parameters:
            wavefields (list): List of wavefields (ux, uy) for each time step.
            ground_truth_x (list): Ground truth data for ux.
            ground_truth_y (list): Ground truth data for uy.
            ground_truth_u (list): Ground truth data for u = sqrt(ux^2 + uy^2).
            time_steps (list): List of time steps corresponding to the wavefields.
        """
        save_directory = f'{self.PATH}/plots/{self.iter}/'
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Define maximum figure size in inches (180mm x 215mm)
        # max_figsize = (180 / 25.4, 215 / 25.4)  # (7.09, 8.46)

        # Loop over time steps
        for i, t in enumerate(time_steps):
            # Create a subdirectory for each time step
            plot_directory = os.path.join(save_directory, f"plot_{t}")
            if not os.path.exists(plot_directory):
                os.makedirs(plot_directory)

            # Extract wavefields for the current time step
            ux_pinn, uy_pinn = wavefields[i]
            u_pinn = np.sqrt(ux_pinn ** 2 + uy_pinn ** 2)
            gt_idx = int(round(float(t) / dt_frame))

            if gt_idx > GT_MAX_INDEX:
                # No ground truth past t = GT_MAX_TIME; nothing to cross-section against.
                continue

            ux_ground_truth = ground_truth_x[gt_idx]
            uy_ground_truth = ground_truth_y[gt_idx]
            u_ground_truth = ground_truth_u[gt_idx]

            # Determine the x-axis coordinate assuming the cross-section length is the number of columns/rows
            # and we want to span from -1.5 to 1.5.
            num_cols = ux_pinn.shape[1]  # assuming square wavefields, shape: (npoints, npoints)
            x_axis = np.linspace(-1.5, 1.5, num=num_cols)

            # -------------------------
            # Cross-section of Ux
            # -------------------------
            # For Ux, the original code plots: 
            #    PINN: uy_pinn[256, :], Devito: ux_ground_truth[256, :]
            fig = plt.figure(dpi=600)
            plt.plot(x_axis, ux_pinn[:, 256], label='PINN', color='blue', linewidth=1)
            plt.plot(x_axis, uy_ground_truth[:, 256], label='Numerical', color='orange', linewidth=1)
            plt.title("$U_x$", fontsize=8)
            plt.xlabel("$x$", fontsize=8)
            # plt.ylabel("Amplitude", fontsize=8)
            plt.xlim(-1.5, 1.5)
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8)
            plt.legend(fontsize=8, loc="upper right")
            plt.tight_layout()
            # Save the Ux cross-section plot separately
            plt.savefig(os.path.join(plot_directory, "Cross_section_Ux_Comparison.png"), dpi=600)
            plt.close()

            # -------------------------
            # Cross-section of Uy
            # -------------------------
            # For Uy, the original code plots:
            #    PINN: ux_pinn[:, 256], Devito: uy_ground_truth[:, 256]
            fig = plt.figure(dpi=600)
            plt.plot(x_axis, uy_pinn[256, :], label='PINN', color='blue', linewidth=1)
            plt.plot(x_axis, ux_ground_truth[256, :], label='Numerical', color='orange', linewidth=1)
            plt.title("$U_y$", fontsize=8)
            plt.xlabel("$y$", fontsize=8)
            # plt.ylabel("Amplitude", fontsize=8)
            plt.xlim(-1.5, 1.5)
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8)
            plt.legend(fontsize=8, loc="upper right")
            plt.tight_layout()
            # Save the Uy cross-section plot separately
            plt.savefig(os.path.join(plot_directory, "Cross_section_Uy_Comparison.png"), dpi=600)
            plt.close()


    def plot_and_save_velocity_model(self, lambda_values, mu_values, rho_values, training_set,save_path="velocity_model"):
        """
        Plots and saves the velocity model for lambda, mu, and rho.
        
        Args:
            lambda_values (torch.Tensor): 2D tensor of lambda values (on device).
            mu_values (torch.Tensor): 2D tensor of mu values (on device).
            rho_values (torch.Tensor): 2D tensor of rho values (on device).
            save_path (str): Base path for saving the plots.
        """
        # Detach and convert to numpy
        # print(lambda_values)
        x = training_set[:, 1].detach().cpu().numpy()
        y = training_set[:, 2].detach().cpu().numpy()

        lambda_values = lambda_values.detach().cpu().numpy()
        mu_values = mu_values.detach().cpu().numpy()
        rho_values = rho_values.detach().cpu().numpy()

        # Create figure for lambda
        plt.figure(figsize=(8, 6))
        plt.scatter(x, y, c=lambda_values, s=10, alpha=0.7, edgecolors='k', label="Lambda Values")
        plt.colorbar(label=r"$\lambda$ Values")
        plt.title("Velocity Model: Lambda ($\lambda$)")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        # plt.savefig(f"{save_path}_lambda.png", dpi=300)
        plt.close()  # Close the figure to free memory

        # Create figure for mu
        plt.figure(figsize=(8, 6))
        plt.scatter(x, y, c=mu_values, s=10, alpha=0.7, edgecolors='k', label="Mu Values")
        plt.colorbar(label=r"$\mu$ Values")
        plt.title("Velocity Model: Mu ($\mu$)")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        # plt.savefig(f"{save_path}_mu.png", dpi=300)
        plt.close()

        # Create figure for rho
        plt.figure(figsize=(8, 6))
        plt.scatter(x, y, c=rho_values, s=10, alpha=0.7, edgecolors='k', label="Rho")
        plt.colorbar(label="Rho")
        plt.title("Velocity Model: Rho")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        # plt.savefig(f"{save_path}_rho.png", dpi=300)
        plt.close()


    def compute_energy_difference_over_time(self, pinn_wavefields, ground_truth_wavefields, time_steps, save_path, verbose=False):
        """
        Compute the normalized energy difference between PINN and ground truth solutions over time,
        considering the squared norm of Ux and Uy fields.

        Parameters:
            pinn_wavefields (list): List of wavefields (each element has Ux and Uy fields) from the PINN at each time step.
            ground_truth_wavefields (list): List of wavefields (each element has Ux and Uy fields) from the ground truth at each time step.
            time_steps (list): List of time steps corresponding to the wavefields.

        Returns:
            energy_diff (list): Normalized energy difference for each time step.
        """
        energy_diff = []
        numpoints_sqrt = 512

        for t, (pinn_snapshot, gt_snapshot) in enumerate(zip(pinn_wavefields, ground_truth_wavefields)):
            # Extract Ux and Uy for PINN and ground truth
            ux_pinn, uy_pinn = pinn_snapshot  # PINN Ux and Uy fields
    #         ux_gt, uy_gt = gt_snapshot  # Ground truth Ux and Uy fields

            # Compute squared norm of Ux and Uy for both PINN and ground truth
            pinn_norm = np.sqrt(ux_pinn**2 + uy_pinn**2)

            diffu = pinn_norm - gt_snapshot
            diffu[0:9, :] = 0
            diffu[:, 0:9] = 0
            diffu[:, numpoints_sqrt-9:numpoints_sqrt]  = 0
            diffu[numpoints_sqrt-9:numpoints_sqrt:, :] = 0
        

            # Numerator: Sum of squared differences
            numerator = np.sum((diffu) ** 2)

            # Denominator: Total energy in the ground truth wavefield
            denominator = np.sum(gt_snapshot ** 2)

            # Avoid division by zero
            if denominator == 0:
                energy = 0
            else:
                energy = numerator / denominator

            energy_diff.append(energy)

            if verbose:
                print(f"Time step {time_steps[t]}: Energy Difference = {energy:.6f}")
        
        relative_l2_errors = [np.sqrt(e) for e in energy_diff]
        avg_relative_l2_error = np.mean(relative_l2_errors)
        print(f"Average Relative L2 Error: {avg_relative_l2_error:.6f}")

        # Save energy differences to a file
        with open(save_path, 'wb') as f:
            pickle.dump(energy_diff, f)
        print(f"Energy differences saved to {save_path}")

        return energy_diff
    
    def plot_energy_difference(self, time_steps, energy_difference, title='Energy Difference Over Time', save_path=None,):
        """
        Plot the normalized energy difference over time.

        Parameters:
            time_steps (list or array): List of time steps.
            energy_difference (list or array): Normalized energy difference for each time step.
            title (str): Title of the plot (default: 'Energy Difference Over Time').
            save_path (str, optional): File path to save the plot as an image. If None, the plot is not saved.

        Returns:
            None
        """
        plt.figure(figsize=(10, 6))
        plt.plot(time_steps, energy_difference, label='Energy Difference', marker='o', linestyle='-', color='b')
        plt.xlabel('Time Steps')
        plt.ylabel('Normalized Energy Difference')
        plt.title(title)
        plt.legend()
        plt.grid()
        
        # Save the plot if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        # Show the plot
        plt.show()



def latest_checkpoint(model_dir):
    """
    Newest checkpoint of a training window.

    Returns:
        (path, evaluations): the checkpoint_epoch_<n>.pth file with the largest n and the number
        of loss evaluations done when it was written (n - 1, see compute_loss), or (None, 0)
        if the folder holds no checkpoint.
    """
    prefix, suffix = 'checkpoint_epoch_', '.pth'
    if not os.path.isdir(model_dir):
        return None, 0
    numbers = [int(name[len(prefix):-len(suffix)]) for name in os.listdir(model_dir)
               if name.startswith(prefix) and name.endswith(suffix) and name[len(prefix):-len(suffix)].isdigit()]
    if not numbers:
        return None, 0
    newest = max(numbers)
    return os.path.join(model_dir, f'{prefix}{newest}{suffix}'), max(newest - 1, 0)


# One training window [0, 1], trained from a randomly initialized network (no curriculum).
time_steps = [1.0]
# No warm start: the first leg trains from a randomly initialized network.
previous_ckpt_path = None

print("################################################################################################")
print("RUN CONFIG")
print(f"  Run name: twolayer_vanilla_5x128_T1.0_soap_rad")
print(f"  Architecture: Vanilla MLP (NeuralNet)")
print(f"  Activation: {activation}")
print(f"  Network size: {n_layers} hidden layers x {n_neurons}")
print(f"  No Fourier embedding. Dense layers: Glorot uniform with gain 5/3, zero bias; sine activation set in global_AF.py")
print(f"  Hard-constraint time scale t1: {t1_value}")
print(f"  Optimizer: SOAP, lr {soap_lr} after a linear warm-up from 0 over {soap_warmup_epochs} epochs, then times "
      f"{soap_decay_rate} every {soap_decay_epochs} epochs; betas {soap_betas}, no weight decay, "
      f"preconditioner updated every {soap_precondition_frequency} step(s)")
print(f"  Epochs: {soap_epochs}, each one pass over the collocation set and the boundary points in equal shuffled "
      f"batches, one optimizer step per batch; batches per epoch: "
      f"{soap_batches_per_epoch if soap_batches_per_epoch else 'the fewest that fit on the card, see the batch test below'}")
print(f"  Resampling (RAD): every {rad_interval} epochs the whole collocation set is drawn anew from "
      f"{rad_candidates} uniform candidates, probability ~ r^{rad_power} / mean + {rad_c}, r clipped at its 99th percentile")
print(f"  Collocation points: {n_int} Sobol for t > {source_t_max} + {n_early_uniform} Sobol for t <= {source_t_max} "
      f"+ {n_source} source region + {n_source_t0} source region at t=0 + {n_t0_uniform} uniform at t=0 "
      f"= {n_int + n_early_uniform + n_source + n_source_t0 + n_t0_uniform}")
print(f"  Source region: t <= {source_t_max}, |x| and |y| <= {source_half_width}; "
      f"points at t=0: |x| and |y| <= {source_t0_half_width}")
print(f"  Residual: rho u_tt - div(sigma), lambda and mu evaluated from the differentiable input")
print(f"  Initial displacement scaled by the analytic peak exp(-1/2)/sigma (no batch maximum)")
print(f"  Absorbing boundary term: {n_boundary_per_edge} Sobol points per edge, weight {bc_weight}; "
      f"loss = mean(PDE residual^2) + weight * mean(boundary residual^2)")
print(f"  Medium: two layers, interface at y = 0 smoothed with tanh(90 y); lambda, mu = 25, 35 for y < 0 and 20, 30 for y > 0; rho = 100")
print(f"  Seeds: script torch.manual_seed(128); network weight initialization seed 42")
print(f"  Training windows [0, T] for T in: {time_steps}")
print(f"  First leg starts from: {previous_ckpt_path}")
print(f"  Reference frames: {input_directory_x}")
print("################################################################################################")

for i, time_max in enumerate(time_steps):
    # Skip a leg that an earlier job already finished, so a resubmitted job continues
    # from the first unfinished leg instead of retraining completed ones.
    stage_done_file = f'experiments/full_time_ext_tmax_{time_max}/STAGE_DONE.json'
    if os.path.exists(stage_done_file):
        with open(stage_done_file) as f:
            previous_ckpt_path = json.load(f)['checkpoint']
        print(f"\n=== Leg [0, {time_max:.2f}] already complete; next leg starts from {previous_ckpt_path} ===")
        continue
    print(f"\n=========================== Training started ===================================================")
    print(f"\n=== Training on [0, {time_max:.2f}] ===")
    
    # 1) Create a new PINN instance for the current time_max
    pinn = Pinns(n_int, t1_value, tmax=time_max)

    # A job that reached the time limit leaves checkpoints but no STAGE_DONE file. Continue such a
    # window from its newest checkpoint with the evaluations that are left. L-BFGS then starts
    # with an empty curvature history.
    window_ckpt_path, evaluations_done = latest_checkpoint(os.path.join(pinn.PATH, 'model'))
    if window_ckpt_path is not None:
        previous_ckpt_path = window_ckpt_path
        pinn.loss_history = []   # compute_loss resets the evaluation counter unless this exists
        pinn.iter = evaluations_done
        print(f"Continuing window [0, {time_max:.2f}] from {window_ckpt_path}: {evaluations_done} evaluations done")
    print(f"SOAP epochs in this job: {soap_epochs - evaluations_done}")

    # 2) Train this PINN, optionally resuming from previous_ckpt_path
    loss_history = pinn.train(
        optimizer="soap",
        num_epochs=soap_epochs,
        resume_from=previous_ckpt_path
    )
    
    # 3) Save model weights so we can resume next iteration
    current_ckpt_path = pinn.save_model(pinn.iter)
    print(f"Saved checkpoint to {current_ckpt_path}")
    with open(stage_done_file, 'w') as f:
        json.dump({'tmax': time_max, 'checkpoint': current_ckpt_path}, f)
    
    # 4) Update 'previous_ckpt_path' for the next iteration
    previous_ckpt_path = current_ckpt_path
    print(f"\n=========================== Training ended ===================================================")



# Routine for curriculum training.


def plot_training_losses(pinn):    
    # Subplot 1: Training Loss
    plt.plot(pinn.collocation_loss, label='Collocation Loss', color='blue', linewidth=2)
    plt.yscale('log')
    plt.title('Training Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (Log Scale)', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(fontsize=10)
    

    # Enhancing the plot aesthetics
    #     plt.style.use('seaborn-darkgrid')
    
    # Adjust layout
    plt.tight_layout()

    plt.savefig(f'loss_{pinn.tmax}_s.png')
    plt.clf()


# # Slope recovery * 10 , 50,000 collocation points , 4000 tb
# plot_training_losses(pinn1)

# plot_training_losses(pinn2)


