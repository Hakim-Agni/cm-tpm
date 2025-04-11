from abc import ABC, abstractmethod
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dist
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from scipy.stats import qmc
import networkx as nx

# TODO:
#   Check: Optimize latent selection (top-K selection)
#   Check: Implement latent optimization for fine-tuning integration points
#   Allow custom Optimizers?
#   Improve Neural Network PhiNet
#   Choose optimal standard hyperparameters
#   Add PC structure(s) -> PCs, CLTs, ...       (also parameter for max depth?)
#   Do some testing with accuracy/log likelihood etc.
#   Add GPU acceleration

class CM_TPM(nn.Module):
    def __init__(self, pc_type, input_dim, latent_dim, num_components, net=None, custom_layers=[2, 64, "ReLU", False, 0.0], random_state=None):
        """
        The CM-TPM class the performs all the steps from the CM-TPM.

        Parameters:
            pc_type: Type of PC to use ("factorized", "spn", "clt").
            input_dim: Dimensionality of input data.
            latent_dim: Dimensionality of latent variable z.
            num_components: Number of mixture components (integration points).
            net (optional): A custom neural network for PC structure generation.
            random_state (optional): Random seed for reproducibility.

        Attributes:
            phi_net: The neural network that is used to generate PCs.
            pcs: The PCs that are used for computing log likelihoods, number of PCs is equal to num_components.
            is_trained: Whether the CM-TPM has been trained.
        """
        super().__init__()
        self.pc_type = pc_type
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_components = num_components
        self.random_state = random_state
        self.z = None

        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)

        # Neural network to generate PC parameters
        self.phi_net = PhiNet(latent_dim, input_dim, pc_type=pc_type, net=net, hidden_layers=custom_layers[0], neurons_per_layer=custom_layers[1], activation=custom_layers[2], batch_norm=custom_layers[3], dropout_rate=custom_layers[4])

        # # Create multiple PCs (one per component)
        # self.pcs = nn.ModuleList([get_probabilistic_circuit(pc_type, input_dim) for _ in range(num_components)])

        self._is_trained = False
    
    def forward(self, x, z_samples, w, k=None, n_components=None):
        """
        Compute the mixture likelihood.

        Parameters:
            x: Input batch of shape (batch_size, input_dim).
            z_samples: Integration points of shape (num_components, latent_dim).
            w: Weights of each integration points.
            k (optional): Number of top components to consider for the mixture likelihood.
            n_components (optional): Number of mixture components. If none, use the same as during training.
        Returns:
            mixture_likelihood: The likelihood of the x given z_samples.
        """
        # Set the corrrect amount of components
        if n_components is not None:
            num_components = n_components
        else:
            num_components = self.num_components
        
        # Create multiple PCs (one per component)
        pcs = nn.ModuleList([get_probabilistic_circuit(self.pc_type, self.input_dim) for _ in range(num_components)])

        if x.shape[1] != self.input_dim:
            raise ValueError(f"Invalid input tensor x. Expected shape: ({x.shape[0]}, {self.input_dim}), but got shape: ({x.shape[0]}, {x.shape[1]}).")
        if z_samples.shape[0] != num_components or z_samples.shape[1] != self.latent_dim:
            raise ValueError(f"Invalid input tensor z_samples. Expected shape: ({num_components}, {self.latent_dim}), but got shape: ({z_samples.shape[0]}, {z_samples.shape[1]}).")

        if self.z is None:
            phi_z = self.phi_net(z_samples)  # Generate parameters for each PC, shape: (num_components, 2 * input_dim)
        else:
            phi_z = self.phi_net(self.z)
        
        mask = ~torch.isnan(x)

        # Compute likelihood without filling in missing values
        # TODO: Fix bug!
        x = torch.where(mask, x, 0.5)       # This line does not do anything, but the code breaks if I remove it

        likelihoods = []
        for i in range(num_components):
            pcs[i].set_params(phi_z[i])  # Assign PC parameters
            likelihood = pcs[i](x, mask)  # Compute p(x | phi(z_i)), shape: (batch_size)

            if torch.isnan(likelihood).any():
                raise ValueError(f"NaN detected in likelihood at component {i}: {likelihood}")

            likelihoods.append(likelihood + torch.log(w[i]))   # Add weighted likelihood to the list

        likelihoods = torch.stack(likelihoods, dim=0)   # Shape: (num_components, batch_size)

        if k is not None and k < num_components:        
            top_k_values, _ = torch.topk(likelihoods, k, dim=0)  # Get top K values and indices
            mixture_likelihood = torch.logsumexp(top_k_values, dim=0)  # Take the sum of the weighted likelihoods, shape: (batch_size)
        else:
            mixture_likelihood = torch.logsumexp(likelihoods, dim=0)      # Take the sum of the weighted likelihoods, shape: (batch_size)
        
        return torch.mean(mixture_likelihood)  # Average over batch

class BaseProbabilisticCircuit(nn.Module, ABC):
    """Base Probabilistic Circuit, other PC structures inherit from this class"""
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim

    @abstractmethod
    def set_params(self, params):
        """Set the parameters of the PC"""
        pass

    @abstractmethod
    def forward(self, x):
        """Compute p(x | phi(z))"""
        pass

class FactorizedPC(BaseProbabilisticCircuit):
    """A factorized PC structure."""
    def __init__(self, input_dim):
        super().__init__(input_dim)
        self.params = None
    
    def set_params(self, params):
        """Set the parameters for the factorized PC"""
        self.params = params

    def forward(self, x, ignore_mask=None):
        """Compute the likelihood using a factorized Gaussian model"""
        if self.params is None:
            raise ValueError("PC parameters are not set. Call set_params(phi_z) first.")
        if self.params.shape != (self.input_dim * 2,):  # Ensure twice the size of features
            raise ValueError(f"Expected params of shape ({self.input_dim * 2},), got {self.params.shape}")
        
        # Extract the means and log variances from the parameters
        means, log_vars = torch.chunk(self.params, 2, dim=-1)
        stds = torch.exp(0.5 * log_vars).clamp(min=1e-3)

        # Compute Gaussian likelihood per feature
        log_prob = -0.5 * (((x - means) / stds) ** 2 + 2 * torch.log(stds) + math.log(2 * math.pi))

        # epsilon = 0.1       # Small epsilon to compute approximate probability
        # prob_low = 0.5 * (1 + torch.erf((x - epsilon - means) / (stds * math.sqrt(2))))     # Lower probability
        # prob_high = 0.5 * (1 + torch.erf((x + epsilon - means) / (stds * math.sqrt(2))))    # Upper probability
        # prob = (prob_high - prob_low).clamp(min=1e-9)       # Set minimum to avoid 0
        # log_prob = torch.log(prob)

        # Set all NaN values to 0
        if torch.any(torch.isnan(log_prob)):
            log_prob[torch.isnan(log_prob)] = 0.0
        # If we ignore missing values, set those to 0
        if ignore_mask is not None:
            log_prob = torch.where(ignore_mask, log_prob, 0.0)

        # Sum the probabilities to obtain a likelihood for each sample
        log_likelihood = torch.sum(log_prob, dim=-1)
        return log_likelihood
        

class SPN(BaseProbabilisticCircuit):
    """An SPN PC structure."""
    def __init__(self, input_dim, num_sums=1, num_prods=2):
        super().__init__(input_dim)
        self.num_sums = num_sums
        self.num_prods = num_prods
        self.params = None

        self.sum_weights = nn.Parameter(torch.randn(num_sums, num_prods))   # TODO: Make weight initialized by phi
    
    def set_params(self, params):
        """Set the parameters for the SPN"""
        self.params = params

    def forward(self, x, ignore_mask=None):
        """SPN computation"""
        if self.params is None:
            raise ValueError("PC parameters are not set. Call set_params(phi_z) first.")
        if self.params.shape != (self.input_dim * 2,):  # Ensure twice the size of features
            raise ValueError(f"Expected params of shape ({self.input_dim * 2},), got {self.params.shape}")
        
        batch_size = x.shape[0]
        means, log_vars = torch.chunk(self.params, 2, dim=-1)
        stds = torch.exp(0.5 * log_vars) + 1e-6

        # Compute Gaussian likelihoods
        leaf_probs = torch.exp(-0.5 * ((x.unsqueeze(1) - means) / stds) ** 2) / (stds * torch.sqrt(torch.tensor(2 * torch.pi)))
        
        # Dynamically compute feature groups and number of product nodes
        feature_group_sizes, num_products = _dynamic_feature_grouping(self.input_dim, self.num_sums, self.num_prods)

        # Update sum_weights to match new `num_products`
        self.sum_weights = nn.Parameter(torch.randn(self.num_sums, num_products))

        # Split leaf_probs into dynamically assigned feature groups
        split_indices = torch.cumsum(torch.tensor([0] + feature_group_sizes), dim=0)
        grouped_leaf_probs = [leaf_probs[:, :, split_indices[i]:split_indices[i+1]] for i in range(num_products)]

        # Compute product node probabilities
        product_probs = [torch.prod(group, dim=-1) for group in grouped_leaf_probs]
        product_probs = torch.stack(product_probs, dim=-1)  # Shape: (batch_size, num_sums, num_products)

        # Sum node weighted aggregation (now correctly aligned)
        sum_weights = F.softmax(self.sum_weights, dim=-1)  # Ensure sum_weights shape matches product_probs
        sum_probs = torch.sum(sum_weights * product_probs, dim=-1)  # Weighted sum over product nodes

        return torch.mean(sum_probs, dim=-1)

class ChowLiuTreePC(BaseProbabilisticCircuit):
    """A Chow Liu Tree PC structrue. (Not implemented yet)"""
    def __init__(self, input_dim):
        super().__init__(input_dim)
        self.params = None
        self.tree_structure = None
    
    def set_params(self, params):
        """Set the parameters for the CLT"""

        if torch.isnan(params).any():
            raise ValueError(f"NaN detected in phi_z output before splitting: {params}")
    
        if params.shape != (self.input_dim * 2,):  # Ensure twice the size of features
            raise ValueError(f"Expected params of shape ({self.input_dim * 2},), got {params.shape}")
        
        raw_means, raw_stds = params[:self.input_dim], params[self.input_dim:]
        if torch.isnan(raw_means).any() or torch.isnan(raw_stds).any():
            raise ValueError(f"NaN detected after splitting: means={means}, raw_stds={raw_stds}")
        
        means = torch.tanh(raw_means) * 2
        stds = torch.nn.functional.softplus(raw_stds) + 1e-6
        if torch.isnan(stds).any():
            raise ValueError(f"NaN detected after softplus transformation: stds={stds}")

        self.params = torch.stack((means, stds), dim=1)

    def fit_tree(self, data, ignore_mask=None):
        n_features = data.shape[1]
        mutual_info_matrix = np.zeros((n_features, n_features))

        for i in range(n_features):
            for j in range(i + 1, n_features):
                corr = np.corrcoef(data[:, i], data[:, j])[0, 1]    # Bug: Returns null if elements are identical
                mutual_info_matrix[i, j] = mutual_info_matrix[j, i] = np.abs(corr)

        G = nx.Graph()
        for i in range(n_features):
            for j in range(i + 1, n_features):
                G.add_edge(i, j, weight=mutual_info_matrix[i, j])

        self.tree_structure = nx.maximum_spanning_tree(G)

    def forward(self, x):
        """Placeholder for CLT computation"""
        if self.params is None:
            raise ValueError("PC parameters are not set. Call set_params(phi_z) first.")
        if self.tree_structure is None:
            self.fit_tree(x)

        log_likelihood = torch.zeros(x.shape[0])
        for edge in self.tree_structure.edges():
            i, j = edge
            mean_i, std_i = self.params[i]
            mean_j, std_j = self.params[j]

            if torch.isnan(mean_i) or torch.isnan(std_i):
                raise ValueError(f"NaN detected in mean/std: mean_i={mean_i}, std_i={std_i}")

            normal_i = dist.Normal(mean_i, std_i)
            normal_j = dist.Normal(mean_j, std_j)

            if torch.isnan(x[:, i]).any() or torch.isnan(x[:, j]).any():
                raise ValueError(f"NaN detected in input x[:, {i}] or x[:, {j}]")

            p_xi_given_xj = normal_i.log_prob(x[:, i]) + normal_j.log_prob(x[:, j])  # Log probabilities
            if torch.isnan(p_xi_given_xj).any():
                raise ValueError(f"NaN detected in log_prob computation: mean_i={mean_i}, std_i={std_i}, mean_j={mean_j}, std_j={std_j}")
            
            log_likelihood += p_xi_given_xj

        return log_likelihood

def get_probabilistic_circuit(pc_type, input_dim):
    """Factory function for the different PC types."""
    types = ["factorized", "spn", "clt"]
    if pc_type == "factorized":
        return FactorizedPC(input_dim)
    elif pc_type == "spn" or pc_type == "SPN":
        return SPN(input_dim)
    elif pc_type == "clt" or pc_type == "CLT":
        return ChowLiuTreePC(input_dim)
    else:
        raise ValueError(f"Unknown PC type: '{pc_type}', use one of the following types: {types}")

class PhiNet(nn.Module):
    """
    Neural network for mapping latent variable z to PC parameters.
    
    Parameters:
        latent_dim: Dimensionality of latent variable z.
        pc_param_dim: Dimensionality of the PC parameters.
        net (optional): A custom neural network.
        hidden_layers (optional): Number of hidden layers in the neural network.
        neurons_per_layer (optional): Number of neurons per layer in the neural network.
        activation (optional): The activation function in the neural network.
        batch_norm (optional): Whether to use batch normalization in the neural network.
        dropout_rate (optional): Dropout rate in the neural network.
    """
    def __init__(self, latent_dim, pc_param_dim, pc_type="factorized", net=None, hidden_layers=2, neurons_per_layer=64, activation="ReLU", batch_norm=False, dropout_rate=0.0):
        super().__init__()
        out_dim = pc_param_dim * 2
        if net:
            if not isinstance(net, nn.Sequential):
                raise ValueError(f"Invalid input net. Please provide a Sequential neural network from torch.nn .")
            if net[0].in_features != latent_dim:
                raise ValueError(f"Invalid input net. The first layer should have {latent_dim} input features, but is has {net[0].in_features} input features.")
            if net[-1].out_features != out_dim:
                raise ValueError(f"Invalid input net. The final layer should have {out_dim} output features, but is has {net[-1].out_features} output features.")
            self.net = net
        else:
            # Extend single value neurons_per_layer to a list of size hidden_layers
            if isinstance(neurons_per_layer, int): 
                neurons_per_layer = [neurons_per_layer] * hidden_layers
            # Check if the sizes of hidden_layers and neuron_per_layer match
            if len(neurons_per_layer) != hidden_layers:
                raise ValueError(f"The hidden layers and neurons per layer do not match. Hidden layers: {hidden_layers}, neurons per layer: {neurons_per_layer}")

            # Get the chosen activation function
            activations_list = {
                "ReLU": nn.ReLU(),
                "Tanh": nn.Tanh(),
                "Sigmoid": nn.Sigmoid(),
                "LeakyReLU": nn.LeakyReLU(),
            }
            activation_fn = activations_list.get(activation, nn.ReLU())     # Default is ReLU

            # Create the neural network layer by layer
            layers = []
            for i in range(hidden_layers):
                # Set the input and output dimensions
                input_dim = latent_dim if i == 0 else neurons_per_layer[i-1]
                output_dim = neurons_per_layer[i]
                layers.append(nn.Linear(input_dim, output_dim))
                # Add batch normalization if enabled
                if batch_norm:
                    layers.append(nn.BatchNorm1d(output_dim))
                # Add the activation function
                layers.append(activation_fn)
                # Add a dropout layer if enabled
                if dropout_rate > 0.0:
                    layers.append(nn.Dropout(dropout_rate))

            # Create the output layer
            layers.append(nn.Linear(neurons_per_layer[-1], out_dim))

            # Store the neural network
            self.net = nn.Sequential(*layers)

    def forward(self, z):
        """
        Run z through the neural network.

        Parameters:
            z: Integration points of shape (num_components, latent_dim)

        Returns: 
            phi(z): The pc parameters obtained by runiing z throuh the neural network, of shape (num_components, pc_param_dim)
        """
        if z.shape[1] != self.net[0].in_features:
            raise ValueError(f"Invalid input to the neural network. Expected shape for z: ({z.shape[0]}, {self.net[0].in_features}), but got shape: ({z.shape[0]}, {z.shape[1]}).")
        return self.net(z)

def generate_rqmc_samples(num_samples, latent_dim, random_state=None):
    """
    Generates samples using Randomized Quasi Monte Carlo.
    
    Parameters:
        num_samples: The number of samples to generate.
        latent_dim: Dimensionality of the latent variable

    Returns:
        z_samples: The sampled values z of shape (num_samples, latent_dim)
        w: The weights for the mixture components
    """
    sampler = qmc.Sobol(d=latent_dim, scramble=True, seed=random_state)
    z_samples = sampler.random(n=num_samples)
    z_samples = torch.tensor(qmc.scale(z_samples, -3, 3), dtype=torch.float32)  # Scale for Gaussian prior
    w = torch.full(size=(num_samples,), fill_value=1 / num_samples)     # Uniform weights
    return z_samples, w

def train_cm_tpm(
        train_data, 
        pc_type="factorized", 
        latent_dim=16, 
        num_components=256,
        num_components_impute=None,
        k=None,
        lo=False,
        net=None, 
        hidden_layers=2,
        neurons_per_layer=64,
        activation="ReLU",
        batch_norm=False,
        dropout_rate=0.0,
        epochs=100,
        batch_size=32,
        tol=1e-5, 
        lr=0.001,
        weight_decay=1e-5,
        random_state=None,
        verbose=0,
        ):
    """
    The training function for CM-TPM. Creates a CM-TPM model and trains the parameters.
    
    Parameters:
        train_data: The data to train the CM-TPM on.
        pc_type (optional): The type of PC to use (factorized, spn, clt).
        latent_dim (optional): Dimensionality of the latent variable. 
        num_components (optional): Number of mixture components.
        num_components_impute (optional): Number of mixture components for imputation.
        net (optional): A custom neural network.
        hidden_layers (optional): Number of hidden layers in the neural network.
        neurons_per_layer (optional): Number of neurons per layer in the neural network.
        activation (optional): The activation function in the neural network.
        batch_norm (optional): Whether to use batch normalization in the neural network.
        dropout_rate (optional): Dropout rate in the neural network.
        epochs (optional): The number of training loops.
        batch_size (optional): The batch size for training or None if not using batches.
        tol (optional): Tolerance for the convergence criterion.
        lr (optional): The learning rate of the optimizer.
        weight_decay (optional): Weight decay for the optimizer.
        random_state (optional): A random seed for reproducibility. 
        verbose (optional): Verbosity level.

    Returns:
        model: A trained CM-TPM model
    """
    input_dim = train_data.shape[1]

    # Define the model
    model = CM_TPM(pc_type, input_dim, latent_dim, num_components, net=net, 
                   custom_layers=[hidden_layers, neurons_per_layer, activation, batch_norm, dropout_rate], random_state=random_state)

    if verbose > 1:
        print(f"Finished building CM-TPM model with {num_components} components.")

    # Set the optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if verbose > 0:
        print(f"Starting training with {epochs} epochs...")
    prev_loss = -float('inf')       # Initial loss
    start_time = time.time()        # Keep track of training time
        
    # Create DataLoader
    x_tensor = torch.tensor(train_data, dtype=torch.float32)
    if batch_size is not None:
        train_loader = DataLoader(TensorDataset(x_tensor), batch_size=batch_size, shuffle=True)
    else:   # No batches
        train_loader = [torch.unsqueeze(x_tensor, 0)]   # Add batch dimension

    for epoch in range(epochs):
        start_time_epoch = time.time()

        total_loss = 0.0       # Keep track of the total loss for the epoch

        for batch in train_loader:  # Iterate over batches
            x_batch = batch[0]      # Extract batch data

            # Generate new z samples and weights
            z_samples, w = generate_rqmc_samples(num_components, latent_dim, random_state=random_state)

            optimizer.zero_grad()       # Reset gradients

            loss = -model(x_batch, z_samples, w, k=k)    # Compute loss

            if torch.isnan(loss).any():
                raise ValueError(f"NaN detected in loss at epoch {epoch}: {loss}")

            loss.backward()     # Backpropagation

            for name, param in model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    raise ValueError(f"NaN detected in gradient of {name} at epoch {epoch}")
            
            optimizer.step()        # Update model parameters

            total_loss += loss.item()       # Accumulate loss

        average_loss = total_loss / len(train_loader)       # Average loss over batches

        # Check early stopping criteria
        if epoch > 10 and abs(average_loss - prev_loss) < tol:
                if verbose > 0:
                    print(f"Early stopping at epoch {epoch} due to small log likelihood improvement.")
                break
        prev_loss = average_loss
            
        if verbose > 1:
            print(f"Epoch {epoch}, Log-Likelihood: {-average_loss}, Training time: {time.time() - start_time_epoch}")
        elif verbose > 0:
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Log-Likelihood: {-average_loss}')

    if verbose > 0:
        print(f"Training complete.")
        print(f"Final Training Log-Likelihood: {-average_loss}")
    if verbose > 1:
        print(f"Total training time: {time.time() - start_time}")
    model._is_trained = True        # Mark model as trained

    if lo:
        model.z = latent_optimization(model, train_loader, num_components_impute, latent_dim, epochs=math.ceil(epochs/2), tol=tol, lr=lr, weight_decay=weight_decay, random_state=random_state, verbose=verbose)  # Optimize z_samples

    return model

def latent_optimization(
        model,
        train_loader, 
        num_components=None,
        latent_dim=16,
        epochs=100, 
        tol=1e-5,
        lr=0.01, 
        weight_decay=1e-5,
        random_state=None, 
        verbose=0):
    """
    Optimizes the integration points z after training.

    Parameters:
        model: A trained CM-TPM model.
        train_loader: A DataLoader for the training data.
        num_components (optional): The number of mixture components.
        latent_dim (optional): The dimensionality of the latent variable.
        epochs (optional): The number of optimization loops.
        tol (optional): Tolerance for the convergence criterion.
        lr (optional): The learning rate during optimization. 
        weight_decay (optional): Weight decay for the optimizer.
        random_state (optional): A random seed for reproducibility. 
        verbose (optional): Verbosity level.

    Returns:
        Optimized z_samples.
    """
    # Set the corrrect amount of components
    if num_components is not None:
        n_components = num_components
    else:
        n_components = model.num_components

    # Generate new z samples and weights
    z_samples, w = generate_rqmc_samples(n_components, latent_dim, random_state=random_state)

    # Make z_samples a parameter to optimize
    z_optimized = torch.nn.Parameter(z_samples.clone().detach(), requires_grad=True)  
    optimizer = optim.Adam([z_optimized], lr=lr, weight_decay=weight_decay)

    if verbose > 0:
        print(f"Starting latent optimization with {epochs} epochs...")
    prev_loss = -float('inf')       # Initial loss
    start_time = time.time()        # Keep track of training time

    for epoch in range(epochs):
        start_time_epoch = time.time()

        total_loss = 0.0       # Keep track of the total loss for the epoch

        for batch in train_loader:  # Iterate over batches
            x_batch = batch[0]      # Extract batch data

            optimizer.zero_grad()

            # Compute the loss with optimized z
            loss = -model(x_batch, z_optimized, w, n_components=n_components)    # Compute loss

            loss.backward()  # Backpropagation

            optimizer.step()  # Update z_samples

            total_loss += loss.item()       # Accumulate loss

        average_loss = total_loss / len(train_loader)       # Average loss over batches

        # Check early stopping criteria
        if epoch > 10 and abs(average_loss - prev_loss) < tol:
                if verbose > 0:
                    print(f"Early stopping at epoch {epoch} due to small log likelihood improvement.")
                break
        prev_loss = average_loss

        if verbose > 1:
            print(f"Epoch {epoch}, Log-Likelihood: {-average_loss}, Training time: {time.time() - start_time_epoch}")
        elif verbose > 0:
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Log-Likelihood: {-average_loss}')

    if verbose > 0:
        print(f"Latent optimization complete.")
        print(f"Final Latent Optimization Log-Likelihood: {-average_loss}")
    if verbose > 1:
        print(f"Total optimization time: {time.time() - start_time}")
    
    return z_optimized.detach()  # Return optimized z_samples


def impute_missing_values(
        x_incomplete, 
        model,
        num_components=None,
        epochs=100,
        lr=0.01,
        random_state=None,
        verbose=0,
        skip=False,
        ):
    """
    Imputes missing data using a specified model.
    
    Parameters:
        x_incomplete: The input data with missing values.
        model: A CM-TPM model to use for data imputation.
        epochs (optional): The number of imputation loops.
        lr (optional): The learning rate during imputation. 
        random_state (optional): A random seed for reproducibility. 
        verbose (optional): Verbosity level.
        skip (optional): Skips the model fitted check, used for EM.

    Returns:
        x_imputed: A copy of x_incomplete with the missing values imputed.
    """
    if not np.isnan(x_incomplete).any():
        return x_incomplete
    
    if not model._is_trained and not skip:
        raise ValueError("The model has not been fitted yet. Please call the fit method first.")
    
    if x_incomplete.shape[1] != model.input_dim:
        raise ValueError(f"The missing data does not have the same number of features as the training data. Expected features: {model.input_dim}, but got features: {x_incomplete.shape[1]}.")
    
    if verbose > 0:
        print(f"Starting with imputing data...")
    start_time = time.time()

    if random_state is not None:
        set_random_seed(random_state)

    # Set the corrrect amount of components
    if num_components is not None:
        n_components = num_components
    else:
        n_components = model.num_components

    # Generate new samples and weights
    z_samples, w = generate_rqmc_samples(n_components, model.latent_dim, random_state=random_state)
    
    # Create a tensor with the data to impute
    x_incomplete = torch.tensor(x_incomplete, dtype=torch.float32)
    x_imputed = x_incomplete.clone().detach()

    # Get a copy of X with only rows with missing values
    nan_rows_mask = torch.any(torch.isnan(x_incomplete), dim=1)  # True if a row has NaN
    nan_rows_indices = torch.where(nan_rows_mask)[0]  # Get row indices
    x_missing_rows = x_incomplete[nan_rows_mask]  # Extract rows with NaN

    # Initially impute with a standard value of 0.5
    mask = ~torch.isnan(x_missing_rows)
    x_missing_rows[~mask] = 0.5

    # Create tensor with only values that need to be changed
    x_missing_vals = x_missing_rows[~mask].clone().detach().requires_grad_(True)

    # Set up optimizer
    optimizer = optim.Adam([x_missing_vals], lr=lr)

    # Imputation loop
    for epoch in range(epochs):
        optimizer.zero_grad()

        # Insert the missing values into the correct places
        x_missing_rows = x_missing_rows.clone().detach()
        x_missing_rows = x_missing_rows.masked_scatter(~mask, x_missing_vals)

        # Optimization step
        loss = -model(x_missing_rows, z_samples, w, n_components=n_components)
        loss.backward()
        optimizer.step()

        # Keep imputed values in (0,1) range
        with torch.no_grad():
            x_missing_vals.clamp_(0, 1)

        if verbose > 1:
                print(f"Epoch {epoch}, Log-Likelihood: {-loss.item()}")
        elif verbose > 0:
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Log-Likelihood: {-loss.item()}')

    if verbose > 0:
        print(f"Finished imputing data.")
        print(f"Succesfully imputed {torch.sum(~mask).item()} values.")
        print(f"Final Imputed Data Log-Likelihood: {-loss.item()}")
    if verbose > 1:
        print(f"Total imputation time: {time.time() - start_time}")

    # Return completed data with imputed values
    x_missing_rows = x_missing_rows.masked_scatter(~mask, x_missing_vals)
    x_imputed[nan_rows_indices] = x_missing_rows
    return x_imputed.detach().cpu().numpy()

def impute_missing_values_exact(
        x_incomplete, 
        model,
        random_state=None,
        verbose=0,
        skip=False,
        ):
    """
    Imputes missing data using a specified model.
    
    Parameters:
        x_incomplete: The input data with missing values.
        model: A CM-TPM model to use for data imputation.
        epochs (optional): The number of imputation loops.
        lr (optional): The learning rate during imputation. 
        random_state (optional): A random seed for reproducibility. 
        verbose (optional): Verbosity level.
        skip (optional): Skips the model fitted check, used for EM.

    Returns:
        x_imputed: A copy of x_incomplete with the missing values imputed.
    """
    if not np.isnan(x_incomplete).any():
        return x_incomplete
    
    if not model._is_trained and not skip:
        raise ValueError("The model has not been fitted yet. Please call the fit method first.")
    
    if x_incomplete.shape[1] != model.input_dim:
        raise ValueError(f"The missing data does not have the same number of features as the training data. Expected features: {model.input_dim}, but got features: {x_incomplete.shape[1]}.")
    
    if verbose > 0:
        print(f"Starting with imputing data...")
    start_time = time.time()

    if random_state is not None:
        set_random_seed(random_state)

    # Generate new samples and weights
    z_samples, w = generate_rqmc_samples(model.num_components, model.latent_dim, random_state=random_state)
    
    # Store which values are missing
    x_incomplete = torch.tensor(x_incomplete, dtype=torch.float32)
    mask = ~torch.isnan(x_incomplete)

    # Initially impute randomly
    x_imputed = x_incomplete.clone().detach()

    # Sample means and variances from the neural network
    phi_z = model.phi_net(z_samples)
    means, log_vars = torch.chunk(phi_z, 2, dim=-1)
    stds = torch.exp(0.5 * log_vars).clamp(min=1e-3)

    epsilon = 0.1       # Small epsilon to compute approximate probability

    for k in range(x_incomplete.shape[0]):      # Iterate over each sample
        # If there are no missing values, skip this sample
        if not torch.any(torch.isnan(x_incomplete[k])):
            continue
        
        # Create tensor that tracks the likelihood for each component
        likelihoods = torch.zeros(means.shape[0])  
        for i in range(means.shape[0]):     # Iterate over each component
            likelihood = 0
            for j in range(x_incomplete.shape[1]):      # Iterate over each feature
                x = x_incomplete[k, j]
                # Likelihood computation only for non-missing values
                if not torch.isnan(x):
                    mean, std = means[i, j], stds[i, j]
                    # log_prob_low = 0.5 * (1 + torch.erf((x - epsilon - mean) / (std * math.sqrt(2))))
                    # log_prob_high = 0.5 * (1 + torch.erf((x + epsilon - mean) / (std * math.sqrt(2))))
                    # log_prob = torch.log((log_prob_high - log_prob_low).clamp(min=1e-9))
                    log_prob = -0.5 * (((x - mean) / std) ** 2 + 2 * torch.log(std) + math.log(2 * math.pi))
                    likelihood += log_prob      # Take the sum of the log likelihoods
            # Store the likelihood for this component
            likelihoods[i] = likelihood
        
        # Get the component with the maximum likelihood for this sample
        i_max = torch.argmax(likelihoods)

        if verbose > 1:
            print(f"Sample {k}, most likely component: {i_max}, component means: {means[i_max]}.")

        # Impute the missing values with the means of the most likely component
        for j in range(x_incomplete.shape[1]):
            if torch.isnan(x_incomplete[k, j]):
                x_imputed[k, j] = means[i_max, j]
        
    return x_imputed.detach().cpu().numpy()

def set_random_seed(seed):
    """Ensure reproducibility by setting random seeds for all libraries."""
    np.random.seed(seed)  # NumPy's random generator
    torch.manual_seed(seed)  # PyTorch CPU random generator
    torch.cuda.manual_seed_all(seed)  # If using GPU

def _dynamic_feature_grouping(input_dim, num_sums, num_products):
    """Dynamically assigns feature groups when input_dim is not evenly divisible."""
    num_groups = num_sums * num_products  # Total product nodes
    base_group_size = input_dim // num_groups  # Minimum features per product node
    remainder = input_dim % num_groups  # Features that need to be distributed

    # Create feature sizes for each group
    feature_group_sizes = [base_group_size + (1 if i < remainder else 0) for i in range(num_groups)]
    
    return feature_group_sizes, num_groups

# Example Usage
if __name__ == '__main__':
    # train_data = np.random.rand(1000, 10)
    # train_data = np.random.uniform(low=-1, high=1, size=(1000, 10))
    # train_data[999, 9] = np.nan
    # model = train_cm_tpm(train_data, pc_type="spn", random_state=None, missing_strategy="mean", epochs=10, verbose=1)

    # x_incomplete = train_data[:3].copy()
    # x_incomplete[0, 0] = np.nan
    # x_incomplete[2, 9] = np.nan
    # x_imputed = impute_missing_values(x_incomplete, model, random_state=None, verbose=1)
    # print("Original Data:", train_data[:3])
    # print("Data with missing:", x_incomplete)
    # print("Imputed values:", x_imputed)

    start_time = time.time()
    all_zeros = np.full((100, 10), 0.89)
    all_zeros[50, 3] = np.nan
    all_zeros[10, 2] = np.nan
    all_zeros[92, 0] = np.nan
    model = train_cm_tpm(all_zeros, 
                         pc_type="factorized", 
                         verbose=1, 
                         epochs=100, 
                         lr=0.001, 
                         num_components=256,
                         num_components_impute=512,
                         k=5,
                         lo=True,
                         batch_size=None,
                         )
    #imputed = impute_missing_values_exact(all_zeros, model, verbose=1)
    imputed = impute_missing_values(all_zeros, model, num_components=512, verbose=1)
    print(imputed[50, 3])
    print(imputed[10, 2])
    print(imputed[92, 0])
    print("Imputation time:", time.time() - start_time)

    # all_zeros = np.full((6, 6), 0.23)
    # all_zeros[0, 0] = np.nan
    # model = train_cm_tpm(all_zeros, pc_type="factorized", verbose=2, epochs=150, lr=0.001, num_components=1024)
    # imputed = impute_missing_values_exact(all_zeros, model, lr=0.01, epochs=100, verbose=1)
    # print(imputed[0, 0])


    # test_x = torch.randn(5, 10)  # Small test batch
    # test_pc = ChowLiuTreePC(input_dim=10)
    # test_params = torch.randn(10 * 2)  # Simulated params
    # test_pc.set_params(test_params)

    # print("Running test forward pass...")
    # likelihood = test_pc.forward(test_x)
    # print("Likelihood output:", likelihood)
