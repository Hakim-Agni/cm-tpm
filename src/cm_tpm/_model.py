from abc import ABC, abstractmethod
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dist
import numpy as np
from scipy.stats import qmc
import networkx as nx

# TODO:
#   Data preprocessing:  
#           Detect binary variable that are floats
#   Batches
#   Allow custom Optimizers?
#   Add PC structure(s) -> PCs, CLTs, ...       (also parameter for max depth?)
#   Improve Neural Network PhiNet
#   Optimize latent selection (top-K selection)
#   Implement latent optimization for fine-tuning integration points
#   Do some testing with accuracy/log likelihood etc.
#   Choose optimal standard hyperparameters
#   Add GPU acceleration

class CM_TPM(nn.Module):
    def __init__(self, pc_type, input_dim, latent_dim, num_components, missing_strategy="mean", net=None, custom_layers=[2, 64, "ReLU", False, 0.0], smooth=1e-6, random_state=None):
        """
        The CM-TPM class the performs all the steps from the CM-TPM.

        Parameters:
            pc_type: Type of PC to use ("factorized", "spn", "clt").
            input_dim: Dimensionality of input data.
            latent_dim: Dimensionality of latent variable z.
            num_components: Number of mixture components (integration points).
            missing_strategy (optional): Strategy for dealing with missing values in training data ("mean", "zero", "em", "ignore").
            net (optional): A custom neural network for PC structure generation.
            smooth (optional): A smooting parameter to avoid division by zero.
            random_state (optional): Random seed for reproducibility.

        Attributes:
            phi_net: The neural network that is used to generate PCs.
            pcs: The PCs that are used for computing log likelihoods, number of PCs is equal to num_components.
            is_trained: Whether the CM-TPM has been trained.
        """
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_components = num_components
        self.missing_strategy = missing_strategy
        self.random_state = random_state

        if missing_strategy not in ["mean", "zero", "em", "ignore"]:
            raise ValueError(f"Unknown missing values strategy: '{missing_strategy}', use one of the following: 'mean', 'zero', 'em', 'ignore'.")

        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)

        # Neural network to generate PC parameters
        self.phi_net = PhiNet(latent_dim, input_dim, pc_type=pc_type, net=net, hidden_layers=custom_layers[0], neurons_per_layer=custom_layers[1], activation=custom_layers[2], batch_norm=custom_layers[3], dropout_rate=custom_layers[4])

        # Create multiple PCs (one per component)
        self.pcs = nn.ModuleList([get_probabilistic_circuit(pc_type, input_dim, smooth) for _ in range(num_components)])

        self._is_trained = False
    
    def forward(self, x, z_samples, w):
        """
        Compute the mixture likelihood.

        Parameters:
            x: Input batch of shape (batch_size, input_dim).
            z_samples: Integration points of shape (num_components, latent_dim).
            w: Weights of each integration points

        Returns:
            mixture_likelihood: The likelihood of the x given z_samples.
        """
        if x.shape[1] != self.input_dim:
            raise ValueError(f"Invalid input tensor x. Expected shape: ({x.shape[0]}, {self.input_dim}), but got shape: ({x.shape[0]}, {x.shape[1]}).")
        if z_samples.shape[0] != self.num_components or z_samples.shape[1] != self.latent_dim:
            raise ValueError(f"Invalid input tensor z_samples. Expected shape: ({self.num_components}, {self.latent_dim}), but got shape: ({z_samples.shape[0]}, {z_samples.shape[1]}).")

        phi_z = self.phi_net(z_samples)  # Generate parameters for each PC, shape: (num_components, input_dim)
        mask = ~torch.isnan(x)
        ignore = None

        if self.missing_strategy == "mean":
            # Fill missing values with the mean of the feature
            x = torch.where(mask, x, torch.nanmean(x, dim=0, keepdim=True))
        elif self.missing_strategy == "ignore":
            # Compute likelihood without filling in missing values
            # TODO: Fix errors
            ignore = mask
            x = torch.where(mask, x, 0.0)
        elif self.missing_strategy == "zero":
            # Fill missing values with zero
            x = torch.where(mask, x, 0.0)

        likelihoods = []
        for i in range(self.num_components):
            self.pcs[i].set_params(phi_z[i])  # Assign PC parameters
            likelihood = self.pcs[i](x, ignore)  # Compute p(x | phi(z_i)), shape: (batch_size)

            if torch.isnan(likelihood).any():
                raise ValueError(f"NaN detected in likelihood at component {i}: {likelihood}")

            likelihoods.append(likelihood + torch.log(w[i]))   # Add weighted likelihood to the list

        # TODO: Other idea: take minimum likelihood per component? Since different components approximate different samples?

        likelihoods = torch.stack(likelihoods, dim=0)   # Shape: (num_components, batch_size)
        mixture_likelihood = torch.logsumexp(likelihoods, dim=0)      # Take the sum of the weighted likelihoods, shape: (batch_size)
        return torch.mean(mixture_likelihood)  # Average over batch

class BaseProbabilisticCircuit(nn.Module, ABC):
    """Base Probabilistic Circuit, other PC structures inherit from this class"""
    def __init__(self, input_dim, smooth=1e-6):
        super().__init__()
        self.input_dim = input_dim
        self.smooth = smooth

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
    def __init__(self, input_dim, smooth=1e-6):
        super().__init__(input_dim, smooth=smooth)
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
        # #log_prob = -0.5 * (((x - means) / stds) ** 2 + 2 * torch.log(stds) + math.log(2 * math.pi))
        # log_prob = -0.5 * (((x - means) / stds) ** 2)
        # # Sum across all features to get the total log-likelihood
        # log_likelihood = torch.sum(log_prob, dim=-1)

        epsilon = 0.1       # Small epsilon to compute approximate probability
        prob_low = 0.5 * (1 + torch.erf((x - epsilon - means) / (stds * math.sqrt(2))))     # Lower probability
        prob_high = 0.5 * (1 + torch.erf((x + epsilon - means) / (stds * math.sqrt(2))))    # Upper probability
        prob = (prob_high - prob_low).clamp(min=1e-9)       # Set minimum to avoid 0
        log_prob = torch.log(prob)

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
    def __init__(self, input_dim, smooth=1e-6, num_sums=1, num_prods=2):
        super().__init__(input_dim, smooth=smooth)
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
    def __init__(self, input_dim, smooth=1e-6):
        super().__init__(input_dim, smooth=smooth)
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

def get_probabilistic_circuit(pc_type, input_dim, smooth=1e-6):
    """Factory function for the different PC types."""
    types = ["factorized", "spn", "clt"]
    if pc_type == "factorized":
        return FactorizedPC(input_dim, smooth)
    elif pc_type == "spn" or pc_type == "SPN":
        return SPN(input_dim, smooth)
    elif pc_type == "clt" or pc_type == "CLT":
        return ChowLiuTreePC(input_dim, smooth)
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
        missing_strategy="mean",
        net=None, 
        hidden_layers=2,
        neurons_per_layer=64,
        activation="ReLU",
        batch_norm=False,
        dropout_rate=0.0,
        epochs=100,
        tol=1e-5, 
        lr=0.001,
        smooth=1e-6,
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
        missing_strategy (optional): Strategy for dealing with missing values in training data ("mean", "zero", "em", "ignore").
        net (optional): A custom neural network.
        hidden_layers (optional): Number of hidden layers in the neural network.
        neurons_per_layer (optional): Number of neurons per layer in the neural network.
        activation (optional): The activation function in the neural network.
        batch_norm (optional): Whether to use batch normalization in the neural network.
        dropout_rate (optional): Dropout rate in the neural network.
        epochs (optional): The number of training loops.
        tol (optional): Tolerance for the convergence criterion.
        lr (optional): The learning rate of the optimizer.
        random_state (optional): A random seed for reproducibility. 
        verbose (optional): Verbosity level.

    Returns:
        model: A trained CM-TPM model
    """
    input_dim = train_data.shape[1]
    model = CM_TPM(pc_type, input_dim, latent_dim, num_components, missing_strategy=missing_strategy, net=net, 
                   custom_layers=[hidden_layers, neurons_per_layer, activation, batch_norm, dropout_rate], smooth=smooth, random_state=random_state)

    if verbose > 1:
        print(f"Finished building CM-TPM model with {num_components} components.")

    optimizer = optim.AdamW(model.parameters(), lr=lr)

    if verbose > 0:
        print(f"Starting training with {epochs} epochs...")
    prev_loss = -float('inf')
    start_time = time.time()

    em_iters = 5 if missing_strategy == "em" and np.isnan(train_data).any() else 1
    for em_iter in range(em_iters):
        if verbose > 0 and em_iters > 1:
            print(f"EM Iteration {em_iter + 1}/{em_iters}")

        if missing_strategy == "em":
            imputed_data = impute_missing_values(train_data, model, skip=True)
            x_batch = torch.tensor(imputed_data, dtype=torch.float32)
        else:
            x_batch = torch.tensor(train_data, dtype=torch.float32)

        for epoch in range(epochs):
            start_time_epoch = time.time()

            z_samples, w = generate_rqmc_samples(num_components, latent_dim, random_state=random_state)

            optimizer.zero_grad()
            loss = -model(x_batch, z_samples, w)

            if torch.isnan(loss).any():
                raise ValueError(f"NaN detected in loss at epoch {epoch}: {loss}")
            
            if epoch > 10 and abs(loss - prev_loss) < tol:
                if verbose > 0:
                    print(f"Early stopping at epoch {epoch} due to small log likelihood improvement.")
                break
            prev_loss = loss

            loss.backward()

            for name, param in model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    raise ValueError(f"NaN detected in gradient of {name} at epoch {epoch}")
        
            optimizer.step()

            if verbose > 1:
                print(f"Epoch {epoch}, Log-Likelihood: {-loss.item()}, Training time: {time.time() - start_time_epoch}")
            elif verbose > 0:
                if epoch % 10 == 0:
                    print(f'Epoch {epoch}, Log-Likelihood: {-loss.item()}')

    if verbose > 0:
        print(f"Training complete.")
        print(f"Final Log-Likelihood: {-loss.item()}")
    if verbose > 1:
        print(f"Total training time: {time.time() - start_time}")
    model._is_trained = True
    return model

def impute_missing_values_sample(
        x_incomplete, 
        model,
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

    x_incomplete = torch.tensor(x_incomplete, dtype=torch.float32)
    x_filled = x_incomplete.clone()
    full_mask = ~torch.isnan(x_incomplete)

    for i in range(x_incomplete.shape[0]):
        x_sample = x_incomplete[i]
        
        # If there are no missing values, skip this loop
        if not torch.any(torch.isnan(x_sample)):
            continue

        x_sample = torch.unsqueeze(x_sample, 0)   # Add another dimension

        # Generate new samples and weights
        z_samples, w = generate_rqmc_samples(model.num_components, model.latent_dim, random_state=random_state)
        
        # Store which values are missing
        mask = ~torch.isnan(x_sample)

        if verbose > 0:
            print(f"Starting imputing {torch.sum(~mask).item()} values in sample {i}...")

        # Initially impute randomly
        x_imputed = x_sample.clone().detach()
        #x_imputed[~mask] = torch.rand_like(x_imputed[~mask])
        x_imputed[~mask] = 0.5
        x_imputed = x_imputed.clone().detach()

        # Create tensor with only values that need to be changed
        x_missing = x_imputed[~mask].clone().detach().requires_grad_(True)

        # Set up optimizer
        optimizer = optim.Adam([x_missing], lr=lr)

        # Imputation loop
        for epoch in range(epochs):
            optimizer.zero_grad()

            # Insert x_missing into x_imputed
            x_imputed = x_imputed.clone().detach()
            x_imputed = x_imputed.masked_scatter(~mask, x_missing)

            # # Optimization step
            loss = -model(x_imputed, z_samples, w)
            loss.backward()
            optimizer.step()

            # Keep imputed values in (0,1) range
            with torch.no_grad():
                x_missing.clamp_(0, 1)

            if verbose > 1:
                    print(f"Epoch {epoch}, Log-Likelihood: {-loss.item()}")
            elif verbose > 0:
                if epoch % 10 == 0:
                    print(f'Epoch {epoch}, Log-Likelihood: {-loss.item()}')

        if verbose > 0:
            print(f"Finished imputing {torch.sum(~mask).item()} values in sample {i}.")
        
        x_imputed = x_imputed.masked_scatter(~mask, x_missing)
        x_filled[i] = x_imputed

    if verbose > 0:
            print(f"Finished imputing data.")
            print(f"Succesfully imputed {torch.sum(~full_mask).item()} values.")
    if verbose > 1:
            print(f"Total imputation time: {time.time() - start_time}")

    # Return completed data with imputed values
    return x_filled.detach().cpu().numpy()

def impute_missing_values(
        x_incomplete, 
        model,
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

    # Generate new samples and weights
    z_samples, w = generate_rqmc_samples(model.num_components, model.latent_dim, random_state=random_state)
    
    # Store which values are missing
    x_incomplete = torch.tensor(x_incomplete, dtype=torch.float32)
    mask = ~torch.isnan(x_incomplete)

    # Initially impute randomly
    x_imputed = x_incomplete.clone().detach()
    #x_imputed[~mask] = torch.rand_like(x_imputed[~mask])
    x_imputed[~mask] = 0.5
    x_imputed = x_imputed.clone().detach()

    # Create tensor with only values that need to be changed
    x_missing = x_imputed[~mask].clone().detach().requires_grad_(True)

    # Set up optimizer
    optimizer = optim.Adam([x_missing], lr=lr)

    # Imputation loop
    for epoch in range(epochs):
        optimizer.zero_grad()

        # Insert x_missing into x_imputed
        x_imputed = x_imputed.clone().detach()
        x_imputed = x_imputed.masked_scatter(~mask, x_missing)

        # # Optimization step
        loss = -model(x_imputed, z_samples, w)
        loss.backward()
        optimizer.step()

        # Keep imputed values in (0,1) range
        with torch.no_grad():
            x_missing.clamp_(0, 1)

        if verbose > 1:
                print(f"Epoch {epoch}, Log-Likelihood: {-loss.item()}")
        elif verbose > 0:
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Log-Likelihood: {-loss.item()}')

    if verbose > 0:
        print(f"Finished imputing data.")
        print(f"Succesfully imputed {torch.sum(~mask).item()} values.")
    if verbose > 1:
        print(f"Total imputation time: {time.time() - start_time}")

    # Return completed data with imputed values
    x_imputed = x_imputed.masked_scatter(~mask, x_missing)
    return x_imputed.detach().cpu().numpy()

def impute_missing_values_exact(
        x_incomplete, 
        model,
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

    # Generate new samples and weights
    z_samples, w = generate_rqmc_samples(model.num_components, model.latent_dim, random_state=random_state)
    
    # Store which values are missing
    x_incomplete = torch.tensor(x_incomplete, dtype=torch.float32)
    mask = ~torch.isnan(x_incomplete)

    # Initially impute randomly
    x_imputed = x_incomplete.clone().detach()

    phi_z = model.phi_net(z_samples)
    means, log_vars = torch.chunk(phi_z, 2, dim=-1)
    stds = torch.exp(0.5 * log_vars).clamp(min=1e-3)
    #print(means, stds)
    #print(means.shape, stds.shape)

    epsilon = 0.1

    for k in range(x_incomplete.shape[0]):
        if not torch.any(torch.isnan(x_incomplete[k])):
            continue

        likelihoods = torch.zeros(means.shape[0])
        for i in range(means.shape[0]):
            likelihood = 0
            for j in range(x_incomplete.shape[1]):
                x = x_incomplete[k, j]
                if not torch.isnan(x):
                    mean, std = means[i, j], stds[i, j]
                    log_prob_low = 0.5 * (1 + torch.erf((x - epsilon - mean) / (std * math.sqrt(2))))
                    log_prob_high = 0.5 * (1 + torch.erf((x + epsilon - mean) / (std * math.sqrt(2))))
                    log_prob = torch.log(log_prob_high - log_prob_low)
                    likelihood += log_prob
            likelihoods[i] = likelihood
        
        i_max = torch.argmax(likelihoods)
        #print(f"Sample {k}, most likely component: {i_max}.")
        #print(means[i_max])
        #print(means)

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

    
    all_zeros = np.full((100, 10), 0.23)
    all_zeros[50, 3] = np.nan
    all_zeros[10, 2] = np.nan
    all_zeros[92, 0] = np.nan
    model = train_cm_tpm(all_zeros, 
                         pc_type="factorized", 
                         verbose=2, 
                         epochs=150, 
                         lr=0.001, 
                         num_components=256, 
                         missing_strategy="zero")
    imputed = impute_missing_values_exact(all_zeros, model, lr=0.01, epochs=100, verbose=1)
    print(imputed[50, 3])
    print(imputed[10, 2])
    print(imputed[92, 0])

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
