from abc import ABC, abstractmethod
import time
import math
from tqdm import tqdm
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dist
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from scipy.stats import qmc
import networkx as nx
import gc

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
    def __init__(self, pc_type, input_dim, latent_dim, num_components, net=None, custom_layers=[2, 64, "ReLU", False, 0.0, False], random_state=None):
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
        self.phi_net = PhiNet(latent_dim, input_dim, pc_type=pc_type, net=net, hidden_layers=custom_layers[0], neurons_per_layer=custom_layers[1], activation=custom_layers[2], batch_norm=custom_layers[3], dropout_rate=custom_layers[4], skip_layers=custom_layers[5])

        # Create multiple PCs (one per component)
        self.pcs = nn.ModuleList([get_probabilistic_circuit(pc_type, input_dim) for _ in range(num_components)])

        self._is_trained = False

    def forward(self, x, z_samples=None, w=None, k=None, n_components=None, device=None):
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
        num_components = n_components or self.num_components

        # Use RQMC samples for z if not provided
        if z_samples is None:
            rng = np.random.default_rng(self.random_state)  # Random number generator for reproducibility
            z_samples, w = generate_rqmc_samples(num_components, self.latent_dim, random_state=rng.integers(1e9), device=device)

        # Error checks
        if x.shape[1] != self.input_dim:
            raise ValueError(f"Invalid input tensor x. Expected shape: ({x.shape[0]}, {self.input_dim}), but got shape: ({x.shape[0]}, {x.shape[1]}).")
        if z_samples.shape[0] != num_components or z_samples.shape[1] != self.latent_dim:
            raise ValueError(f"Invalid input tensor z_samples. Expected shape: ({num_components}, {self.latent_dim}), but got shape: ({z_samples.shape[0]}, {z_samples.shape[1]}).")

        # Choose the optimized z if provided, else use the generated z
        phi_input = self.z if self.z is not None else z_samples
        phi_z = self.phi_net(phi_input)  # Generate parameters for each PC, shape: (num_components, 2 * input_dim)
        
        if self.pc_type == "factorized" and self._can_use_fastpath():
            # Faster computation path for factorized PCs
            return self._fast_forward_factorized(x, phi_z, w, k)
        
        # Create a new list of PCs if the number of components has changed
        if len(self.pcs) != num_components:
            self.pcs = nn.ModuleList([get_probabilistic_circuit(self.pc_type, self.input_dim) for _ in range(num_components)])
        
        # Get the positions of the missing values
        mask = x != -1

        likelihoods = []
        for i in range(num_components):
            self.pcs[i].set_params(phi_z[i])  # Assign PC parameters
            likelihood = self.pcs[i](x, mask)  # Compute p(x | phi(z_i)), shape: (batch_size,)

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
    
    def _can_use_fastpath(self):
        """Check if the fast path for factorized PCs can be used."""
        return True

    def _fast_forward_factorized(self, x, phi_z, w, k=None):
        """
        Vectorized forward pass for fully factorized PCs.
        This functions computes the log likelihoods of all components at once, speeding up the process significantly.
        """
        batch_size = x.shape[0]
        num_components = phi_z.shape[0]
        input_dim = x.shape[1]

        means, log_vars = torch.chunk(phi_z, 2, dim=-1)
        stds = torch.exp(0.5 * log_vars).clamp(min=1e-3)

        if k is not None and k < num_components:
            # First forward pass (no gradients) to find the top k components per sample
            with torch.no_grad():
                x_expanded = x.unsqueeze(0).expand(num_components, -1, -1)  # Shape: (num_components, batch_size, input_dim)
                mean = means.unsqueeze(1)  # Shape: (num_components, 1, input_dim)
                std = stds.unsqueeze(1)  # Shape: (num_components, 1, input_dim)

                log_prob = -0.5 * (((x_expanded - mean) / std) ** 2 + 2 * torch.log(std) + math.log(2 * math.pi))

                # Ignore missing values in the log probability
                if torch.isnan(log_prob).any():
                    log_prob[torch.isnan(log_prob)] = 0.0

                if (x == -1).any():
                    mask = (x != -1).unsqueeze(0)  # Shape: (1, batch_size, input_dim)
                    log_prob = torch.where(mask, log_prob, 0.0)

                log_likelihoods = log_prob.sum(dim=-1)  # Shape: (num_components, batch_size)

                # Add weights
                log_weights = torch.log(w).view(-1, 1)  # Shape: (num_components, 1)
                weighted_log_likelihoods = log_likelihoods + log_weights  # Shape: (num_components, batch_size)

                top_k_values, top_k_indices = torch.topk(weighted_log_likelihoods, k, dim=0)  # Get top K values and indices

            # Second forward pass, over only top k components
            topk_indices = top_k_indices.transpose(0, 1)  # (batch_size, k)

            means_topk = means[topk_indices]  # (batch_size, k, input_dim)
            stds_topk = stds[topk_indices]  # (batch_size, k, input_dim)
            w_topk = w[topk_indices]  # (batch_size, k)

            x_expanded_topk = x.unsqueeze(1)  # Shape: (batch_size, 1, input_dim)

            # Try loop per k?

            log_prob = -0.5 * (((x_expanded_topk - means_topk) / stds_topk) ** 2 + 2 * torch.log(stds_topk) + math.log(
                2 * math.pi))

            # Ignore missing values in the log probability
            if torch.isnan(log_prob).any():
                log_prob[torch.isnan(log_prob)] = 0.0

            if (x == -1).any():
                mask = (x != -1).unsqueeze(1)  # Shape: (batch_size, 1, input_dim)
                log_prob = torch.where(mask, log_prob, 0.0)

            log_lik = log_prob.sum(dim=-1)  # (batch_size, k)
            log_weighted = log_lik + torch.log(w_topk)  # (batch_size, k)
            mixture_log_lik = torch.logsumexp(log_weighted, dim=1)  # (batch_size,)

            return mixture_log_lik.mean()

        else:
            # Full forward pass, no top k selection
            x_expanded = x.unsqueeze(0).expand(num_components, -1, -1)  # Shape: (num_components, batch_size, input_dim)
            means = means.unsqueeze(1)  # Shape: (num_components, 1, input_dim)
            stds = stds.unsqueeze(1)  # Shape: (num_components, 1, input_dim)

            log_prob = -0.5 * (((x_expanded - means) / stds) ** 2 + 2 * torch.log(stds) + math.log(2 * math.pi))

            # Ignore missing values in the log probability
            if torch.isnan(log_prob).any():
                log_prob[torch.isnan(log_prob)] = 0.0

            if (x == -1).any():
                mask = (x != -1).unsqueeze(0)  # Shape: (1, batch_size, input_dim)
                log_prob = torch.where(mask, log_prob, 0.0)

            log_likelihoods = log_prob.sum(dim=-1)  # Shape: (num_components, batch_size)

            # Add weights
            log_weights = torch.log(w).view(-1, 1)  # Shape: (num_components, 1)
            weighted_log_likelihoods = log_likelihoods + log_weights  # Shape: (num_components, batch_size)
            mixture_likelihood = torch.logsumexp(weighted_log_likelihoods,dim=0)
            return torch.mean(mixture_likelihood)

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
    """An SPN PC structure. (Not implemented yet)"""
    def __init__(self, input_dim):
        super().__init__(input_dim)
    
    def set_params(self, params):
        """Set the parameters for the SPN"""
        return 0

    def forward(self, x, ignore_mask=None):
        """SPN computation"""
        return 0

class ChowLiuTreePC(BaseProbabilisticCircuit):
    """A Chow Liu Tree PC structrue. (Not implemented yet)"""
    def __init__(self, input_dim):
        super().__init__(input_dim)
    
    def set_params(self, params):
        """Set the parameters for the CLT"""
        return 0

    def forward(self, x, ignore_mask=None):
        """Placeholder for CLT computation"""
        return 0

def get_probabilistic_circuit(pc_type, input_dim):
    """Factory function for the different PC types."""
    types = ["factorized", "spn", "clt"]
    if pc_type == "factorized":
        return FactorizedPC(input_dim)
    elif pc_type == "spn" or pc_type == "SPN":
        raise NotImplementedError("SPN is not implemented yet.")
        return SPN(input_dim)
    elif pc_type == "clt" or pc_type == "CLT":
        raise NotImplementedError("Chow Liu Tree is not implemented yet.")
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
    def __init__(self, latent_dim, pc_param_dim, pc_type="factorized", net=None, hidden_layers=2, neurons_per_layer=64, activation="ReLU", batch_norm=False, dropout_rate=0.0, skip_layers=False):
        super().__init__()
        out_dim = pc_param_dim * 2
        self.out_dim = out_dim
        self.skip_layers = skip_layers
        if net:
            if not isinstance(net, nn.Sequential):
                raise ValueError(f"Invalid input net. Please provide a Sequential neural network from torch.nn .")
            if not isinstance(net[0], nn.Conv2d) and net[0].in_features != latent_dim:
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
                "relu": nn.ReLU(),
                "tanh": nn.Tanh(),
                "sigmoid": nn.Sigmoid(),
                "leakyrelu": nn.LeakyReLU(),
                "identity": nn.Identity(),
                "": nn.Identity(),      # Empty string is also seen as Identity
                None: nn.Identity(),    # None is also seen as Identity
            }
            if isinstance(activation, str):
                activation = activation.lower()     # Make it case insensitive
            activation_fn = activations_list.get(activation, nn.ReLU())     # Default is ReLU

            # Create the neural network layer by layer
            layers = []
            for i in range(hidden_layers):
                # Set the input and output dimensions
                input_dim = latent_dim if i == 0 else neurons_per_layer[i-1]
                if self.skip_layers and i > 0:   # If we use skiplayers, add the previous layer to the inputs
                    if i == 1:
                        input_dim += latent_dim
                    else:
                        input_dim += neurons_per_layer[i-2]
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
            if self.skip_layers and hidden_layers > 1:
                layers.append(nn.Linear(neurons_per_layer[-2] + neurons_per_layer[-1], out_dim))
            else:
                layers.append(nn.Linear(neurons_per_layer[-1], out_dim))

            # Store the neural network
            self.net = nn.Sequential(*layers)

    def forward(self, z):
        """
        Run z through the neural network.

        Parameters:
            z: Integration points of shape (num_components, latent_dim)

        Returns: 
            phi(z): The pc parameters obtained by running z throuh the neural network, of shape (num_components, pc_param_dim)
        """
        if isinstance(self.net[0], nn.Conv2d):
            z = z.view(z.shape[0], 1, int(z.shape[1]/2), 2)
        elif z.shape[1] != self.net[0].in_features:
            raise ValueError(f"Invalid input to the neural network. Expected shape for z: ({z.shape[0]}, {self.net[0].in_features}), but got shape: ({z.shape[0]}, {z.shape[1]}).")
        
        outputs = []
        x = z
        h_i = 0
        for i in range(len(self.net)):
            if self.skip_layers and isinstance(self.net[i], nn.Linear):
                outputs.append(x)
                if h_i == 0:
                    inp = x
                else:
                    inp = torch.cat([outputs[-1], outputs[-2]], dim=1)
                x = self.net[i](inp)
                h_i += 1
            else:
                x = self.net[i](x)
        
        return x

def generate_rqmc_samples(num_samples, latent_dim, random_state=None, device="cpu"):
    """
    Generates samples using Randomized Quasi Monte Carlo.
    
    Parameters:
        num_samples: The number of samples to generate.
        latent_dim: Dimensionality of the latent variable
        random_state (optional): A random seed for reproducibility.

    Returns:
        z_samples: The sampled values z of shape (num_samples, latent_dim)
        w: The weights for the mixture components
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The balance properties of Sobol' points require n to be a power of 2."
        )
        sampler = qmc.Sobol(d=latent_dim, scramble=True, seed=random_state)
        z_samples = sampler.random(n=num_samples)
        z_samples = torch.tensor(qmc.scale(z_samples, -3, 3), dtype=torch.float32, device=device)  # Scale for Gaussian prior
        w = torch.full(size=(num_samples,), fill_value=1 / num_samples, dtype=torch.float32, device=device)     # Uniform weights
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
        skip_layers=False,
        epochs=100,
        batch_size=32,
        tol=1e-5,
        patience=5, 
        lr=0.001,
        weight_decay=1e-5,
        use_gpu=True,
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
        k (optional): Number of top components to consider for the mixture likelihood.
        lo (optional): Whether to perform latent optimization post training.
        net (optional): A custom neural network.
        hidden_layers (optional): Number of hidden layers in the neural network.
        neurons_per_layer (optional): Number of neurons per layer in the neural network.
        activation (optional): The activation function in the neural network.
        batch_norm (optional): Whether to use batch normalization in the neural network.
        dropout_rate (optional): Dropout rate in the neural network.
        epochs (optional): The number of training loops.
        batch_size (optional): The batch size for training or None if not using batches.
        tol (optional): Tolerance for the convergence criterion.
        patience (optional): Number of epochs to wait if no improvement and then stop the training.
        lr (optional): The learning rate of the optimizer.
        weight_decay (optional): Weight decay for the optimizer.
        use_gpu (optional): Whether to use GPU for computation if available.
        random_state (optional): A random seed for reproducibility. 
        verbose (optional): Verbosity level.

    Returns:
        model: A trained CM-TPM model.
        likelihoods: The computed likelihoods at each epoch during training.
    """
    rng = np.random.default_rng(random_state)  # Random number generator for reproducibility
    input_dim = train_data.shape[1]

    # Disable patience if set to None
    patience = patience if patience is not None else float('inf')

    # Define the model
    model = CM_TPM(pc_type, input_dim, latent_dim, num_components, net=net, 
                   custom_layers=[hidden_layers, neurons_per_layer, activation, batch_norm, dropout_rate, skip_layers], random_state=random_state)

    if verbose > 1:
        print(f"Finished building CM-TPM model with {num_components} components.")

    # Set the optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if verbose > 0:
        print(f"Starting training with {epochs} epochs...")
    prev_loss = -float('inf')       # Initial loss
    best_loss = float('inf')       # Keep track of best loss
    patience_counter = 0            # Early stopping parameter
    start_time = time.time()        # Keep track of training time

    # Use GPU if available
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    if verbose > 0:
        if use_gpu and device == "cpu":
            print("No GPU device detected, using cpu instead")
        else:
            print(f"Using device: {device}")

    # Move model to device
    model = model.to(device)
    model.train()
        
    # Convert training data to tensor
    x_tensor = torch.tensor(train_data, dtype=torch.float32, device=device)  

    # Replace NaN values with -1, so the missing values can be filtered out later while computing the likelihood
    # This is necessary to avoid NaN values in the gradients of the loss function
    mask = ~torch.isnan(x_tensor)       # Get the mask of NaN values
    x_tensor = torch.where(mask, x_tensor, -1)

    likelihoods = []    # Store likelihoods during training       

    if batch_size is not None:
        train_loader = DataLoader(TensorDataset(x_tensor), batch_size=batch_size, shuffle=True)
    else:   # No batches
        train_loader = [torch.unsqueeze(x_tensor, 0)]   # Add batch dimension

    for epoch in tqdm(range(epochs), disable=not verbose == 1, desc="Training"):
        start_time_epoch = time.time()

        total_loss = 0.0       # Keep track of the total loss for the epoch

        for batch in train_loader:  # Iterate over batches
            x_batch = batch[0]      # Extract batch data

            # Generate new z samples and weights
            z_samples, w = generate_rqmc_samples(num_components, latent_dim, random_state=rng.integers(1e9), device=device) 

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
        likelihoods.append(-average_loss)

        # Update best loss and early stopping counter
        if average_loss > best_loss:
            patience_counter += 1
        else:
            best_loss = average_loss
            patience_counter = 0

        # Check early stopping criteria
        if epoch > 10 and (abs(average_loss - prev_loss) < tol or patience_counter > patience):
                if verbose > 1:
                    print(f"Early stopping at epoch {epoch} due to small log likelihood improvement.")
                break
        prev_loss = average_loss
            
        if verbose > 2:
            print(f"Epoch {epoch}, Log-Likelihood: {-average_loss}, Training time: {time.time() - start_time_epoch}")
        elif verbose > 1:
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Log-Likelihood: {-average_loss}')

    if verbose > 0:
        print(f"Training complete.")
        print(f"Final Training Log-Likelihood: {-average_loss}")
    if verbose > 1:
        print(f"Total training time: {time.time() - start_time}")
    model._is_trained = True        # Mark model as trained

    if lo:
        # Optimize z_samples
        model.z = latent_optimization(model, train_loader, num_components_impute, latent_dim, 
                                      epochs=math.ceil(epochs/2), tol=tol, patience=patience, lr=lr, 
                                      weight_decay=weight_decay, random_state=random_state, 
                                      verbose=verbose, device=device).to(device)  
    return model, likelihoods        # Return the trained model

def latent_optimization(
        model,
        train_loader, 
        num_components=None,
        latent_dim=16,
        epochs=100, 
        tol=1e-5,
        patience=5,
        lr=0.01, 
        weight_decay=1e-5,
        random_state=None, 
        verbose=0,
        device="cpu"):
    """
    Optimizes the integration points z after training.

    Parameters:
        model: A trained CM-TPM model.
        train_loader: A DataLoader for the training data.
        num_components (optional): The number of mixture components.
        latent_dim (optional): The dimensionality of the latent variable.
        epochs (optional): The number of optimization loops.
        tol (optional): Tolerance for the convergence criterion.
        patience (optional): Number of epochs to wait if no improvement and then stop the optimization.
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

    # Generate new z samples and weights (move to every epoch?)
    z_samples, w = generate_rqmc_samples(n_components, latent_dim, random_state=random_state, device=device)
    w = w.to(device).detach()  # Detach weights to avoid gradient tracking

    # Make z_samples a parameter to optimize
    z_optimized = torch.nn.Parameter(z_samples.clone().to(device), requires_grad=True)  
    optimizer = optim.Adam([z_optimized], lr=lr, weight_decay=weight_decay)

    full_x = torch.cat([batch[0].to(device) for batch in train_loader], dim=0)  # Concatenate all batches to get the full dataset

    if verbose > 0:
        print(f"Starting latent optimization with {epochs} epochs...")
    prev_loss = -float('inf')       # Initial loss
    best_loss = float('inf')
    patience_counter = 0
    best_z = z_optimized.detach().clone()     # Keep track of the best z_samples
    start_time = time.time()        # Keep track of training time

    model.eval()       # Set model to evaluation mode
    for epoch in tqdm(range(epochs), disable=not verbose == 1, desc="Latent Optimization"):
        start_time_epoch = time.time()

        optimizer.zero_grad()       # Reset gradients
        loss = -model(full_x, z_optimized, w, n_components=n_components)    # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update z_samples

        # Evaluate loss without tracking gradients
        with torch.no_grad():
            eval_loss = -model(full_x, z_optimized, w, n_components=n_components)  

        if eval_loss > best_loss:
            patience_counter += 1
        else:
            best_z = z_optimized.detach().clone()
            best_loss = eval_loss
            patience_counter = 0

        # Check early stopping criteria
        if epoch > 10 and (abs(eval_loss - prev_loss) < tol or patience_counter > patience):
                if verbose > 1:
                    print(f"Early stopping at epoch {epoch} due to small log likelihood improvement.")
                break
        prev_loss = eval_loss

        if verbose > 2:
            print(f"Epoch {epoch}, Log-Likelihood: {-eval_loss}, Training time: {time.time() - start_time_epoch}")
        elif verbose > 1:
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Log-Likelihood: {-eval_loss}')

    if verbose > 0:
        print(f"Latent optimization complete.")
        print(f"Final latent optimization Log-Likelihood: {-best_loss}")
    if verbose > 1:
        print(f"Total latent optimization time: {time.time() - start_time}")
    return best_z  # Return optimized z_samples


def impute_missing_values_exact(
        x_incomplete, 
        model,
        num_components=None,
        epochs=100,
        lr=0.01,
        max_batch_size=512,
        use_gpu=True,
        random_state=None,
        verbose=0,
        skip=False,
        ):
    """
    Imputes missing data using a specified model.
    Imputation is done by optimizing the imputed values with the log-likelihood.
    
    Parameters:
        x_incomplete: The input data with missing values.
        model: A CM-TPM model to use for data imputation.
        num_components (optional): The number of mixture components.
        epochs (optional): The number of imputation loops.
        lr (optional): The learning rate during imputation. 
        max_batch_size (optional): The maximum batch size for imputation or None if not using batches.
        use_gpu (optional): Whether to use GPU for computation if available.
        random_state (optional): A random seed for reproducibility. 
        verbose (optional): Verbosity level.
        skip (optional): Skips the model fitted check, used for EM.

    Returns:
        x_imputed: A copy of x_incomplete with the missing values imputed.
        log_likelihood: The log-likelihood of the imputed data.
        likelihoods: The computed likelihoods at each epoch during inference.
    """
    if not np.isnan(x_incomplete).any():
        return x_incomplete, None, None
    
    if not model._is_trained and not skip:
        raise ValueError("The model has not been fitted yet. Please call the fit method first.")
    
    if x_incomplete.shape[1] != model.input_dim:
        raise ValueError(f"The missing data does not have the same number of features as the training data. Expected features: {model.input_dim}, but got features: {x_incomplete.shape[1]}.")
    
    if verbose > 0:
        print(f"Starting with imputing data...")
    start_time = time.time()

    # Use GPU if available
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    if verbose > 0:
        if use_gpu and device == "cpu":
            print("No GPU device detected, using cpu instead")
        else:
            print(f"Using device: {device}")

    if random_state is not None:
        set_random_seed(random_state)

    # Set the correct amount of components
    n_components = num_components or model.num_components

    # Move model to device
    model = model.to(device)
    model.eval()

    # Generate new samples and weights
    z_samples, w = generate_rqmc_samples(n_components, model.latent_dim, random_state=random_state, device=device)
    
    # Create a tensor with the data to impute
    x_incomplete = torch.tensor(x_incomplete, dtype=torch.float32, device=device)
    x_imputed = x_incomplete.clone()

    # Identify rows with missing values
    nan_mask = torch.isnan(x_incomplete)
    row_mask = torch.any(nan_mask, dim=1)
    nan_indices = torch.where(row_mask)[0]
    total_nan_rows = len(nan_indices)

    if verbose > 0:
        print(f"Found {total_nan_rows} rows with missing values.")

    # If we do not use batches, set batch size to the number of rows with missing values
    if max_batch_size is None or max_batch_size > total_nan_rows:
        iterator = range(1)
        batch_size = total_nan_rows
    else:   # Set correct batch size
        batch_size = max_batch_size
        iterator = tqdm(range(0, total_nan_rows, batch_size), disable=(verbose != 1), desc="Imputing")

    likelihoods = []    # Store the likelihoods during imputation
    final_likelihoods = []       # Store the final likelihoods for each batch


    index = 0
    for start_idx in iterator:
        likelihoods.append([])       # Add list for the batch

        # Get the current batch of rows with missing values
        end_idx = min(start_idx + batch_size, total_nan_rows)
        batch_rows = nan_indices[start_idx:end_idx]
        x_batch = x_incomplete[batch_rows].clone()
        batch_mask = ~torch.isnan(x_batch)

        # Create a tensor for the batch with missing values
        x_batch[~batch_mask] = 0.5      # Initialize missing values to 0.5
        x_fixed = x_batch.clone()
        x_vals = x_batch[~batch_mask].clone().detach().requires_grad_(True)

        optimizer = optim.Adam([x_vals], lr=lr)  # Set optimizer

        # Optimize the missing values
        for epoch in tqdm(range(epochs), disable=(batch_size != total_nan_rows or verbose != 1), desc="Imputation Epoch"):
            optimizer.zero_grad()
            x_batch_imputed = x_fixed.masked_scatter(~batch_mask, x_vals)
            loss = -model(x_batch_imputed, z_samples, w, n_components=n_components)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                x_vals.clamp_(0, 1)
                likelihoods[index].append(-loss.item())     # Store the likelihood at this epoch

            if verbose > 2 and batch_size == total_nan_rows:
                print(f"Epoch {epoch}, Log-likelihood: {-loss.item()}")
            elif epoch % 10 == 0 and verbose > 1 and batch_size == total_nan_rows:
                print(f"Epoch {epoch}, Log-likelihood: {-loss.item()}")
            elif epoch % 10 == 0 and verbose > 2:
                print(f"Batch {start_idx}-{end_idx} | Epoch {epoch}, Log-likelihood: {-loss.item()}")

        if verbose > 1 and batch_size != total_nan_rows:
            print(f"Batch {start_idx}-{end_idx} | Final Log-likelihood: {-loss.item()}")

        # Finalize batch
        with torch.no_grad():
            x_final = x_fixed.masked_scatter(~batch_mask, x_vals)
            x_imputed[batch_rows] = x_final
            final_likelihoods.append(-loss.item())
            index += 1

    if verbose > 0:
        print(f"Finished imputing data.")
        print(f"Succesfully imputed {total_nan_rows} rows.")
        print(f"Final Imputed Data Log-Likelihood: {np.mean(final_likelihoods)}")
    if verbose > 1:
        print(f"Total imputation time: {time.time() - start_time}")

    # Return the imputed data and the final log-likelihood
    return x_imputed.detach().cpu().numpy(), np.mean(final_likelihoods), np.mean(likelihoods, axis=0)  

def impute_missing_values_component(
        x_incomplete, 
        model,
        num_components=None,
        k=None,
        max_batch_size=256,
        use_gpu=True,
        random_state=None,
        verbose=0,
        skip=False,
        ):
    """
    Imputes missing data using a specified model.
    Imputation is done by selecting the most likely component for each sample.
    
    Parameters:
        x_incomplete: The input data with missing values.
        model: A CM-TPM model to use for data imputation.
        num_components (optional): The number of mixture components.
        k (optional): The number of top components to consider for the mixture likelihood. If None, all components are used.
        max_batch_size (optional): The maximum batch size for imputation or None if not using batches.
        use_gpu (optional): Whether to use GPU for computation if available.
        random_state (optional): A random seed for reproducibility. 
        verbose (optional): Verbosity level.
        skip (optional): Skips the model fitted check, used for EM.

    Returns:
        x_imputed: A copy of x_incomplete with the missing values imputed.
        log_likelihood: The log-likelihood of the imputed data.
    """
    if not np.isnan(x_incomplete).any():
        return x_incomplete, None
    
    if not model._is_trained and not skip:
        raise ValueError("The model has not been fitted yet. Please call the fit method first.")
    
    if x_incomplete.shape[1] != model.input_dim:
        raise ValueError(f"The missing data does not have the same number of features as the training data. Expected features: {model.input_dim}, but got features: {x_incomplete.shape[1]}.")
    
    if verbose > 0:
        print(f"Starting with imputing data...")
    start_time = time.time()

    # Use GPU if available
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    if verbose > 0:
        if use_gpu and device == "cpu":
            print("No GPU device detected, using cpu instead")
        else:
            print(f"Using device: {device}")

    if random_state is not None:
        set_random_seed(random_state)

    # Set the correct amount of components
    n_components = num_components or model.num_components

    # Move model to device
    model = model.to(device)
    model.eval()

    # Generate component samples
    z_samples, w = generate_rqmc_samples(n_components, model.latent_dim, random_state=random_state, device=device)
    if model.z is not None:
        z_samples = model.z

    # Convert the input data to a tensor
    x_incomplete_tensor = torch.tensor(x_incomplete, dtype=torch.float32, device=device, requires_grad=False)
    x_imputed_tensor = x_incomplete_tensor.clone().requires_grad_(False)

    # Identify rows with missing values
    missing_full_mask = torch.isnan(x_incomplete_tensor).requires_grad_(False)
    missing_row_mask = torch.isnan(x_incomplete_tensor).any(dim=1).requires_grad_(False)  # shape: (batch_size,)
    x_rows_with_missing = x_incomplete_tensor[missing_row_mask].requires_grad_(False)
    n_incomplete = x_rows_with_missing.shape[0]
    n_features = x_incomplete.shape[1]

    # Forward pass through neural network
    phi_z = model.phi_net(z_samples)  # shape: (C, 2 * F)
    means, log_vars = torch.chunk(phi_z, 2, dim=-1)
    stds = torch.exp(0.5 * log_vars).clamp(min=1e-3)
    means_exp = means.unsqueeze(1).detach()                                       # (n_components, 1, input_dim)
    stds_exp = stds.unsqueeze(1).detach()                                         # (n_components, 1, input_dim)

    log_2pi = math.log(2 * math.pi)

    # If we do not use batches, set batch size to the number of rows with missing values
    if max_batch_size is None or max_batch_size > n_incomplete:
        iterator = range(1)
        batch_size = n_incomplete
    else:   # Set correct batch size
        batch_size = max_batch_size
        iterator = tqdm(range(0, n_incomplete, batch_size), disable=(verbose != 1), desc="Imputing")

    imputed_batches = []
    log_likelihood_total = 0.0

    with torch.no_grad():
        for i in iterator:
            # Get the current batch
            x_batch = x_rows_with_missing[i:i+batch_size].requires_grad_(False) 
            # Build mask for missing values
            missing_mask = torch.isnan(x_batch).requires_grad_(False)           # shape: (batch_size, input_dim)

            # Expand dims for broadcasting
            x_exp = x_batch.unsqueeze(0).expand(n_components, -1, -1).requires_grad_(False)  # (n_components, batch_size, input_dim)
            
            # Compute log probabilities
            diff = (x_exp - means_exp) / stds_exp
            log_probs = -0.5 * (diff ** 2 + 2 * torch.log(stds_exp) + log_2pi)

            # Zero out missing value contributions
            log_probs[torch.isnan(log_probs)] = 0.0

            # Sum over features → log-likelihoods for (n_components, n_missing_rows)
            log_likelihoods = log_probs.sum(dim=-1)

            if k is None:       # Use all components for imputation
                weights = torch.softmax(log_likelihoods, dim=0)  # (n_components, n_missing_rows)
                weights_expanded = weights.unsqueeze(-1)    # (n_components, n_missing_rows, 1)
                best_means = (means_exp * weights_expanded).sum(dim=0)  # (n_missing_rows, input_dim)
            elif k == 1:
                # Find the best component for each sample
                best_components = torch.argmax(log_likelihoods, dim=0)  # (n_missing_rows,)
                best_means = means[best_components]                    # (n_missing_rows, input_dim)
            else:
                # Find the k best components for each sample
                top_k_values, top_k_indices = torch.topk(log_likelihoods, k=k, dim=0)
                top_k_means = means[top_k_indices]                    # (k, n_missing_rows, input_dim)
                
                # Compute the weighted sum of means for the top k components
                weights = torch.softmax(top_k_values, dim=0)  # (k, n_missing_rows)
                weights_expanded = weights.unsqueeze(-1)  # (k, n_missing_rows, 1)
                best_means = (top_k_means * weights_expanded).sum(dim=0)  # (n_missing_rows, input_dim)

            # Fill missing values with the mean from the best component(s)
            batch_imputed = x_batch.clone()
            batch_imputed[missing_mask] = best_means[missing_mask]
            imputed_batches.append(batch_imputed)

            # Compute the final log-likelihood of the imputed data for logging
            with torch.no_grad():
                log_likelihood_total += model(batch_imputed, z_samples, w, n_components=n_components).item()

            del x_batch, missing_mask, x_exp, diff, log_probs, log_likelihoods, best_means, batch_imputed
            gc.collect()
            torch.cuda.empty_cache()

        # Insert imputed rows back into the full tensor
        x_rows_imputed = torch.cat(imputed_batches, dim=0)
        x_imputed_tensor[missing_row_mask] = x_rows_imputed

        # Compute average log-likelihood over the batches
        avg_log_likelihood = log_likelihood_total / (n_incomplete / batch_size)

    if verbose > 0:
        print(f"Finished imputing data.")
        print(f"Successfully imputed {missing_full_mask.sum().item()} missing values across {n_incomplete} samples.")
        print(f"Final imputed data log-likelihood: {avg_log_likelihood}")
    if verbose > 1:    
        print(f"Total imputation time: {time.time() - start_time:.2f}s")

    return x_imputed_tensor.detach().cpu().numpy(), avg_log_likelihood  # Return the imputed data and the log-likelihood

def set_random_seed(seed):
    """Ensure reproducibility by setting random seeds for all libraries."""
    np.random.seed(seed)  # NumPy's random generator
    torch.manual_seed(seed)  # PyTorch CPU random generator
    torch.cuda.manual_seed_all(seed)  # If using GPU

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

    # start_time = time.time()
    # all_zeros = np.full((100, 10), 0.89)
    # all_zeros[50, 3] = np.nan
    # all_zeros[10, 2] = np.nan
    # all_zeros[92, 0] = np.nan
    # model, train_likelihoods = train_cm_tpm(all_zeros, 
    #                      pc_type="factorized", 
    #                      verbose=1, 
    #                      epochs=100, 
    #                      lr=0.001, 
    #                      tol=1e-5,
    #                      patience=5,
    #                      num_components=256,
    #                      num_components_impute=512,
    #                      k=None,
    #                      lo=True,
    #                      batch_size=None,
    #                      random_state=0
    #                      )
    # imputed, likelihood = impute_missing_values_component(all_zeros, model, num_components=512, k=1, verbose=1, random_state=0)
    # #imputed, likelihood, impute_likelihoods = impute_missing_values_exact(all_zeros, model, num_components=512, verbose=0, random_state=0)
    # print(imputed[50, 3])
    # print(imputed[10, 2])
    # print(imputed[92, 0])
    # print("Log-Likelihood:", likelihood)
    # print("Imputation time:", time.time() - start_time)

    # print(train_likelihoods)
    # print(impute_likelihoods)

    net = PhiNet(4, 10, hidden_layers=4, neurons_per_layer=[64, 128, 256, 512], skip_layers=True)
    z = torch.tensor([[0.5, 0.1, 0.4, 0.9]])

    x = net(z)
    print(x)
