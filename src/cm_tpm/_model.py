from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
import numpy as np
from scipy.stats import qmc     # For RQMC sampling
import networkx as nx

# TODO:
#   Include log_w from the rqmc sampler
#   Add settable mean and variance in rqmc sampler
#   Incorporate model parameters in _cm.py
#   Dealing with missing values in training data
#   Data preprocessing 
#       - scale numerical features
#       - handle categorical features
#       - handle binary features
#   Add ways to use custom nets
#   Add PC structure(s) -> PCs, CLTs, ...
#   Improve Neural Network PhiNet
#   Optimize latent selection (top-K selection)
#   Implement latent optimization for fine-tuning integration points
#   Do some testing with accuracy/log likelihood etc.
#   Make hyperparameters tunable
#   Choose optimal standard hyperparameters

class CM_TPM(nn.Module):
    def __init__(self, pc_type, input_dim, latent_dim, num_components):
        """
        pc_type: Type of PC to use ("factorized", "spn", "clt").
        input_dim: Dimensionality of input data.
        latent_dim: Dimensionality of latent variable z.
        num_components: Number of mixture components (integration points).
        """
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_components = num_components

        # Neural network to generate PC parameters
        self.phi_net = PhiNet(latent_dim, input_dim)

        # Create multiple PCs (one per component)
        self.pcs = nn.ModuleList([get_probabilistic_circuit(pc_type, input_dim) for _ in range(num_components)])

        self._is_trained = False
    
    def forward(self, x, z_samples):
        """
        Compute the mixture likelihood.
        x: Input batch of shape (batch_size, input_dim).
        z_samples: Integration points of shape (num_components, latent_dim).
        """
        if x.shape[1] != self.input_dim:
            raise ValueError(f"Invalid input tensor x. Expected shape: ({x.shape[0]}, {self.input_dim}), but got shape: ({x.shape[0]}, {x.shape[1]}).")
        if z_samples.shape[0] != self.num_components or z_samples.shape[1] != self.latent_dim:
            raise ValueError(f"Invalid input tensor z_samples. Expected shape: ({self.num_components}, {self.latent_dim}), but got shape: ({z_samples.shape[0]}, {z_samples.shape[1]}).")

        phi_z = self.phi_net(z_samples)  # Generate parameters for each PC, shape: (num_components, input_dim)
        
        likelihoods = []
        for i in range(self.num_components):
            self.pcs[i].set_params(phi_z[i])  # Assign PC parameters
            likelihood = self.pcs[i](x)  # Compute p(x | phi(z_i)), shape: (batch_size)
            
            if torch.isnan(likelihood).any():
                raise ValueError(f"NaN detected in likelihood at component {i}: {likelihood}")

            likelihoods.append(likelihood)

        likelihoods = torch.stack(likelihoods, dim=0)  # Shape: (num_components, batch_size)
        weights = 1.0 / self.num_components  # Uniform weights
        mixture_likelihood = torch.sum(weights * likelihoods, dim=0)  # Sum over components, shape: (batch_size)
        return mixture_likelihood.mean()  # Average over batch

#  Base Probabilistic Circuit, other PC structures inherit from this class
class BaseProbabilisticCircuit(nn.Module, ABC):
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

# Factorized PC implementation
class FactorizedPC(BaseProbabilisticCircuit):
    def __init__(self, input_dim):
        super().__init__(input_dim)
        self.params = None
    
    def set_params(self, params):
        """Set the parameters for the factorized PC"""
        self.params = params

    def forward(self, x):
        """Compute the likelihood using a factorize Gaussian model"""
        if self.params is None:
            raise ValueError("PC parameters are not set. Call set_params(phi_z) first.")
        if self.params.shape[0] != x.shape[1]:
            raise ValueError(f"The size of x and the size of params do not match. Expected shape for params: ({x.shape[1]}), but got shape: ({self.params.shape[0]}).")
        
        return torch.exp(-torch.sum((x - self.params) ** 2, dim=-1))  # Gaussian-like PC

# SPN implementation
class SPN(BaseProbabilisticCircuit):
    def __init__(self, input_dim):
        super().__init__(input_dim)
        self.params = None
    
    def set_params(self, params):
        """Set the parameters for the SPN"""
        self.params = params

    def forward(self, x):
        """Placeholder for SPN computation"""
        if self.params is None:
            raise ValueError("PC parameters are not set. Call set_params(phi_z) first.")
        return torch.exp(-torch.sum((x - self.params) ** 2, dim=-1))  # Gaussian-like PC

# CLT implementation
class ChowLiuTreePC(BaseProbabilisticCircuit):
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

    def fit_tree(self, data):
        n_features = data.shape[1]
        mutual_info_matrix = np.zeros((n_features, n_features))

        for i in range(n_features):
            for j in range(i + 1, n_features):
                mi = np.corrcoef(data[:, i], data[:, j])[0, 1]
                mutual_info_matrix[i, j] = mutual_info_matrix[j, i] = mi

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

# PC factory function
def get_probabilistic_circuit(pc_type, input_dim):
    types = ["factorized", "spn", "clt"]
    if pc_type == "factorized":
        return FactorizedPC(input_dim)
    elif pc_type == "spn":
        return SPN(input_dim)
    elif pc_type == "clt":
        return ChowLiuTreePC(input_dim)
    else:
        raise ValueError(f"Unknown PC type: '{pc_type}', use one of the following types: {types}")

# Neural Network for Parameter Mapping: phi(z) -> PC parameters
class PhiNet(nn.Module):
    def __init__(self, latent_dim, pc_param_dim, net=None):
        super().__init__()
        out_dim = pc_param_dim
        if net:
            if net[0].in_features != latent_dim:
                raise ValueError(f"Invalid input net. The first layer should have {latent_dim} input features, but is has {net[0].in_features} input features.")
            if net[-1].out_features != out_dim:
                raise ValueError(f"Invalid input net. The final layer should have {out_dim} output features, but is has {net[-1].out_features} output features.")
            self.net = net
        else:
            self.net = nn.Sequential(
                nn.Linear(latent_dim, 64),
                nn.ReLU(),
                nn.Linear(64, out_dim),
            )

    def forward(self, z):
        """
        Run z through the neural network.
        z: Integration points of shape (num_components, latent_dim)
        Returns: A Torch tensor of shape (num_components, pc_param_dim)
        """
        if z.shape[1] != self.net[0].in_features:
            raise ValueError(f"Invalid input to the neural network. Expected shape for z: ({z.shape[0]}, {self.net[0].in_features}), but got shape: ({z.shape[0]}, {z.shape[1]}).")
        return self.net(z)

# Generate RQMC Samples
def generate_rqmc_samples(num_samples, latent_dim):
    sampler = qmc.Sobol(d=latent_dim, scramble=True)
    z_samples = sampler.random(n=num_samples)
    z_samples = torch.tensor(qmc.scale(z_samples, -3, 3), dtype=torch.float32)  # Scale for Gaussian prior
    return z_samples

# Training Loop
def train_cm_tpm(train_data, pc_type="factorized", latent_dim=4, num_components=256, epochs=100, lr=0.001):
    if np.isnan(train_data).any():
        raise ValueError("NaN detected in training data. The training data cannot have missing values.")
    
    input_dim = train_data.shape[1]
    model = CM_TPM(pc_type, input_dim, latent_dim, num_components)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(epochs):
        z_samples = generate_rqmc_samples(num_components, latent_dim)
        x_batch = torch.tensor(train_data, dtype=torch.float32)

        optimizer.zero_grad()
        loss = -model(x_batch, z_samples)

        if torch.isnan(loss).any():
            raise ValueError(f"NaN detected in loss at epoch {epoch}: {loss}")

        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                raise ValueError(f"NaN detected in gradient of {name} at epoch {epoch}")
       
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Log-Likelihood: {-loss.item()}')

    model._is_trained = True
    return model

# Missing Data Imputation
def impute_missing_values(x_incomplete, model):
    if not np.isnan(x_incomplete).any():
        return x_incomplete
    
    if not model._is_trained:
        raise ValueError("The model has not been fitted yet. Please call the fit method first.")
    
    if x_incomplete.shape[1] != model.input_dim:
        raise ValueError(f"The missing data does not have the same number of features as the training data. Expected features: {model.input_dim}, but got features: {x_incomplete.shape[1]}.")
    
    z_samples = generate_rqmc_samples(model.num_components, model.latent_dim)
    x_incomplete = torch.tensor(x_incomplete, dtype=torch.float32)
    mask = ~torch.isnan(x_incomplete)
    x_filled = torch.where(mask, x_incomplete, torch.tensor(0.0, dtype=torch.float32))

    likelihoods = []
    for i in range(z_samples.shape[0]):
        model.pcs[i].set_params(model.phi_net(z_samples)[i])
        likelihood = model.pcs[i](x_filled)

        marginalized_likelihood = torch.where(mask, likelihood.unsqueeze(-1).expand_as(mask), torch.mean(likelihood).expand_as(mask))
        likelihoods.append(marginalized_likelihood)

    likelihoods = torch.stack(likelihoods, dim=0)
    imputed_values = torch.mean(likelihoods, dim=0)

    imputed_values_final = torch.where(mask, x_incomplete, imputed_values)
    return imputed_values_final.detach().cpu().numpy()

# Example Usage
if __name__ == '__main__':
    train_data = np.random.rand(1000, 10)
    model = train_cm_tpm(train_data, pc_type="factorized")

    x_incomplete = train_data[:3].copy()
    x_incomplete[0, 0] = np.nan
    x_incomplete[2, 9] = np.nan
    x_imputed = impute_missing_values(x_incomplete, model)
    print("Original Data:", train_data[:3])
    print("Data with missing:", x_incomplete)
    print("Imputed values:", x_imputed)

    # test_x = torch.randn(5, 10)  # Small test batch
    # test_pc = ChowLiuTreePC(input_dim=10)
    # test_params = torch.randn(10 * 2)  # Simulated params
    # test_pc.set_params(test_params)

    # print("Running test forward pass...")
    # likelihood = test_pc.forward(test_x)
    # print("Likelihood output:", likelihood)
