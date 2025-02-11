from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.stats import qmc     # For RQMC sampling

# TODO:
#   Add PC structure(s) -> PCs, CLTs, ...
#   Improve Neural Network PhiNet
#   Optimize latent selection (top-K selection)
#   Implement latent optimization for fine-tuning integration points
#   Do some testing with accuracy/log likelihood etc.
#   Make hyperparameters tunable
#   Choose optimal standard hyperparameters
#   When everything works -> Implementation with cm-tpm package

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
        self.model = None
    
    def set_params(self, params):
        """Set the parameters for the CLT"""
        self.params = params

    def forward(self, x):
        """Placeholder for CLT computation"""
        if self.params is None:
            raise ValueError("PC parameters are not set. Call set_params(phi_z) first.")
        return torch.exp(-torch.sum((x - self.params) ** 2, dim=-1))  # Gaussian-like PC

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
    def __init__(self, latent_dim, pc_param_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, pc_param_dim),
        )

    def forward(self, z):
        return self.net(z)

# Generate RQMC Samples
def generate_rqmc_samples(num_samples, latent_dim):
    sampler = qmc.Sobol(d=latent_dim, scramble=True)
    z_samples = sampler.random(n=num_samples)
    z_samples = torch.tensor(qmc.scale(z_samples, -3, 3), dtype=torch.float32)  # Scale for Gaussian prior
    return z_samples

# Compute Log-Likelihood with Numerical Integration
def compute_log_likelihood(x_batch, phi_net, pc, z_samples):
    phi_z = phi_net(z_samples)
    likelihoods = []
    for i in range(z_samples.shape[0]):
        pc.set_params(phi_z[i])
        likelihood = pc(x_batch)
        likelihoods.append(likelihood.requires_grad_(True))

    likelihoods = torch.stack(likelihoods, dim=0)
    weights = 1.0 / z_samples.shape[0]  # Uniform RQMC weights
    return torch.log(torch.sum(likelihoods * weights, dim=0) + 1e-9).mean()

# Training Loop
def train_cm_tpm(train_data, pc_type="factorized", latent_dim=4, num_integration_points=256, epochs=100, lr=0.01):
    phi_net = PhiNet(latent_dim, train_data.shape[1])
    pc = get_probabilistic_circuit(pc_type, train_data.shape[1])
    optimizer = optim.Adam(list(phi_net.parameters()), lr=lr)

    for epoch in range(epochs):
        z_samples = generate_rqmc_samples(num_integration_points, latent_dim)
        x_batch = torch.tensor(train_data, dtype=torch.float32)

        optimizer.zero_grad()
        loss = -compute_log_likelihood(x_batch, phi_net, pc, z_samples)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Log-Likelihood: {-loss.item()}')

    return phi_net, pc

# Missing Data Imputation
def impute_missing_values(x_incomplete, phi_net, pc, num_integration_points=256):
    z_samples = generate_rqmc_samples(num_integration_points, phi_net.net[0].in_features)
    phi_z = phi_net(z_samples)

    mask = ~torch.isnan(x_incomplete)

    x_filled = torch.where(mask, x_incomplete, torch.tensor(0.0, dtype=torch.float32))

    likelihoods = []
    for i in range(z_samples.shape[0]):
        pc.set_params(phi_z[i])
        likelihood = pc(x_filled)

        marginalized_likelihood = torch.where(mask, likelihood.unsqueeze(-1).expand_as(mask), torch.mean(likelihood).expand_as(mask))
        likelihoods.append(marginalized_likelihood)

    likelihoods = torch.stack(likelihoods, dim=0)
    imputed_values = torch.mean(likelihoods, dim=0)

    imputed_values_final = torch.where(mask, x_incomplete, imputed_values)
    return imputed_values_final

# Example Usage
if __name__ == '__main__':
    train_data = np.random.rand(1000, 10)
    phi_net, pc = train_cm_tpm(train_data, pc_type="clt")

    x_incomplete = train_data[:3].copy()
    x_incomplete[0, 0] = np.nan
    x_incomplete[2, 9] = np.nan
    x_imputed = impute_missing_values(torch.tensor(x_incomplete, dtype=torch.float32), phi_net, pc)
    print("Original Data:", train_data[:3])
    print("Data with missing:", x_incomplete)
    print("Imputed values:", x_imputed)
