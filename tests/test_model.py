import pytest
from cm_tpm._model import CM_TPM, FactorizedPC, SPN, ChowLiuTreePC
from cm_tpm._model import PhiNet
from cm_tpm._model import get_probabilistic_circuit, generate_rqmc_samples, train_cm_tpm, impute_missing_values
import numpy as np
import torch
import torch.nn as nn

class TestCM_TPM():
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup method for the test class."""
        self.model = CM_TPM(
            pc_type="factorized",
            input_dim=20,
            latent_dim=10,
            num_components=64
        )

    def test_instance(self):
        """Test the instantiation of the CM_TPM class"""
        assert self.model is not None

    def test_parameters(self):
        """Test the class parameters"""
        assert isinstance(self.model.pcs, nn.ModuleList)
        assert len(self.model.pcs) == 64
        assert isinstance(self.model.pcs[0], FactorizedPC)
        assert self.model.input_dim == 20
        assert self.model.pcs[0].input_dim == 20
        assert self.model.latent_dim == 10
        assert self.model.num_components == 64
        assert isinstance(self.model.phi_net, PhiNet)

    def test_invalid_pc_type(self):
        """Test instantiating a model with an invalid PC type"""
        try:
            model = CM_TPM(
                pc_type="some pc",
                input_dim=20,
                latent_dim=10,
                num_components=64
            )
            assert False
        except ValueError as e:
            assert str(e).startswith("Unknown PC type: 'some pc'")

    def test_forward(self):
        """"Test the forward function of the model"""
        x = torch.tensor(np.random.rand(100, 20), dtype=torch.float32)
        z_samples = torch.tensor(np.random.rand(64, 10), dtype=torch.float32)
        likelihood = self.model(x, z_samples)
        assert isinstance(likelihood, torch.Tensor)
        assert likelihood.shape == torch.Size([])
        assert isinstance(likelihood.item(), float)

    def test_forward_wrong_dimensions_x(self):
        """""Test the forward function with invalid dimensions for x"""
        x = torch.tensor(np.random.rand(100, 30), dtype=torch.float32)
        z_samples = torch.tensor(np.random.rand(64, 10), dtype=torch.float32)
        try:
            likelihood = self.model(x, z_samples)
            assert False
        except ValueError as e:
            assert str(e) == "Invalid input tensor x. Expected shape: (100, 20), but got shape: (100, 30)."

    def test_forward_wrong_dimensions_z(self):
        """""Test the forward function with invalid dimensions for z"""
        x = torch.tensor(np.random.rand(100, 20), dtype=torch.float32)
        z_samples = torch.tensor(np.random.rand(16, 25), dtype=torch.float32)
        try:
            likelihood = self.model(x, z_samples)
            assert False
        except ValueError as e:
            assert str(e) == "Invalid input tensor z_samples. Expected shape: (64, 10), but got shape: (16, 25)."

class TestFactorizedPC():
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup method for the test class."""
        self.pc = FactorizedPC(input_dim=20)

    def test_instance(self):
        """Test the instantiation of a Factorized PC class"""
        assert self.pc is not None
        assert self.pc.params is None

    def test_set_params(self):
        """Test setting the parameters of the PC"""
        params = torch.tensor(np.random.rand(20))
        self.pc.set_params(params)
        assert torch.equal(self.pc.params, params)

    def test_forward(self):
        """Test the forward method of the PC"""
        x = torch.tensor(np.random.rand(50, 20), dtype=torch.float32)
        params = torch.tensor(np.random.rand(20))
        self.pc.set_params(params)
        likelihoods = self.pc(x)
        assert isinstance(likelihoods, torch.Tensor)
        assert likelihoods.shape == torch.Size([50])

    def test_forward_no_params(self):
        """Test the forward method when the parameters have not been set"""
        x = torch.tensor(np.random.rand(50, 20), dtype=torch.float32)
        try:
            likelihoods = self.pc(x)
            assert False
        except ValueError as e:
            assert str(e) == "PC parameters are not set. Call set_params(phi_z) first."

    def test_forward_wrong_param_dimensions(self):
        """Test the forward method when the data and parameter dimensions do not match"""
        x = torch.tensor(np.random.rand(50, 25), dtype=torch.float32)
        params = torch.tensor(np.random.rand(20))
        self.pc.set_params(params)
        try:
            likelihoods = self.pc(x)
            assert False
        except ValueError as e:
            assert str(e) == "The size of x and the size of params do not match. Expected shape for params: (25), but got shape: (20)."

class TestSPN():
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup method for the test class."""
        self.pc = SPN(input_dim=20)
    
    # TODO: Add tests


class TestCLT():
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup method for the test class."""
        self.pc = ChowLiuTreePC(input_dim=20)
    
    # TODO: Add tests

class TestPCFactory():
    def test_get_factorized(self):
        """Test getting a Factorized PC"""
        pc = get_probabilistic_circuit("factorized", 20)
        assert isinstance(pc, FactorizedPC)

    def test_get_spn(self):
        """Test getting a SPN"""
        pc = get_probabilistic_circuit("spn", 20)
        assert isinstance(pc, SPN)

    def test_get_clt(self):
        """Test getting a Chow Liu Tree PC"""
        pc = get_probabilistic_circuit("clt", 20)
        assert isinstance(pc, ChowLiuTreePC)

    def test_invalid(self):
        """Test getting an invalid PC"""
        try:
            pc = get_probabilistic_circuit("Does not exist", 20)
            assert False
        except ValueError as e:
            assert str(e).startswith("Unknown PC type: 'Does not exist'")

# TODO: Add tests for Neural Network and all other functions
