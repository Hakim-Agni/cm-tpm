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
        w = torch.tensor(np.random.rand(64))
        likelihood = self.model(x, z_samples, w)
        assert isinstance(likelihood, torch.Tensor)
        assert likelihood.shape == torch.Size([])
        assert isinstance(likelihood.item(), float)

    def test_forward_wrong_dimensions_x(self):
        """""Test the forward function with invalid dimensions for x"""
        x = torch.tensor(np.random.rand(100, 30), dtype=torch.float32)
        z_samples = torch.tensor(np.random.rand(64, 10), dtype=torch.float32)
        w = torch.tensor(np.random.rand(64))
        try:
            likelihood = self.model(x, z_samples, w)
            assert False
        except ValueError as e:
            assert str(e) == "Invalid input tensor x. Expected shape: (100, 20), but got shape: (100, 30)."

    def test_forward_wrong_dimensions_z(self):
        """""Test the forward function with invalid dimensions for z"""
        x = torch.tensor(np.random.rand(100, 20), dtype=torch.float32)
        z_samples = torch.tensor(np.random.rand(16, 25), dtype=torch.float32)
        w = torch.tensor(np.random.rand(16))
        try:
            likelihood = self.model(x, z_samples, w)
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

class TestNeuralNet():
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup method for the test class."""
        self.neural_net = PhiNet(latent_dim=20, pc_param_dim=10)

    def test_instance(self):
        """Test the instantiation of the neural network"""
        assert self.neural_net is not None 

    def test_parameters(self):
        """Test the parameters of the neural network"""
        assert isinstance(self.neural_net.net, nn.Sequential)
        assert self.neural_net.net[0].in_features == 20
        assert self.neural_net.net[-1].out_features == 10

    def test_custom_net(self):
        """Test setting a custom neural network"""
        net = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )
        neural_net = PhiNet(latent_dim=20, pc_param_dim=10, net=net)
        assert isinstance(self.neural_net.net, nn.Sequential)
        assert len(neural_net.net) == 5
        assert neural_net.net[0].in_features == 20 and neural_net.net[0].out_features == 64
        assert isinstance(neural_net.net[1], nn.ReLU)
        assert neural_net.net[2].in_features == 64 and neural_net.net[2].out_features == 256
        assert isinstance(neural_net.net[3], nn.ReLU)
        assert neural_net.net[4].in_features == 256 and neural_net.net[4].out_features == 10

    def test_invalid_custom_net_in_features(self):
        """Test setting a custom neural network with invalid input features"""
        net = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )
        try:
            neural_net = PhiNet(latent_dim=20, pc_param_dim=10, net=net)
            assert False
        except ValueError as e:
            assert str(e) == "Invalid input net. The first layer should have 20 input features, but is has 10 input features."

    def test_invalid_custom_net_out_features(self):
        """Test setting a custom neural network with invalid output features"""
        net = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 30),
        )
        try:
            neural_net = PhiNet(latent_dim=20, pc_param_dim=10, net=net)
            assert False
        except ValueError as e:
            assert str(e) == "Invalid input net. The final layer should have 10 output features, but is has 30 output features."

    def test_forward(self):
        """Test the forward function on the neural network"""
        z = torch.tensor(np.random.rand(100, 20), dtype=torch.float32)
        out = self.neural_net(z)
        assert isinstance(out, torch.Tensor)
        assert out.shape == torch.Size([100, 10])

    def test_forward_wrong_dimensions(self):
        """Test putting a tensor with the wrong dimensions into the network"""
        z = torch.tensor(np.random.rand(100, 40), dtype=torch.float32)
        try:
            out = self.neural_net(z)
            assert False
        except ValueError as e:
            assert str(e) == "Invalid input to the neural network. Expected shape for z: (100, 20), but got shape: (100, 40)."
        
class TestRQMS():
    def test_rqmc(self):
        """Test the function that generates z and w using RQMC"""
        z, w = generate_rqmc_samples(num_samples=32, latent_dim=10)
        assert isinstance(z, torch.Tensor)
        assert z.shape == torch.Size([32, 10])
        assert isinstance(w, torch.Tensor)
        assert w.shape == torch.Size([32])

class TestTrainCM_TPM():
    def test_train_dafault(self):
        """Test training data with a default model"""
        train_data = np.random.rand(100, 10)
        model = train_cm_tpm(train_data=train_data)
        assert isinstance(model, CM_TPM)

    def test_train_parameters(self):
        """Test training data with a model with different parameters"""
        train_data = np.random.rand(100, 10)
        model = train_cm_tpm(train_data=train_data, pc_type="spn", latent_dim=6, num_components=64, epochs=50, lr=0.01)
        assert isinstance(model, CM_TPM)
        assert model.input_dim == 10
        assert model.latent_dim == 6
        assert model.num_components == 64

    def test_train_missing_values(self):
        """Test training data with missing values"""
        train_data = np.random.rand(100, 10)
        train_data[0, 0] = np.nan
        try:
            model = train_cm_tpm(train_data=train_data, pc_type="spn", latent_dim=6, num_components=64, epochs=50, lr=0.01)
            assert False
        except ValueError as e:
            assert str(e) == "NaN detected in training data. The training data cannot have missing values."

class TestImpute():
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup method for the test class."""
        self.train_data = np.random.rand(100, 10)
        self.model = train_cm_tpm(train_data=self.train_data)

    def test_impute_data(self):
        """Test imputing data with missing values"""
        data_incomplete = np.random.rand(30, 10)
        data_incomplete[4, 6] = np.nan
        data_incomplete[25, 1] = np.nan
        data_incomplete[0, 7] = np.nan
        data_imputed = impute_missing_values(data_incomplete, self.model)
        assert isinstance(data_imputed, np.ndarray)
        assert data_imputed.shape == data_incomplete.shape
        assert data_imputed[4, 6] != np.nan
        assert data_imputed[25, 1] != np.nan
        assert data_imputed[0, 7] != np.nan
        assert not np.isnan(data_imputed).any()

    def test_impute_data_no_missing(self):
        """Test imputing data with no missing values"""
        data_incomplete = np.random.rand(30, 10)
        data_imputed = impute_missing_values(data_incomplete, self.model)
        assert data_imputed.shape == data_incomplete.shape
        assert np.array_equal(data_incomplete, data_imputed)

    def test_impute_data_different_dimension(self):
        """Test imputing data with a different dimension than the training data"""
        data_incomplete = np.random.rand(50, 5)
        data_incomplete[0, 0] = np.nan
        try:
            data_imputed = impute_missing_values(data_incomplete, self.model)
            assert False
        except ValueError as e:
            assert str(e) == "The missing data does not have the same number of features as the training data. Expected features: 10, but got features: 5."

    def test_impute_data_no_training(self):
        """Test imputing data using a model that has not been trained"""
        model = CM_TPM("factorized", 10, 5, 32)
        data_incomplete = np.random.rand(50, 10)
        data_incomplete[0, 0] = np.nan
        try:
            data_imputed = impute_missing_values(data_incomplete, model)
            assert False
        except ValueError as e:
            assert str(e) == "The model has not been fitted yet. Please call the fit method first."
