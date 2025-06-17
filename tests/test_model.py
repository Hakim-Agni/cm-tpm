import pytest
from cm_tpm._model import CM_TPM, FactorizedPC, SPN, ChowLiuTreePC
from cm_tpm._model import PhiNet
from cm_tpm._model import get_probabilistic_circuit, generate_rqmc_samples, train_cm_tpm, impute_missing_values_optimization, impute_missing_values_component
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
            num_components=64,
        )

    def test_instance(self):
        """Test the instantiation of the CM_TPM class."""
        assert self.model is not None

    def test_parameters(self):
        """Test the class parameters."""
        assert self.model.pc_type == "factorized"
        assert self.model.input_dim == 20
        assert self.model.latent_dim == 10
        assert self.model.num_components == 64
        assert self.model._is_trained is False
        assert isinstance(self.model.phi_net, PhiNet)

    def test_invalid_pc_type(self):
        """Test instantiating a model with an invalid PC type."""
        try:
            model = CM_TPM(
                pc_type="some pc",
                input_dim=20,
                latent_dim=10,
                num_components=64
            )
            x = torch.tensor(np.random.rand(100, 20), dtype=torch.float32)
            z_samples = torch.tensor(np.random.rand(64, 10), dtype=torch.float32)
            w = torch.tensor(np.random.rand(64))
            likelihood = model(x, z_samples, w)
            assert False
        except ValueError as e:
            assert str(e).startswith("Unknown PC type: 'some pc'")

    def test_forward(self):
        """"Test the forward function of the model."""
        x = torch.tensor(np.random.rand(100, 20), dtype=torch.float32)
        z_samples = torch.tensor(np.random.rand(64, 10), dtype=torch.float32)
        w = torch.tensor(np.random.rand(64))
        likelihood = self.model(x, z_samples, w)
        assert isinstance(likelihood, torch.Tensor)
        assert likelihood.shape == torch.Size([])
        assert isinstance(likelihood.item(), float)

    def test_forward_wrong_dimensions_x(self):
        """""Test the forward function with invalid dimensions for x."""
        x = torch.tensor(np.random.rand(100, 30), dtype=torch.float32)
        z_samples = torch.tensor(np.random.rand(64, 10), dtype=torch.float32)
        w = torch.tensor(np.random.rand(64))
        try:
            likelihood = self.model(x, z_samples, w)
            assert False
        except ValueError as e:
            assert str(e) == "Invalid input tensor x. Expected shape: (100, 20), but got shape: (100, 30)."

    def test_forward_wrong_dimensions_z(self):
        """""Test the forward function with invalid dimensions for z."""
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
        """Test the instantiation of a Factorized PC class."""
        assert self.pc is not None
        assert self.pc.params is None

    def test_set_params(self):
        """Test setting the parameters of the PC."""
        params = torch.tensor(np.random.rand(20))
        self.pc.set_params(params)
        assert torch.equal(self.pc.params, params)

    def test_forward(self):
        """Test the forward method of the PC."""
        x = torch.tensor(np.random.rand(50, 20), dtype=torch.float32)
        params = torch.tensor(np.random.rand(40))
        self.pc.set_params(params)
        likelihoods = self.pc(x)
        assert isinstance(likelihoods, torch.Tensor)
        assert likelihoods.shape == torch.Size([50])

    def test_forward_no_params(self):
        """Test the forward method when the parameters have not been set."""
        x = torch.tensor(np.random.rand(50, 20), dtype=torch.float32)
        try:
            likelihoods = self.pc(x)
            assert False
        except ValueError as e:
            assert str(e) == "PC parameters are not set. Call set_params(phi_z) first."

    def test_forward_wrong_param_dimensions(self):
        """Test the forward method when the data and parameter dimensions do not match"""
        x = torch.tensor(np.random.rand(50, 20), dtype=torch.float32)
        params = torch.tensor(np.random.rand(20))
        self.pc.set_params(params)
        try:
            likelihoods = self.pc(x)
            assert False
        except ValueError as e:
            assert str(e) == "Expected params of shape (40,), got torch.Size([20])"

class TestSPN():
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup method for the test class."""
        self.pc = SPN(input_dim=20)
    
    # Add more tests when implementing


class TestCLT():
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup method for the test class."""
        self.pc = ChowLiuTreePC(input_dim=20)
    
    # Add more tests when implementing

class TestPCFactory():
    def test_get_factorized(self):
        """Test getting a Factorized PC."""
        pc = get_probabilistic_circuit("factorized", 20)
        assert isinstance(pc, FactorizedPC)

    def test_get_spn(self):
        """Test getting a SPN."""
        try:
            pc = get_probabilistic_circuit("spn", 20)
            assert isinstance(pc, SPN)
        except NotImplementedError as e:
            assert str(e) == "SPN is not implemented yet."

    def test_get_clt(self):
        """Test getting a Chow Liu Tree PC."""
        try:
            pc = get_probabilistic_circuit("clt", 20)
            assert isinstance(pc, ChowLiuTreePC)
        except NotImplementedError as e:
            assert str(e) == "Chow Liu Tree is not implemented yet."

    def test_invalid(self):
        """Test getting an invalid PC."""
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
        """Test the instantiation of the neural network."""
        assert self.neural_net is not None 

    def test_parameters(self):
        """Test the parameters of the default neural network."""
        assert isinstance(self.neural_net.net, nn.Sequential)
        assert len(self.neural_net.net) == 5
        assert self.neural_net.net[0].in_features == 20 and self.neural_net.net[0].out_features == 64
        assert isinstance(self.neural_net.net[1], nn.ReLU)
        assert self.neural_net.net[2].in_features == 64 and self.neural_net.net[2].out_features == 64
        assert isinstance(self.neural_net.net[3], nn.ReLU)
        assert self.neural_net.net[4].in_features == 64 and self.neural_net.net[4].out_features == 20

    def test_custom_net(self):
        """Test setting a custom neural network."""
        net = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 20),
        )
        neural_net = PhiNet(latent_dim=20, pc_param_dim=10, net=net)
        assert isinstance(neural_net.net, nn.Sequential)
        assert len(neural_net.net) == 5
        assert neural_net.net[0].in_features == 20 and neural_net.net[0].out_features == 64
        assert isinstance(neural_net.net[1], nn.ReLU)
        assert neural_net.net[2].in_features == 64 and neural_net.net[2].out_features == 256
        assert isinstance(neural_net.net[3], nn.ReLU)
        assert neural_net.net[4].in_features == 256 and neural_net.net[4].out_features == 20

    def test_invalid_custom_net_in_features(self):
        """Test setting a custom neural network with invalid input features."""
        net = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 20),
        )
        try:
            neural_net = PhiNet(latent_dim=20, pc_param_dim=10, net=net)
            assert False
        except ValueError as e:
            assert str(e) == "Invalid input net. The first layer should have 20 input features, but is has 10 input features."

    def test_invalid_custom_net_out_features(self):
        """Test setting a custom neural network with invalid output features."""
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
            assert str(e) == "Invalid input net. The final layer should have 20 output features, but is has 30 output features."

    def test_forward(self):
        """Test the forward function on the neural network."""
        z = torch.tensor(np.random.rand(100, 20), dtype=torch.float32)
        out = self.neural_net(z)
        assert isinstance(out, torch.Tensor)
        assert out.shape == torch.Size([100, 20])

    def test_forward_wrong_dimensions(self):
        """Test putting a tensor with the wrong dimensions into the network."""
        z = torch.tensor(np.random.rand(100, 40), dtype=torch.float32)
        try:
            out = self.neural_net(z)
            assert False
        except ValueError as e:
            assert str(e) == "Invalid input to the neural network. Expected shape for z: (100, 20), but got shape: (100, 40)."
        
    def test_custom_params_simple(self):
        """Test setting simple custom parameters for a neural network."""
        neural_net = PhiNet(latent_dim=20, pc_param_dim=10, hidden_layers=3, neurons_per_layer=128, activation="Tanh", batch_norm=True, dropout_rate=0.1)
        assert isinstance(neural_net.net, nn.Sequential)
        assert len(neural_net.net) == 13
        assert neural_net.net[0].in_features == 20 and neural_net.net[0].out_features == 128
        assert isinstance(neural_net.net[1], nn.BatchNorm1d)
        assert isinstance(neural_net.net[2], nn.Tanh)
        assert isinstance(neural_net.net[3], nn.Dropout)
        assert neural_net.net[4].in_features == 128 and neural_net.net[4].out_features == 128
        assert isinstance(neural_net.net[5], nn.BatchNorm1d)
        assert isinstance(neural_net.net[6], nn.Tanh)
        assert isinstance(neural_net.net[7], nn.Dropout)
        assert neural_net.net[8].in_features == 128 and neural_net.net[8].out_features == 128
        assert isinstance(neural_net.net[9], nn.BatchNorm1d)
        assert isinstance(neural_net.net[10], nn.Tanh)
        assert isinstance(neural_net.net[11], nn.Dropout)
        assert neural_net.net[12].in_features == 128 and neural_net.net[12].out_features == 20

    def test_custom_params_complex(self):
        """Test setting more complex custom parameters for a neural network."""
        neural_net = PhiNet(latent_dim=20, pc_param_dim=10, hidden_layers=3, neurons_per_layer=[64, 128, 256], activation="Sigmoid", batch_norm=True, dropout_rate=0.1)
        assert isinstance(neural_net.net, nn.Sequential)
        assert len(neural_net.net) == 13
        assert neural_net.net[0].in_features == 20 and neural_net.net[0].out_features == 64
        assert isinstance(neural_net.net[1], nn.BatchNorm1d)
        assert isinstance(neural_net.net[2], nn.Sigmoid)
        assert isinstance(neural_net.net[3], nn.Dropout)
        assert neural_net.net[4].in_features == 64 and neural_net.net[4].out_features == 128
        assert isinstance(neural_net.net[5], nn.BatchNorm1d)
        assert isinstance(neural_net.net[6], nn.Sigmoid)
        assert isinstance(neural_net.net[7], nn.Dropout)
        assert neural_net.net[8].in_features == 128 and neural_net.net[8].out_features == 256
        assert isinstance(neural_net.net[9], nn.BatchNorm1d)
        assert isinstance(neural_net.net[10], nn.Sigmoid)
        assert isinstance(neural_net.net[11], nn.Dropout)
        assert neural_net.net[12].in_features == 256 and neural_net.net[12].out_features == 20

    def test_custom_params_invalid(self):
        """Test setting invalid custom parameters for a neural network."""
        try:
            neural_net = PhiNet(latent_dim=20, pc_param_dim=10, hidden_layers=3, neurons_per_layer=[64, 128], activation="Sigmoid", batch_norm=True, dropout_rate=0.1)
        except ValueError as e:
            assert str(e) == "The hidden layers and neurons per layer do not match. Hidden layers: 3, neurons per layer: [64, 128]"

class TestRQMS():
    def test_rqmc(self):
        """Test the function that generates z and w using RQMC."""
        z, w = generate_rqmc_samples(num_samples=32, latent_dim=10)
        assert isinstance(z, torch.Tensor)
        assert z.shape == torch.Size([32, 10])
        assert isinstance(w, torch.Tensor)
        assert w.shape == torch.Size([32])

class TestTrainCM_TPM():
    def test_train_dafault(self):
        """Test training data with a default model."""
        train_data = np.random.rand(100, 10)
        model, _ = train_cm_tpm(train_data=train_data, epochs=5)
        assert isinstance(model, CM_TPM)

    def test_train_parameters(self):
        """Test training data with a model with different parameters."""
        train_data = np.random.rand(100, 10)
        model, _ = train_cm_tpm(train_data=train_data, pc_type="factorized", latent_dim=6, num_components=64, k=5, epochs=5, lr=0.01)
        assert isinstance(model, CM_TPM)
        assert model.input_dim == 10
        assert model.latent_dim == 6
        assert model.num_components == 64

    def test_train_missing_values(self):
        """Test training data with missing values."""
        train_data = np.random.rand(100, 10)
        train_data[0, 0] = np.nan
        model, _ = train_cm_tpm(train_data=train_data, epochs=5)
        assert isinstance(model, CM_TPM)

    def test_train_batches(self):
        """Test training data with batches."""
        train_data = np.random.rand(100, 10)
        model, _ = train_cm_tpm(train_data=train_data, batch_size=32, epochs=5)
        assert isinstance(model, CM_TPM)

    def test_train_no_batches(self):
        """Test training data without batches."""
        train_data = np.random.rand(100, 10)
        model, _ = train_cm_tpm(train_data=train_data, batch_size=None, epochs=5)
        assert isinstance(model, CM_TPM)

    def test_train_lo(self):
        """Test training with latent optimization."""
        train_data = np.random.rand(100, 10)
        model, _ = train_cm_tpm(train_data=train_data, lo=True, epochs=5)
        assert isinstance(model, CM_TPM)

class TestImputeOptimization():
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup method for the test class."""
        self.train_data = np.random.rand(100, 10)
        self.model, _ = train_cm_tpm(train_data=self.train_data, epochs=5)

    def test_impute_data(self):
        """Test imputing data with missing values."""
        data_incomplete = np.random.rand(30, 10)
        data_incomplete[4, 6] = np.nan
        data_incomplete[25, 1] = np.nan
        data_incomplete[0, 7] = np.nan
        data_imputed, _, _ = impute_missing_values_optimization(data_incomplete, self.model, epochs=5)
        assert isinstance(data_imputed, np.ndarray)
        assert data_imputed.shape == data_incomplete.shape
        assert data_imputed[4, 6] != np.nan
        assert data_imputed[25, 1] != np.nan
        assert data_imputed[0, 7] != np.nan
        assert not np.isnan(data_imputed).any()

    def test_impute_lo(self):
        """Test imputing data with latent optimization."""
        data_incomplete = np.random.rand(30, 10)
        data_incomplete[4, 6] = np.nan
        data_incomplete[25, 1] = np.nan
        data_incomplete[0, 7] = np.nan
        model, _ = train_cm_tpm(train_data=self.train_data, lo=True, epochs=5)
        data_imputed, _, _ = impute_missing_values_optimization(data_incomplete, model, epochs=5)
        assert isinstance(data_imputed, np.ndarray)
        assert data_imputed.shape == data_incomplete.shape
        assert data_imputed[4, 6] != np.nan
        assert data_imputed[25, 1] != np.nan
        assert data_imputed[0, 7] != np.nan
        assert not np.isnan(data_imputed).any()

    def test_impute_data_no_missing(self):
        """Test imputing data with no missing values."""
        data_incomplete = np.random.rand(30, 10)
        data_imputed, _, _ = impute_missing_values_optimization(data_incomplete, self.model)
        assert data_imputed.shape == data_incomplete.shape
        assert np.array_equal(data_incomplete, data_imputed)

    def test_impute_data_different_dimension(self):
        """Test imputing data with a different dimension than the training data."""
        data_incomplete = np.random.rand(50, 5)
        data_incomplete[0, 0] = np.nan
        try:
            data_imputed, _, _ = impute_missing_values_optimization(data_incomplete, self.model)
            assert False
        except ValueError as e:
            assert str(e) == "The missing data does not have the same number of features as the training data. Expected features: 10, but got features: 5."

    def test_impute_data_no_training(self):
        """Test imputing data using a model that has not been trained."""
        model = CM_TPM("factorized", 10, 5, 32)
        data_incomplete = np.random.rand(50, 10)
        data_incomplete[0, 0] = np.nan
        try:
            data_imputed, _, _ = impute_missing_values_optimization(data_incomplete, model)
            assert False
        except ValueError as e:
            assert str(e) == "The model has not been fitted yet. Please call the fit method first."

class TestImputeComponent():
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup method for the test class."""
        self.train_data = np.random.rand(100, 10)
        self.model, _ = train_cm_tpm(train_data=self.train_data, epochs=5)

    def test_impute_data(self):
        """Test imputing data with missing values."""
        data_incomplete = np.random.rand(30, 10)
        data_incomplete[4, 6] = np.nan
        data_incomplete[25, 1] = np.nan
        data_incomplete[0, 7] = np.nan
        data_imputed, _ = impute_missing_values_component(data_incomplete, self.model)
        assert isinstance(data_imputed, np.ndarray)
        assert data_imputed.shape == data_incomplete.shape
        assert data_imputed[4, 6] != np.nan
        assert data_imputed[25, 1] != np.nan
        assert data_imputed[0, 7] != np.nan
        assert not np.isnan(data_imputed).any()

    def test_impute_data_k(self):
        """Test imputing data using top k components."""
        data_incomplete = np.random.rand(30, 10)
        data_incomplete[4, 6] = np.nan
        data_incomplete[25, 1] = np.nan
        data_incomplete[0, 7] = np.nan
        data_imputed, _ = impute_missing_values_component(data_incomplete, self.model, k=5)
        assert isinstance(data_imputed, np.ndarray)
        assert data_imputed.shape == data_incomplete.shape
        assert data_imputed[4, 6] != np.nan
        assert data_imputed[25, 1] != np.nan
        assert data_imputed[0, 7] != np.nan
        assert not np.isnan(data_imputed).any()

    def test_impute_data_k1(self):
        """Test imputing data using top 1 components."""
        data_incomplete = np.random.rand(30, 10)
        data_incomplete[4, 6] = np.nan
        data_incomplete[25, 1] = np.nan
        data_incomplete[0, 7] = np.nan
        data_imputed, _ = impute_missing_values_component(data_incomplete, self.model, k=1)
        assert isinstance(data_imputed, np.ndarray)
        assert data_imputed.shape == data_incomplete.shape
        assert data_imputed[4, 6] != np.nan
        assert data_imputed[25, 1] != np.nan
        assert data_imputed[0, 7] != np.nan
        assert not np.isnan(data_imputed).any()

    def test_impute_lo(self):
        """Test imputing data with latent optimization."""
        data_incomplete = np.random.rand(30, 10)
        data_incomplete[4, 6] = np.nan
        data_incomplete[25, 1] = np.nan
        data_incomplete[0, 7] = np.nan
        model, _ = train_cm_tpm(train_data=self.train_data, lo=True, epochs=5)
        data_imputed, _ = impute_missing_values_component(data_incomplete, model)
        assert isinstance(data_imputed, np.ndarray)
        assert data_imputed.shape == data_incomplete.shape
        assert data_imputed[4, 6] != np.nan
        assert data_imputed[25, 1] != np.nan
        assert data_imputed[0, 7] != np.nan
        assert not np.isnan(data_imputed).any()

    def test_impute_data_no_missing(self):
        """Test imputing data with no missing values."""
        data_incomplete = np.random.rand(30, 10)
        data_imputed, _ = impute_missing_values_component(data_incomplete, self.model)
        assert data_imputed.shape == data_incomplete.shape
        assert np.array_equal(data_incomplete, data_imputed)

    def test_impute_data_different_dimension(self):
        """Test imputing data with a different dimension than the training data."""
        data_incomplete = np.random.rand(50, 5)
        data_incomplete[0, 0] = np.nan
        try:
            data_imputed, _ = impute_missing_values_component(data_incomplete, self.model)
            assert False
        except ValueError as e:
            assert str(e) == "The missing data does not have the same number of features as the training data. Expected features: 10, but got features: 5."

    def test_impute_data_no_training(self):
        """Test imputing data using a model that has not been trained."""
        model = CM_TPM("factorized", 10, 5, 32)
        data_incomplete = np.random.rand(50, 10)
        data_incomplete[0, 0] = np.nan
        try:
            data_imputed, _ = impute_missing_values_component(data_incomplete, model)
            assert False
        except ValueError as e:
            assert str(e) == "The model has not been fitted yet. Please call the fit method first."

class TestModelResult():
    def test_cm_factorized_zeros(self):
        """Test imputing data filled with zeros."""
        p = 0.05
        all_zeros = np.zeros((100, 10))
        all_zeros[0, 0] = np.nan
        model, _ = train_cm_tpm(all_zeros, random_state=42)
        imputed, log_likelihood, _ = impute_missing_values_optimization(all_zeros, model, random_state=42)
        assert imputed[0, 0] < p
        assert log_likelihood > 0

    def test_cm_factorized_constant(self):
        """Test imputing data filled with a constant."""
        constant = 0.42
        p = 0.1
        all_const = np.full((100, 10), constant)
        all_const[43, 8] = np.nan
        all_const[10, 2] = np.nan
        all_const[84, 0] = np.nan
        model, _ = train_cm_tpm(all_const, random_state=42)
        imputed, log_likelihood, _ = impute_missing_values_optimization(all_const, model, num_components=1024, random_state=42)
        assert imputed[43, 8] < constant + p and imputed[43, 8] > constant - p
        assert imputed[10, 2] < constant + p and imputed[10, 2] > constant - p
        assert imputed[84, 0] < constant + p and imputed[84, 0] > constant - p
        assert log_likelihood > 0
