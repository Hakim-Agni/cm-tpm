import pytest
from cm_tpm._model import CM_TPM, FactorizedPC, SPN, ChowLiuTreePC
from cm_tpm._model import PhiNet
from cm_tpm._model import get_probabilistic_circuit, generate_rqmc_samples, train_cm_tpm, impute_missing_values
import numpy as np
import torch
import torch.nn as nn

class TestCM_TPM:
    def test_instance(self):
        """Test the instantiation of the CM_TPM class"""
        model = CM_TPM("factorized", 10, 5, 16)
        assert model is not None

    def test_parameters(self):
        """Test the class parameters"""
        model = CM_TPM(
            pc_type="spn",
            input_dim=20,
            latent_dim=10,
            num_components=64
        )
        assert isinstance(model.pcs, nn.ModuleList)
        assert len(model.pcs) == 64
        assert isinstance(model.pcs[0], SPN)
        assert model.pcs[0].input_dim == 20
        assert model.latent_dim == 10
        assert model.num_components == 64
        assert isinstance(model.phi_net, PhiNet)

    def test_forward(self):
        model = CM_TPM(
            pc_type="factorized",
            input_dim=20,
            latent_dim=10,
            num_components=64
        )
        x = torch.tensor(np.random.rand(100, 20), dtype=torch.float32)
        z = torch.tensor(np.random.rand(64, 10), dtype=torch.float32)
        likelihood = model(x, z)
        assert isinstance(likelihood, torch.Tensor)
        print(likelihood.shape)
        #assert likelihood.shape == (1)
        


a = TestCM_TPM
a.test_forward(a)
