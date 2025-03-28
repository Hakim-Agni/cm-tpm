import math
import numpy as np
import pandas as pd
import torch.nn as nn
import warnings
from ._model import train_cm_tpm, impute_missing_values, impute_missing_values_exact
from ._helpers import (
    _load_file, _to_numpy, _restore_format, _missing_to_nan, _all_numeric, 
    _integer_encoding, _restore_encoding, _binary_encoding, _restore_binary_encoding
)

import os

os.environ['OMP_NUM_THREADS'] = '1'

class CMImputer:
    """
    Imputation for completing missing values using Continuous Mixtures of 
    Tractable Probabilistic Models.

    Parameters
    ----------
    missing_values: float, optional (default=np.nan)
        The placeholder for missing values in the input data, all instances of missing_values will be imputed.
    n_components: int, optional (default=8)
        Number of components to use in the mixture model.
    latent_dim: int, optional (default=16)
        Dimensionality of the latent variable.
    pc_type: str, optional (default="factorized"), allowed: "factorized", "spn", "clt"
        The type of PC to use in the model.
    missing_strategy: str, optional (default="mean"), allowed: "mean", "zero", "em", "ignore"
        The strategy to use for missing data in the training data. 
    ordinal_features: dict, optional (default=None)
        A dictionaty containing information on which features have ordinal data and how the values are mapped.
    max_depth: int, optional (default=5)
        Maximum depth of the probabilistic circuits.
    custom_net: nn.Sequential, optional (default=None)
        A custom neural network to use in the model.
    hidden_layers: int, optional (default=2)
        The number of hidden layers in the neural network.
    neuron_per_layer: int or list of ints, optional (default=64)
        The number of neuron in each layer in the neural network.
    activation: str, optional (default="ReLU"), allowed: "ReLU", "Tanh", "Sigmoid", "LeakyReLU"
        The activation function used in the neural network.
    batch_norm: bool, optional (default=False)
        Whether to use batch normalization in the neural network.
    dropout_rate: float, optional (default=0.0)
        The dropout rate used in the neural network.
    max_iter: int, optional (default=100)
        Maximum number of iterations to perform.
    tol: float, optional (default=1e-4)
        Tolerance for the convergence criterion.
    lr:  float, optional (default=0.001)
        The learning rate for the optimizer
    smooth: float, optional (default=1e-6)
        Smoothing parameter to avoid division by zero.
    random_state: int, RandomState instance or None, optional (default=None)
        Random seed for reproducibility.
    verbose: int, optional (default=0)
        Verbosity level, controls the debug messages.
    copy: bool, optional (default=True)
        Whether to copy the input data or modify it in place.
    keep_empty_features: bool, optional (default=False)
        Whether to keep features that only have missing values in the imputed dataset.

    Attributes
    ----------
    is_fitted_: bool
        Whether the model is fitted.
    n_features_in_: int
        Number of features in the input data.
    feature_names_in_: list of str
        Names of the input features.
    components_: list
        List of trained components in the mixture model.
    log_likelihood_: float
        Log likelihood of the data under the model.
    mean_: float
        The mean value for each feature observed during training.
    std_: float
        The standard deviation for each feature observed during training.
    binary_info_: list
        The information about binary features observed during training.
    categorical_info_: list
        The information about categorical features observed during training.
    random_state_: RandomState instance
        RandomState instance that is generated from a seed or a random number generator.

    References
    ----------
    ...

    Examples
    --------
    ...
    """
    def __init__(
            self,
            missing_values: int | float | str | None = np.nan,
            n_components: int = 8,
            latent_dim: int = 16,
            pc_type: str = "factorized",
            missing_strategy: str = "mean",
            ordinal_features: dict = None,
            max_depth: int = 5,
            custom_net: nn.Sequential = None,
            hidden_layers: int = 2,
            neurons_per_layer: int | list = 64,
            activation: str = "ReLU",
            batch_norm: bool = False,
            dropout_rate: float = 0.0,
            max_iter: int = 100,
            tol: float = 1e-4,
            lr: float = 0.001,
            smooth: float = 1e-6, 
            random_state: int = None,
            verbose: int = 0,
            copy: bool = True,
            keep_empty_features: bool = False,
        ):
        """Initialize the CMImputer instance."""
        # Mixture model
        self.model = None
        # Parameters
        self.missing_values = missing_values
        self.n_components = n_components
        self.latent_dim = latent_dim
        self.pc_type = pc_type
        self.missing_strategy = missing_strategy
        self.ordinal_features = ordinal_features
        self.custom_net = custom_net
        self.hidden_layers = hidden_layers
        self.neurons_per_layer = neurons_per_layer
        self.activation = activation
        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate
        self.max_depth = max_depth
        self.max_iter = max_iter
        self.tol = tol
        self.lr = lr
        self.smooth = smooth
        self.random_state = random_state
        self.verbose = verbose
        self.copy = copy
        self.keep_empty_features = keep_empty_features

        # Attributes
        self.is_fitted_ = False
        self.n_features_in_ = None
        self.feature_names_in_ = None
        self.components_ = None
        self.log_likelihood_ = None
        self.min_vals_ = 0.0
        self.max_vals_ = 1.0
        self.binary_info_ = None
        self.encoding_info_ = None
        self.bin_encoding_info_ = None
        self.random_state_ = np.random.RandomState(self.random_state) if self.random_state is not None else np.random

    def fit(self, X: str | np.ndarray | pd.DataFrame | list, sep=",", decimal=".") -> "CMImputer":
        """
        Fit the imputation model to the input dataset

        Parameters:
            X (array-like or str): Input data with missing values.
                - Allowed: np.ndarray, pd.DataFrame, list of lists, or a file path (CSV, XLSX, Parquest, Feather)
            sep (str, optional): Delimiter for CSV files.
            decimal (str, optional): Decimal separator for CSV files.
                
        Returns:
            self (CMImputer): Fitted instance.        
        """
        # If the input data is a string (filepath), load the data from the file
        if isinstance(X, str):
            X = _load_file(X, sep=sep, decimal=decimal)
        # Transform the data to a NumPy array
        X, _, feature_names = _to_numpy(X)
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = feature_names

        # Fit the model using X
        X_preprocessed, self.binary_info_, self.encoding_info_ = self._preprocess_data(X, train=True)
        self.model = train_cm_tpm(
            X_preprocessed, 
            pc_type=self.pc_type,
            latent_dim=self.latent_dim, 
            num_components=self.n_components, 
            missing_strategy=self.missing_strategy,
            net=self.custom_net,
            hidden_layers=self.hidden_layers,
            neurons_per_layer=self.neurons_per_layer,
            activation=self.activation,
            batch_norm=self.batch_norm,
            dropout_rate=self.dropout_rate,
            epochs=self.max_iter,
            tol=self.tol, 
            lr=self.lr,
            smooth=self.smooth,
            random_state=self.random_state,
            verbose=self.verbose,
            )

        self.is_fitted_ = True
        return self
    
    def transform(self, X: str | np.ndarray | pd.DataFrame | list, save_path: str = None, sep=",", decimal="."):
        """
        Impute missing values in the dataset.

        Parameters:
            X (array-like or str): Data with missing values.
                - Allowed: np.ndarray, pd.DataFrame, list of lists, or a file path (CSV, XLSX, Parquest, Feather)
            save_path (str, optional): If provided, saves output to a file. Otherwise, if X is a filepath, save output to 'X + _imputed'.
            sep (str, optional): Delimiter for CSV files.
            decimal (str, optional): Decimal separator for CSV files.
            
        Returns:
            X_imputed (array-like, same type as X): Dataset with missing values replaced. If X is a filepath, X imputed will be a NumPy array.
        """
        if not self.is_fitted_:  # Check if the model is fitted
            raise ValueError("The model has not been fitted yet. Please call the fit method first.")
        
        file_in = None      # Variable to store the input file path
        # If the input data is a string (filepath), load the data from the file
        if isinstance(X, str):
            file_in = X
            X = _load_file(X, sep=sep, decimal=decimal)
        # Transform the data to a NumPy array
        X_np, original_format, columns = _to_numpy(X)
        if file_in:
            original_format = "ndarray"
        # Perfom imputation
        X_imputed = self._impute(X_np)
        # Transform the imputed data to the original format
        result = _restore_format(X_imputed, original_format, columns)
        
        # If save_path is set, save the imputed data to a file
        if save_path or file_in:
            if not save_path:
                save_path = file_in[:file_in.rfind(".")] + "_imputed" + file_in[file_in.rfind("."):]
            if save_path.endswith(".csv"):
                pd.DataFrame(result, columns=columns).to_csv(save_path, index=False)
            elif save_path.endswith(".xlsx"):
                pd.DataFrame(result, columns=columns).to_excel(save_path, index=False, engine="openpyxl")
            elif save_path.endswith(".parquet"):
                pd.DataFrame(result, columns=columns).to_parquet(save_path)
            elif save_path.endswith(".feather"):
                pd.DataFrame(result, columns=columns).to_feather(save_path)
            else:
                raise ValueError("Unsupported file format for saving.")

        # Return the imputed data
        return result
    
    def fit_transform(self, X: str | np.ndarray | pd.DataFrame | list, save_path:str = None, sep=",", decimal="."):
        """
        Fit the model and then transform the data.

        Parameters:
            X (array-like or str): Data with missing values.
                - Allowed: np.ndarray, pd.DataFrame, list of lists, or a file path (CSV, XLSX, Parquest, Feather)
            save_path (str, optional): If provided, saves output to a file. Otherwise, if X is a filepath, save output to 'X + _imputed'.
            sep (str, optional): Delimiter for CSV files.
            decimal (str, optional): Decimal separator for CSV files.
            
        Returns:
            X_imputed (array-like, same type as X): Dataset with missing values replaced. If X is a filepath, X imputed will be a NumPy array.
        """
        return self.fit(X, sep=sep, decimal=decimal).transform(X, save_path, sep=sep, decimal=decimal)
    
    def get_feature_names_out(input_features=None):
        """
        Get output feature names.

        Parameters:
            input_features (list of str or None): Input feature names.
        """
        # TODO: Implement feature names
        return 0
    
    def get_params(self):
        """
        Get parameters for this CMImputer.
        """
        return {
            "missing_values": self.missing_values,
            "n_components": self.n_components,
            "latent_dim": self.latent_dim,
            "pc_type": self.pc_type,
            "missing_strategy": self.missing_strategy,
            "ordinal_features": self.ordinal_features,
            "max_depth": self.max_depth,
            "custom_net": self.custom_net,
            "hidden_layers": self.hidden_layers,
            "neurons_per_layer": self.neurons_per_layer,
            "activation": self.activation,
            "batch_norm": self.batch_norm,
            "dropout_rate": self.dropout_rate,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "lr": self.lr,
            "smooth": self.smooth,
            "random_state": self.random_state,
            "verbose": self.verbose,
            "copy": self.copy,
            "keep_empty_features": self.keep_empty_features
        }
    
    def set_params(self, **params):
        """
        Set parameters for this CMImputer.

        Parameters:
            **params: Parameters to set.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self
    
    def evaluate(self, X: str | np.ndarray | pd.DataFrame | list):
        """
        Evaluate how well the model explains the data.

        Parameters:
            X (array-like or str): Data with missing values.
                - Allowed: np.ndarray, pd.DataFrame, list of lists, or a file path (CSV, XLSX, Parquest, Feather)

        Returns:
            log_likelihood_ (float): Log likelihood of the data under the model.
        """
        # If the input data is a string (filepath), load the data from the file
        if isinstance(X, str):
            X = _load_file(X)
        # Transform the data to a NumPy array
        X, _, _ = _to_numpy(X)

        if not self.is_fitted_:
            raise ValueError("The model has not been fitted yet. Please call the fit method first.")
        
        # Evaluate the model using X
        # TODO
        return 0.0
    
    def _check_consistency(self, X: np.ndarray):
        """Ensures that the input data is consistent with the training data."""
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Mismatch in number of features. Expected {self.n_features_in_}, got {X.shape[1]}.")
        
        mask, info = self.encoding_info_
        for i in range(X.shape[1]):
            if mask[i]:     # Feature is categorical
                if _all_numeric(X[:, i]):
                    raise ValueError(f"Feature {i} was categorical during training but numeric in new data.")
                
                enc_info = info[i]
                # TODO: Keep this feature? Model is not trained to recognize unseen values...
                # If there are new non-binary feature values in the input data, update the encoding info
                next_index = max(enc_info.values(), default=-1) + 1
                for val in X[:, i]:
                    if val not in enc_info and not self.binary_info_[i] and val != "nan" and next_index.bit_count() != 1:
                        warnings.warn(f"New categorical value detected in column {i}: '{val}'. The model has not been trained with this value.")
                        enc_info[val] = next_index
                        next_index += 1

                # Apply the same encoding
                X[:, i] = [enc_info[val] if val in enc_info else np.nan for val in X[:, i]]

            else:       # Feature is numeric
                if not _all_numeric(X[:, i]):
                    raise ValueError(f"Feature {i} was numeric during training but categorical in new data.")

        return X.astype(float), mask, info

    
    def _preprocess_data(self, X: np.ndarray, train: bool = False):
        """Preprocess the input data before imputation."""
        if self.copy:
            X_transformed = X.copy()

        # Change all instances of 'missing_value' to NaN
        X_transformed = _missing_to_nan(X, self.missing_values)

        # Convert non-floats (e.g. strings) to floats using encoding
        if train:
            X_transformed, encoding_mask, encoding_info = _integer_encoding(X_transformed, ordinal_features=self.ordinal_features)
        else:
            X_transformed, encoding_mask, encoding_info = self._check_consistency(X_transformed)
        self.encoding_info_ = (encoding_mask, encoding_info)

        if self.keep_empty_features:
            # Fill columns that consist of only NaN with 0 
            empty_features = np.where(np.all(np.isnan(X_transformed), axis=0))[0]
            X_transformed[:, empty_features] = 0
        else:
            # Remove columns that consist of only NaN
            if train:       # If we are in training process, update feature names and number of features
                self.feature_names_in_ = None if self.feature_names_in_ is None else self.feature_names_in_[~np.all(np.isnan(X_transformed), axis=0)]
                X_transformed = X_transformed[:, ~np.all(np.isnan(X_transformed), axis=0)]
                self.n_features_in_ = X_transformed.shape[1]
            elif self.n_features_in_ and X_transformed.shape[1] != self.n_features_in_:     # Only remove features if they were also removed during training
                X_transformed = X_transformed[:, ~np.all(np.isnan(X_transformed), axis=0)]

        # Apply binary encoding to encoded features
        X_transformed, bin_info = _binary_encoding(X_transformed, encoding_mask, encoding_info, ordinal_features=self.ordinal_features)
        self.bin_encoding_info_ = bin_info

        # Check which features are binary features (only consisting of 0s and 1s)
        binary_mask = np.array([
            np.isin(np.unique(X_transformed[:, i][~np.isnan(X_transformed[:, i])]), [0, 1]).all()
            for i in range(X_transformed.shape[1])
        ])

        if train:       # Update the means and stds only during training
            min_vals = np.nanmin(X_transformed, axis=0)
            self.min_vals_ = np.where(np.isnan(min_vals), 0.0, min_vals)
            max_vals = np.nanmax(X_transformed, axis=0)
            self.max_vals_ = np.where(np.isnan(max_vals), 1.0, max_vals)
        
        # Scale the data
        scale = self.max_vals_ - self.min_vals_
        scale[scale == 0] = 1e-9
        X_scaled = (X_transformed - self.min_vals_) / scale
        X_scaled[:, binary_mask] = X_transformed[:, binary_mask]    # Keep binary columns unscaled

        return X_scaled.astype(float), binary_mask, (encoding_mask, encoding_info)
    
    def _impute(self, X: np.ndarray) -> np.ndarray:
        """Impute missing values in input X"""
        X_in = X.copy()
        X_nan = _missing_to_nan(X, self.missing_values)
        X_preprocessed, _, _ = self._preprocess_data(X, train=False)

        if not np.any(np.isnan(X_preprocessed)):
            warnings.warn(f"No missing values detected in input data, transformation has no effect. Did you set the correct missing value: '{self.missing_values}'?")

        X_imputed = impute_missing_values_exact(
            X_preprocessed, 
            self.model,
            epochs=100,
            lr=0.01,
            random_state=self.random_state,
            verbose = self.verbose,
        )
        
        # Scale the data back to the original
        scale = self.max_vals_ - self.min_vals_
        scale[scale == 0] = 1e-9
        X_scaled = X_imputed * scale + self.min_vals_

        # Round the binary features to the nearest option
        X_scaled[:, self.binary_info_] = np.round(X_imputed[:, self.binary_info_])

        # Decode the non-numerical features
        encoding_mask, encoding_info = self.encoding_info_
        X_decoded = _restore_binary_encoding(X_scaled, self.bin_encoding_info_, X_imputed)
        X_decoded = _restore_encoding(X_decoded, encoding_mask, encoding_info)

        # Make sure the original values remain the same
        try:
            mask = ~np.isnan(X_nan)
        except TypeError:
            mask = X_nan != "nan"
        X_filled = np.where(mask, X_in, X_decoded)
        return X_filled
