import math
import time
import numpy as np
import pandas as pd
import torch.nn as nn
import warnings
from ._model import train_cm_tpm, impute_missing_values_exact, impute_missing_values_component
from ._helpers import (
    _load_file, _to_numpy, _restore_format, _missing_to_nan, _all_numeric, is_valid_integer,
    _integer_encoding, _restore_encoding, _binary_encoding, _restore_binary_encoding
)

class CMImputer:
    """
    Imputation for completing missing values using Continuous Mixtures of 
    Tractable Probabilistic Models.

    Parameters
    ----------
    settings: string, optional (default="custom"), allowed: "custom", "fast", "balanced", "precise"
        The hyperparameter settings to use for the model.
        - custom: allows custom hyperparameters by setting them manually.
        - fast: Quick imputation, acceptable quality. Use this option when speed is most important.
        - balanced: Trade-off between speed and accuracy. Good for general use. Default parameters use the balanced option.
        - precise: Highest quality, slowest. Use this option when accuracy matters most.
    missing_values: float, string, int, list, optional (default=np.nan)
        The placeholder(s) for missing values in the input data, all instances of missing_values will be imputed.
    n_components_train: int, optional (default=256)
        Number of components to use in the mixture model during training.
    n_components_impute: int, optional (default=2048)
        Number of components to use in the mixture model during imputation. If none, the same number of components as used during training.
    latent_dim: int, optional (default=4)
        Dimensionality of the latent variable.
    top_k: int, optional (default=None)
        The number of components to use for efficient learning. If None, all components are used.
    lo: bool, optional (default=False)
        Whether to use latent optimization after training.
    pc_type: str, optional (default="factorized"), allowed: "factorized", "spn", "clt"
        The type of PC to use in the model.
    imputation_method: str, optional (default="EM") allowed: "EM", "exact"
        The imputation method to use during inference.
        - EM: Imputes values by maximizing the expected values, fast method.
        - Exact: Imputed values by finding the optimal values using an optimizer, slow method.
    ordinal_features: dict, optional (default=None)
        A dictionaty containing information on which features have ordinal data and how the values are mapped.
    max_depth: int, optional (default=5)
        Maximum depth of the probabilistic circuits.
    custom_net: nn.Sequential, optional (default=None)
        A custom neural network to use in the model.
    hidden_layers: int, optional (default=4)
        The number of hidden layers in the neural network.
    neuron_per_layer: int or list of ints, optional (default=512)
        The number of neuron in each layer in the neural network.
    activation: str, optional (default="LeakyReLU"), allowed: "ReLU", "Tanh", "Sigmoid", "LeakyReLU", "Identity
        The activation function used in the neural network.
    batch_norm: bool, optional (default=True)
        Whether to use batch normalization in the neural network.
    dropout_rate: float, optional (default=0.1)
        The dropout rate used in the neural network.
    max_iter: int, optional (default=100)
        Maximum number of iterations to perform.
    batch_size_train: int or None, optional (default=1024)
        The batch size to use for training. If None, the entire dataset is used.
    batch_size_impute: int or None, optional (default=256)
        The batch size to use for imputing. If None, the entire dataset is used.
    tol: float, optional (default=1e-4)
        Tolerance for the convergence criterion.
    patience: int, optional (default=10)
        Number of iterations to wait if no improvement and then stop the training.
    lr:  float, optional (default=0.001)
        The learning rate for the optimizer.
    weight_decay: float, optional (default=0.01)
        The weight decay (L2 penalty) for the optimizer.
    use_gpu: bool, optional (default=True)
        Whether to use GPU for training and imputation if available. If False, CPU is used.
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
    training_likelihoods_: list of floats
        List of recorded log likelihoods during training.
    imputing_likelihoods_: list of floats
        List of recorded log likelihoods during imputation.
    mean_: float
        The mean value for each feature observed during training.
    std_: float
        The standard deviation for each feature observed during training.
    binary_info_: list
        The information about binary features observed during training.
    integer_info_: list
        The information about integer features observed during training.
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
            settings: str = "custom",
            missing_values: int | float | str | list | None = np.nan,
            n_components_train: int = 256,
            n_components_impute: int | None = 2048,
            latent_dim: int = 4,
            top_k: int | None = None,
            lo: bool = False,
            pc_type: str = "factorized",
            imputation_method: str = "EM",
            ordinal_features: dict | None = None,
            max_depth: int = 5,
            custom_net: nn.Sequential | None = None,
            hidden_layers: int = 4,
            neurons_per_layer: int | list = 512,
            activation: str = "LeakyReLU",
            batch_norm: bool = True,
            dropout_rate: float = 0.1,
            max_iter: int = 100,
            batch_size_train: int | None = 1024,
            batch_size_impute: int | None = 256,
            tol: float = 1e-4,
            patience: int = 10,
            lr: float = 0.001,
            weight_decay: float = 0.01, 
            use_gpu: bool = True,
            random_state: int | None = None,
            verbose: int = 0,
            copy: bool = True,
            keep_empty_features: bool = False,
        ):
        # Mixture model
        self.model = None

        # Parameters not related to settings
        self.missing_values = missing_values
        self.ordinal_features = ordinal_features
        self.batch_size_train = batch_size_train
        self.batch_size_impute = batch_size_impute
        self.use_gpu = use_gpu
        self.random_state = random_state
        self.verbose = verbose
        self.copy = copy
        self.keep_empty_features = keep_empty_features

        # Parameters related to settings
        if settings != "custom":    # Use preset parameters
            self._apply_preset_settings(settings)
        else:   # Choose user selected parameters
            self.n_components_train = n_components_train
            self.n_components_impute = n_components_impute
            self.latent_dim = latent_dim
            self.top_k = top_k
            self.lo = lo
            self.pc_type = pc_type
            self.imputation_method = imputation_method
            self.custom_net = custom_net
            self.hidden_layers = hidden_layers
            self.neurons_per_layer = neurons_per_layer
            self.activation = activation
            self.batch_norm = batch_norm
            self.dropout_rate = dropout_rate
            self.max_depth = max_depth
            self.max_iter = max_iter
            self.batch_size_train = batch_size_train
            self.batch_size_impute = batch_size_impute
            self.tol = tol
            self.patience = patience
            self.lr = lr
            self.weight_decay = weight_decay

        # Attributes
        self.is_fitted_ = False
        self.n_features_in_ = None
        self.feature_names_in_ = None
        self.components_ = None
        self.log_likelihood_ = None
        self.training_likelihoods_ = None
        self.imputing_likelihoods_ = None
        self.min_vals_ = 0.0
        self.max_vals_ = 1.0
        self.binary_info_ = None
        self.integer_info_ = None
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
        start_time = time.time() 
        # If the input data is a string (filepath), load the data from the file
        if isinstance(X, str):
            X = _load_file(X, sep=sep, decimal=decimal)
        # Transform the data to a NumPy array
        X, _, feature_names = _to_numpy(X)
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = feature_names

        if self.verbose > 1:
            print(f"Training data loading time: {time.time() - start_time:.2f}")

        # Preprocess the data
        start_time_preprocessing = time.time()
        X_preprocessed, self.binary_info_, self.integer_info_, self.encoding_info_ = self._preprocess_data(X, train=True)
        if self.verbose > 1: 
            print(f"Training data preprocessing time: {time.time() - start_time_preprocessing:.2f}")

        # Fit the model using X
        self.model, self.training_likelihoods_ = train_cm_tpm(
            X_preprocessed, 
            pc_type=self.pc_type,
            latent_dim=self.latent_dim, 
            num_components=self.n_components_train,
            num_components_impute=self.n_components_impute, 
            k=self.top_k,
            lo = self.lo,
            net=self.custom_net,
            hidden_layers=self.hidden_layers,
            neurons_per_layer=self.neurons_per_layer,
            activation=self.activation,
            batch_norm=self.batch_norm,
            dropout_rate=self.dropout_rate,
            epochs=self.max_iter,
            batch_size=self.batch_size_train,
            tol=self.tol, 
            patience=self.patience,
            lr=self.lr,
            weight_decay=self.weight_decay,
            use_gpu=self.use_gpu,
            random_state=self.random_state,
            verbose=self.verbose,
            )

        self.is_fitted_ = True
        return self
    
    def transform(self, X: str | np.ndarray | pd.DataFrame | list, save_path: str = None, sep=",", decimal=".", return_format: str = "auto"):
        """
        Impute missing values in the dataset.

        Parameters:
            X (array-like or str): Data with missing values.
                - Allowed: np.ndarray, pd.DataFrame, list of lists, or a file path (CSV, XLSX, Parquest, Feather)
            save_path (str, optional): If provided, saves output to a file. Otherwise, if X is a filepath, save output to 'X + _imputed'.
            sep (str, optional): Delimiter for CSV files.
            decimal (str, optional): Decimal separator for CSV files.
            return_format (str, optional): Format of returned data. One of:
                - "auto": returns same type as input
                - "ndarray": always returns a NumPy array
                - "dataframe": always returns a pandas DataFrame
            
        Returns:
            X_imputed (array-like, same type as X): Dataset with missing values replaced. If X is a filepath, X imputed will be a NumPy array.

        Raises:
            ValueError: If the model is not yet fitted.
            ValueError: If an unknown file format is provided as a save path.
        """
        start_time = time.time()
        if not self.is_fitted_:  # Check if the model is fitted
            raise ValueError("The model has not been fitted yet. Please call the fit method first.")
        
        file_in = None      # Variable to store the input file path
        
        if isinstance(X, str):      # If the input data is a string (filepath), load the data from the file
            file_in = X
            X = _load_file(X, sep=sep, decimal=decimal)

        # Transform the data to a NumPy array
        X_np, original_format, columns = _to_numpy(X)

        # Add a dimension to 1-dimensional data
        if X_np.ndim == 1:
            X_np = np.expand_dims(X_np, 0)

        if file_in:     # Force ndarray as file input default
            original_format = "ndarray"
        
        if self.verbose > 1:
            print(f"Missing data loading time: {time.time() - start_time:.2f}")

        X_imputed = self._impute(X_np)      # Perfom imputation
        
        # Respect return_format
        if return_format == "ndarray":
            result = np.asarray(X_imputed)
        elif return_format == "dataframe":
            result = pd.DataFrame(X_imputed, columns=columns)
        else:  # auto
            result = _restore_format(X_imputed, original_format, columns)
        
        # If save_path is set, save the imputed data to a file
        if save_path or file_in:
            df = pd.DataFrame(result, columns=columns)
            if not save_path:
                save_path = file_in[:file_in.rfind(".")] + "_imputed" + file_in[file_in.rfind("."):]
            if save_path.endswith(".csv"):
                df.to_csv(save_path, index=False)
            elif save_path.endswith(".xlsx"):
                df.to_excel(save_path, index=False, engine="openpyxl")
            elif save_path.endswith(".parquet"):
                df.to_parquet(save_path)
            elif save_path.endswith(".feather"):
                df.to_feather(save_path)
            else:
                raise ValueError("Unsupported file format for saving.")

        # Return the imputed data
        return result
    
    def fit_transform(self, X: str | np.ndarray | pd.DataFrame | list, save_path:str = None, sep=",", decimal=".", return_format: str = "auto"):
        """
        Fit the model and then transform the data.

        Parameters:
            X (array-like or str): Data with missing values.
                - Allowed: np.ndarray, pd.DataFrame, list of lists, or a file path (CSV, XLSX, Parquest, Feather)
            save_path (str, optional): If provided, saves output to a file. Otherwise, if X is a filepath, save output to 'X + _imputed'.
            sep (str, optional): Delimiter for CSV files.
            decimal (str, optional): Decimal separator for CSV files.
            return_format (str, optional): Output format: "auto", "ndarray", or "dataframe".
            
        Returns:
            X_imputed (array-like, same type as X): Dataset with missing values replaced. If X is a filepath, X imputed will be a NumPy array.
        """
        return self.fit(X, sep=sep, decimal=decimal).transform(X, save_path=save_path, sep=sep, decimal=decimal, return_format=return_format)
    
    def get_feature_names_out(self, input_features=None):
        """
        Get output feature names for transformation.

        Parameters:
            input_features (list of str or None): Optional input feature names. If None, uses feature names seen during fit.

        Returns:
            np.ndarray: Array of output feature names.
        """
        if not hasattr(self, "feature_names_in_"):
            raise AttributeError("The model has not been fitted yet. Call `fit` before getting feature names.")

        if input_features is None:
            return np.array(self.feature_names_in_, dtype=str)

        if len(input_features) != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} input features, got {len(input_features)}.")

        return np.array(input_features, dtype=str)
    
    def get_params(self):
        """
        Get parameters for this CMImputer.

        Returns:
            dict: Parameter names mapped to their values.
        """
        return {
            "missing_values": self.missing_values,
            "n_components_train": self.n_components_train,
            "n_components_impute": self.n_components_impute,
            "latent_dim": self.latent_dim,
            "top_k": self.top_k,
            "lo": self.lo,
            "pc_type": self.pc_type,
            "imputation_method": self.imputation_method,
            "ordinal_features": self.ordinal_features,
            "max_depth": self.max_depth,
            "custom_net": self.custom_net,
            "hidden_layers": self.hidden_layers,
            "neurons_per_layer": self.neurons_per_layer,
            "activation": self.activation,
            "batch_norm": self.batch_norm,
            "dropout_rate": self.dropout_rate,
            "max_iter": self.max_iter,
            "batch_size_train": self.batch_size_train,
            "batch_size_impute": self.batch_size_impute,
            "tol": self.tol,
            "patience": self.patience,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "use_gpu": self.use_gpu,
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

        Returns:
            self: Updated instance.

        Raises:
            ValueError: If an invalid parameter is passed.
        """
        valid_params = self.get_params()
        for key, value in params.items():
            if key not in valid_params:
                raise ValueError(f"Invalid parameter '{key}' for CMImputer.")
            setattr(self, key, value)
        return self
    
    def evaluate(self, X: str | np.ndarray | pd.DataFrame | list):
        """
        Evaluate how well the model explains the data.

        Parameters:
            X (array-like or str): Data with missing values.
                - Allowed: np.ndarray, pd.DataFrame, list of lists, or a file path (CSV, XLSX, Parquest, Feather)

        Returns:
            log_likelihood_ (float): Log likelihood of the data under the trained model.
        """
        # If the input data is a string (filepath), load the data from the file
        if isinstance(X, str):
            X = _load_file(X)
        # Transform the data to a NumPy array
        X, _, _ = _to_numpy(X)

        if not self.is_fitted_:
            raise ValueError("The model has not been fitted yet. Please call the fit method first.")
        
        # Evaluate the model using X (One pass through the model?)
        # TODO
        return 0.0
    
    def _apply_preset_settings(self, settings: str):
        """Applies the chosen settings on the class parameters."""
        # The presets for each setting option
        settings_lower = settings.lower()
        presets = {
            "fast": {
                "n_components_train": 128,
                "n_components_impute": 2048,
                "latent_dim": 4,
                "top_k": None,
                "lo": False,
                "pc_type": "factorized",
                "imputation_method": "EM",
                "max_depth": 5,
                "custom_net": None,
                "hidden_layers": 2,
                "neurons_per_layer": 128,
                "activation": "LeakyReLU",
                "batch_norm": False,
                "dropout_rate": 0.0,
                "max_iter": 100,
                "tol": 1e-4,
                "patience": 10,
                "lr": 0.001,
                "weight_decay": 0.01,
            },
            "balanced": {
                "n_components_train": 256,
                "n_components_impute": 2048,
                "latent_dim": 4,
                "top_k": None,
                "lo": False,
                "pc_type": "factorized",
                "imputation_method": "EM",
                "max_depth": 5,
                "custom_net": None,
                "hidden_layers": 4,
                "neurons_per_layer": 512,
                "activation": "LeakyReLU",
                "batch_norm": True,
                "dropout_rate": 0.1,
                "max_iter": 100,
                "tol": 1e-4,
                "patience": 10,
                "lr": 0.001,
                "weight_decay": 0.01,
            },
            "precise": {
                "n_components_train": 256,
                "n_components_impute": 1024,
                "latent_dim": 8,
                "top_k": None,
                "lo": False,
                "pc_type": "factorized",
                "imputation_method": "exact",
                "max_depth": 5,
                "custom_net": None,
                "hidden_layers": 5,
                "neurons_per_layer": 1024,
                "activation": "LeakyReLU",
                "batch_norm": True,
                "dropout_rate": 0.1,
                "max_iter": 200,
                "tol": 1e-4,
                "patience": 10,
                "lr": 0.001,
                "weight_decay": 0.01,
            }
        }

        # Check if valid setting is chosen
        if settings_lower not in presets:
            raise ValueError(f"Unknown settings: '{settings}'")
        
        # Apply the settings
        for param, value in presets[settings_lower].items():
            setattr(self, param, value)
        return
    
    def _check_consistency(self, X: np.ndarray):
        """Ensures that the input data is consistent with the training data."""
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Mismatch in number of features. Expected {self.n_features_in_}, got {X.shape[1]}.")
        
        mask, info = self.encoding_info_
        X_checked = X.copy()

        for i in range(X.shape[1]):
            if mask[i]:     # Feature is categorical
                if _all_numeric(X_checked[:, i]):
                    raise ValueError(f"Feature {i} was categorical during training but numeric in new data.")
                
                # Get the encoding info of this feature
                enc_info = info[i]

                # If there are new categorical feature values in the input data, warn the user
                for val in X_checked[:, i]:
                    if val not in enc_info and val != "nan":
                        warnings.warn(f"New categorical value detected in column {i}: '{val}'. Treating this value as missing.")

                # Apply the same encoding
                X_checked[:, i] = [enc_info[val] if val in enc_info else np.nan for val in X_checked[:, i]]

            else:       # Feature is numeric
                if not _all_numeric(X_checked[:, i]):
                    raise ValueError(f"Feature {i} was numeric during training but categorical in new data.")

        return X_checked.astype(float), mask, info

    
    def _preprocess_data(self, X: np.ndarray, train: bool = False):
        """Preprocess the input data before imputation."""
        if self.copy:
            X = X.copy()

        # Change all instances of 'missing_value' to NaN
        X_transformed = _missing_to_nan(X, self.missing_values)

        # Check which features are integer features (only consisting of whole numbers)
        integer_mask = np.all(np.vectorize(is_valid_integer)(X_transformed), axis=0)

        # Convert non-floats (e.g. strings) to floats using encoding
        if train:
            X_transformed, encoding_mask, encoding_info = _integer_encoding(X_transformed, ordinal_features=self.ordinal_features)
        else:
            # Use encoding info from training to ensure consistent encoding at transform time
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

        # Compute min and max for scaling (during training only)
        if train:       
            min_vals = np.nanmin(X_transformed, axis=0)
            self.min_vals_ = np.where(np.isnan(min_vals), 0.0, min_vals)
            max_vals = np.nanmax(X_transformed, axis=0)
            self.max_vals_ = np.where(np.isnan(max_vals), 1.0, max_vals)
        
        # Apply min-max scaling
        scale = self.max_vals_ - self.min_vals_
        scale[scale == 0] = 1e-9
        X_scaled = (X_transformed - self.min_vals_) / scale

        # Check which features are binary features (only consisting of 0s and 1s)
        binary_mask = np.array([
            np.isin(np.unique(X_scaled[:, i][~np.isnan(X_scaled[:, i])]), [0, 1]).all()
            for i in range(X_scaled.shape[1])
        ])

        return X_scaled.astype(float), binary_mask, integer_mask, (encoding_mask, encoding_info)
    
    def _impute(self, X: np.ndarray) -> np.ndarray:
        """Impute missing values in input X"""
        start_time = time.time()
        X_in = X.copy()
        X_nan = _missing_to_nan(X, self.missing_values)
        X_preprocessed, _, _, _ = self._preprocess_data(X, train=False)

        if self.verbose > 1:
            print(f"Missing data preprocessing time: {time.time() - start_time:.2f}")

        if not np.any(np.isnan(X_preprocessed)):
            warnings.warn(f"No missing values detected in input data, transformation has no effect. Did you set the correct missing value: '{self.missing_values}'?")

        if not self.imputation_method in ["EM", "exact"]:
            warnings.warn(f"Invalid imputation method selected: {self.imputation_method}, defaulting to EM")
            self.imputation_method = "EM"

        if self.imputation_method == "exact":
            X_imputed, self.log_likelihood_, self.imputing_likelihoods_ = impute_missing_values_exact(
                X_preprocessed, 
                self.model,
                num_components=self.n_components_impute,
                # epochs=self.max_iter,
                # lr=self.lr,
                max_batch_size=self.batch_size_impute,
                use_gpu=self.use_gpu,
                random_state=self.random_state,
                verbose = self.verbose,
            )
        else:    # Use EM imputation by default
            X_imputed, self.log_likelihood_ = impute_missing_values_component(
                X_preprocessed, 
                self.model,
                num_components=self.n_components_impute,
                k=None,
                max_batch_size=self.batch_size_impute,
                use_gpu=self.use_gpu,
                random_state=self.random_state,
                verbose = self.verbose,
            )
            self.imputing_likelihoods_ = self.log_likelihood_

        start_time_post = time.time()
        # Round the binary features to the nearest option
        X_imputed[:, self.binary_info_] = np.round(X_imputed[:, self.binary_info_])
        
        # Scale the data back to the original
        scale = self.max_vals_ - self.min_vals_
        scale[scale == 0] = 1e-9
        X_scaled = X_imputed * scale + self.min_vals_

        # Decode the binary features
        encoding_mask, encoding_info = self.encoding_info_
        X_decoded = _restore_binary_encoding(X_scaled, self.bin_encoding_info_, X_imputed)

        # Round the integer features to the nearest integer
        X_decoded[:, self.integer_info_] = np.round(X_decoded[:, self.integer_info_])

        # Decode the non-numerical features
        X_decoded = _restore_encoding(X_decoded, encoding_mask, encoding_info)        

        # Make sure the original values remain the same
        try:
            mask = ~np.isnan(X_nan)
        except TypeError:
            mask = X_nan != "nan"
        X_filled = np.where(mask, X_in, X_decoded)

        if self.verbose > 1:
            print(f"Data post-processing time: {time.time() - start_time_post:.2f}")
        return X_filled
