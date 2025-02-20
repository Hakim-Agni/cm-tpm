import numpy as np
import pandas as pd
import warnings
from ._model import train_cm_tpm, impute_missing_values

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
    missing_strategy: str, optional (default="integration"), allowed: "integration", "ignore"
        The strategy to use for missing data in the training data. 
    net: nn.Sequential, optional (default=None)
        A custom neural network to use in the model.
    max_depth: int, optional (default=5)
        Maximum depth of the probabilistic circuit.
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
            missing_strategy: str = "integration",
            net = None,
            max_depth: int = 5,
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
        self.net = net
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
        self.mean_ = 0.0
        self.std_ = 1.0
        self.binary_info_ = None
        self.encoding_info_ = None
        self.categorical_info_ = None
        self.random_state_ = np.random.RandomState(self.random_state) if self.random_state is not None else np.random

    def _load_file(self, filepath: str, sep=",", decimal=".") -> pd.DataFrame:
        """Loads a dataset from a file into a pandas DataFrame."""
        if filepath.endswith('.csv'):
            return pd.read_csv(filepath, sep=sep, decimal=decimal)
        elif filepath.endswith('.xlsx'):
            return pd.read_excel(filepath, engine="openpyxl")
        elif filepath.endswith('.parquet'):
            return pd.read_parquet(filepath)
        elif filepath.endswith('.feather'):
            return pd.read_feather(filepath)
        else:
            raise ValueError("Unsupported file format. Please provide a CSV, Excel, Parquet, or Feather file.")
        
    def _to_numpy(self, X):
        """Converts input data to NumPy array for internal processing."""
        if isinstance(X, pd.DataFrame):
            return X.to_numpy(), "DataFrame", X.columns
        elif isinstance(X, list):
            return np.array(X), "list", None
        elif isinstance(X, np.ndarray):
            return X, "ndarray", None
        else:
            raise ValueError("Unsupported data type. Please provide a NumPy array, pandas DataFrame or list.")
        
    def _restore_format(self, X_imputed, original_format="ndarray", columns=None):
        """Restore the format of the imputed data based on the original input format."""
        if original_format == "DataFrame":
            return pd.DataFrame(X_imputed, columns=columns)
        elif original_format == "list":
            return X_imputed.tolist()
        return X_imputed
    
    def _all_numeric(self, X: np.ndarray):
        """Checks if all values in a 1D array are numerical."""
        for x in X:
            try:
                float(x)
            except ValueError:
                return False
        return True

    def _integer_encoding(self, X: np.ndarray):
        """Converts Non-numeric features into integers."""
        encoding_mask = np.full(X.shape[1], False)
        encoding_info = {}
        
        try:
            # If all values are numerical, return the input array.
            X = X.astype(float)
            return X, encoding_mask, encoding_info
        except ValueError:
            # Look at each column seperately
            for i in range(X.shape[1]):
                if not self._all_numeric(X[:, i]):
                    # Get unique values in column
                    unique_values = np.unique(X[:, i])
                    if "nan" in unique_values:      # Remove NaN from unique values
                        unique_values = np.delete(unique_values, np.argwhere(unique_values=="nan"))
                        
                    encoding_mask[i] = True
                    value_map = {unique_values[i]: i for i in range(len(unique_values))}    # Create value map for unique values
                    encoding_info[i] = value_map
                    X[:, i] = [value_map[val] if val in value_map else np.nan for val in X[:, i]]   # Apply value map to array

        return X.astype(float), encoding_mask, encoding_info
    
    def _restore_encoding(self, X: np.ndarray, mask, info):
        """Restores encoded features to the original type."""
        restored = X.copy()
        restored = restored.astype(str)
        
        for i in range(X.shape[1]):
            if mask[i]:  # Restore only encoded columns
                reverse_map = {v: k for k, v in info[i].items()}  # Reverse integer mapping
                restored[:, i] = [reverse_map.get(round(val), np.nan) if not np.isnan(val) else np.nan 
                                    for val in X[:, i]]

        try:
            restored = restored.astype(float)
            return restored
        except ValueError:
            return restored
    
    def _preprocess_data(self, X: np.ndarray, train: bool = False):
        """Preprocess the input data before imputation."""
        categorical_info = {}  

        if self.copy:
            X_transformed = X.copy()

        # TODO: Convert non-floats (e.g. strings) to floats using encoding
        X_transformed, encoding_mask, encoding_info = self._integer_encoding(X_transformed)
        #X_transformed = X_transformed.astype(float)     # Make sure the values are floats
        
        # Set all instances of 'missing_values' to NaN
        if self.missing_values is not np.nan:
            X_transformed[X_transformed == self.missing_values] = np.nan

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

        # Check which features are binary features (only consisting of 0s and 1s)
        binary_mask = np.array([
            np.isin(np.unique(X_transformed[:, i][~np.isnan(X_transformed[:, i])]), [0, 1]).all()
            for i in range(X_transformed.shape[1])
        ])

        if train:       # Update the means and stds only during training
            self.mean_ = np.nanmean(X_transformed, axis=0)
            self.mean_ = np.where(np.isnan(self.mean_), 0, self.mean_)      # Replace mean NaNs with 0
            self.std_ = np.nanstd(X_transformed, axis=0)
            self.std_ = np.where(np.isnan(self.std_), 1, self.std_)         # Replace std NaNs with 1
            self.std_[self.std_ == 0] = 1e-8        # Replace 0 with a small value to avoid zero division
        
        # Scale the data
        X_scaled = (X_transformed - self.mean_) / self.std_
        X_scaled[:, binary_mask] = X_transformed[:, binary_mask]    # Keep binary columns unscaled
        
        # for i in range(X.shape[1]):  
        #     col_data = X[:, i]

        #     # Detect categorical columns (string or object)
        #     if np.issubdtype(col_data.dtype, np.object_) or np.issubdtype(col_data.dtype, np.str_):
        #         raise ValueError("Categorical columns are not supported yet.")
        #         # unique_values, encoded_values = np.unique(col_data, return_inverse=True)
        #         # X_transformed[:, i] = encoded_values
        #         # categorical_info[i] = unique_values  # Store mapping of index → categories

        return X_scaled.astype(float), binary_mask, (encoding_mask, encoding_info), categorical_info

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
            X = self._load_file(X, sep=sep, decimal=decimal)
        # Transform the data to a NumPy array
        X, _, feature_names = self._to_numpy(X)
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = feature_names

        # Fit the model using X
        X_preprocessed, self.binary_info_, self.encoding_info_, self.categorical_info_ = self._preprocess_data(X, train=True)
        self.model = train_cm_tpm(
            X_preprocessed, 
            pc_type=self.pc_type,
            latent_dim=self.latent_dim, 
            num_components=self.n_components, 
            missing_strategy=self.missing_strategy,
            net=self.net,
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
            X = self._load_file(X, sep=sep, decimal=decimal)
        # Transform the data to a NumPy array
        X_np, original_format, columns = self._to_numpy(X)
        if file_in:
            original_format = "ndarray"
        # Perfom imputation
        X_imputed = self._impute(X_np)
        # Transform the imputed data to the original format
        result = self._restore_format(X_imputed, original_format, columns)
        
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
    
    def _impute(self, X: np.ndarray) -> np.ndarray:
        """Impute missing values in input X"""
        X_preprocessed, _, _, _ = self._preprocess_data(X, train=False)
        print(X_preprocessed)

        if not np.any(np.isnan(X_preprocessed)):
            warnings.warn(f"No missing values detected in input data, transformation has no effect. Did you set the correct missing value: '{self.missing_values}'?")

        # Add checks that train and missing data are of similar types (same binary features etc.)

        X_imputed = impute_missing_values(
            X_preprocessed, 
            self.model,
            random_state=self.random_state,
            verbose = self.verbose,
        )
        print(X_imputed)
        
        # Scale the data back to the original
        X_scaled = (X_imputed * self.std_) + self.mean_

        # Round the binary features to the nearest option
        X_scaled[:, self.binary_info_] = np.round(X_imputed[:, self.binary_info_])
        print(X_scaled)

        encoding_mask, encoding_info = self.encoding_info_
        X_decoded = self._restore_encoding(X_scaled, encoding_mask, encoding_info)

        # Make sure the original values remain the same
        mask = ~np.isnan(X_preprocessed)
        X_filled = np.where(mask, X, X_decoded)
        print(X_filled)
        return X_filled
    
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
            "net": self.net,
            "max_depth": self.max_depth,
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
            X = self._load_file(X)
        # Transform the data to a NumPy array
        X, _, _ = self._to_numpy(X)

        if not self.is_fitted_:
            raise ValueError("The model has not been fitted yet. Please call the fit method first.")
        
        # Evaluate the model using X
        # TODO
        return 0.0
