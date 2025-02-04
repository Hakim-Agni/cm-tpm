import numpy as np
import pandas as pd
from cm_tpm.cpp._add import add
from cm_tpm.cpp._multiply import multiply
from cm_tpm.cpp._subtract import subtract
from cm_tpm.cpp._divide import divide

class CMImputer:
    """
    Imputation for completing missing values using Continuous Mixtures of 
    Tractable Probabilistic Models.

    Parameters
    ----------
    missing_values: float, optional (default=np.nan)
        The placeholder for missing values in the input data, all instances of missing_values will be imputed.
    n_components: int, optional (default=10)
        Number of components to use in the mixture model.
    max_depth: int, optional (default=5)
        Maximum depth of the probabilistic circuit.
    max_iter: int, optional (default=100)
        Maximum number of iterations to perform.
    tol: float, optional (default=1e-4)
        Tolerance for the convergence criterion.
    weight_sharing: bool, optional (default=True)
        Whether to share parameters across mixture components.
    smooth: float, optional (default=1e-6)
        Smoothing parameter to avoid division by zero.
    random_state: int, RandomState instance or None, optional (default=None)
        Random seed for reproducibility.
    verbose: int, optional (default=0)
        Verbosity level, controls the debug messages.
    copy: bool, optional (default=True)
        Whether to copy the input data or modify it in place.
    keep_empty_features: bool, optional (default=False)
        Whether to keep features that have no missing values in the imputed dataset.

    Attributes
    ----------
    is_fitted_: bool
        Whether the model is fitted.
    n_features_: int
        Number of features in the input data.
    feature_names_in_: list of str
        Names of the input features.
    components_: list
        List of trained components in the mixture model.
    log_likelihood_: float
        Log likelihood of the data under the model.
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
            missing_values: float = np.nan,
            n_components: int = 10,
            max_depth: int = 5,
            max_iter: int = 100,
            tol: float = 1e-4,
            weight_sharing: bool = True,
            smooth: float = 1e-6, 
            random_state: int = None,
            verbose: int = 0,
            copy: bool = True,
            keep_empty_features: bool = False,
        ):
        """Initialize the CMImputer instance."""
        # Parameters
        self.missing_values = missing_values
        self.n_components = n_components
        self.max_depth = max_depth
        self.max_iter = max_iter
        self.tol = tol
        self.weight_sharing = weight_sharing
        self.smooth = smooth
        self.random_state = random_state
        self.verbose = verbose
        self.copy = copy
        self.keep_empty_features = keep_empty_features

        # Attributes
        self.is_fitted_ = False
        self.n_features_ = None
        self.feature_names_in_ = None
        self.components_ = None
        self.log_likelihood_ = None
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
        X, _, _ = self._to_numpy(X)
        # Fit the model using X
        # TODO

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
        """Placeholder for the actual imputation logic"""
        # TODO Add imputation
        return X
    
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
        # TODO: Add all parameters
        return {"random_state": self.random_state}
    
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
    