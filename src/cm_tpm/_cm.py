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
    random_state: int, RandomState instance or None, optional (default=None)
        Random seed for reproducibility

    Attributes
    ----------
    ...

    References
    ----------
    ...

    Examples
    --------
    ...
    """
    def __init__(self, random_state: int = None):
        super().__init__()
        self.random_state = random_state

    def _load_file(self, filepath: str) -> pd.DataFrame:
        """Loads a dataset from a file into a pandas DataFrame."""
        if filepath.endswith('.csv'):
            # TODO: Add support for other separators and decimal markers
            return pd.read_csv(filepath, sep=';', decimal=',')
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

    def fit(self, X: str | np.ndarray | pd.DataFrame | list) -> "CMImputer":
        """
        Fit the imputation model to the input dataset

        Parameters:
            X (array-like or str): Input data with missing values.
                - Allowed: np.ndarray, pd.DataFrame, list of lists, or a file path (CSV, XLSX, Parquest, Feather)

        Returns:
            self (CMImputer): Fitted instance.        
        """
        # If the input data is a string (filepath), load the data from the file
        if isinstance(X, str):
            X = self._load_file(X)
        # Transform the data to a NumPy array
        X, _, _ = self._to_numpy(X)
        # Fit the model using X
        # TODO

        return self
    
    def transform(self, X: str | np.ndarray | pd.DataFrame | list, save_path: str = None):
        """
        Impute missing values in the dataset.

        Parameters:
            X (array-like or str): Data with missing values.
                - Allowed: np.ndarray, pd.DataFrame, list of lists, or a file path (CSV, XLSX, Parquest, Feather)
            save_path (str, optional): If provided, saves output to a file. Otherwise, if X is a filepath, save output to 'X + _imputed'.

        Returns:
            X_imputed (array-like, same type as X): Dataset with missing values replaced. If X is a filepath, X imputed will be a NumPy array.
        """
        # If the input data is a string (filepath), load the data from the file
        if isinstance(X, str):
            file_in = X
            X, _, _ = self._to_numpy(self._load_file(X))
        # Transform the data to a NumPy array
        X_np, original_format, columns = self._to_numpy(X)
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
    
    def fit_transform(self, X: str | np.ndarray | pd.DataFrame | list, save_path:str = None):
        """
        Fit the model and then transform the data.

        Parameters:
            X (array-like or str): Data with missing values.
                - Allowed: np.ndarray, pd.DataFrame, list of lists, or a file path (CSV, XLSX, Parquest, Feather)
            save_path (str, optional): If provided, saves output to a file. Otherwise, if X is a filepath, save output to 'X + _imputed'.

        Returns:
            X_imputed (array-like, same type as X): Dataset with missing values replaced. If X is a filepath, X imputed will be a NumPy array.
        """
        return self.fit(X).transform(X, save_path)
    
    def _impute(self, X: np.ndarray) -> np.ndarray:
        """Placeholder for the actual imputation logic"""
        # TODO Add imputation
        return X
    