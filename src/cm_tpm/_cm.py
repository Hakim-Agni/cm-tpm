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
        "Converts input data to NumPy array for internal processing."
        if isinstance(X, pd.DataFrame):
            return X.to_numpy(), "DataFrame", X.columns
        elif isinstance(X, list):
            return np.array(X), "list", None
        elif isinstance(X, np.ndarray):
            return X, "ndarray", None
        else:
            raise ValueError("Unsupported data type. Please provide a NumPy array, pandas DataFrame or list.")