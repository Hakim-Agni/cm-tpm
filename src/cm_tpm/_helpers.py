import math
import numpy as np
import pandas as pd

def _load_file(filepath: str, sep: str = ",", decimal: str = ".") -> pd.DataFrame:
    """
    Loads a dataset from a file into a pandas DataFrame.
    
    Parameters:
        filepath (str): Path to the file.
        sep (str, optional): Separator for CSV files. Default is ",".
        decimal (str, optional): Decimal point character for CSV files. Default is ".".

    Returns:
        Pandas DataFrame: Loaded dataset.

    Raises:
        ValueError: If the file format is not supported.
    """
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
        
def _to_numpy(X: pd.DataFrame | list | np.ndarray) -> np.ndarray:
    """
    Converts input data to NumPy array for internal processing.
    
    Parameters:
        X (DataFrame or list or ndarray): Input data.

    Returns:
        tuple: Tuple containing the NumPy array, its original format, and column names (if applicable).

    Raises:
        TypeError: If the input data type is not supported.
    """
    if isinstance(X, pd.DataFrame):
        return X.to_numpy(), "DataFrame", X.columns
    elif isinstance(X, list):
        return np.asarray(X), "list", None
    elif isinstance(X, np.ndarray):
        return X, "ndarray", None
    else:
        raise TypeError(f"Unsupported data type: {type(X).__name__}. Expected NumPy ndarray, pandas DataFrame or list.")
        
def _restore_format(X_imputed: np.ndarray, original_format: str = "ndarray", columns=None) -> pd.DataFrame | list | np.ndarray:
    """
    Restore the format of the imputed data based on the original input format.
    
    Parameters:
        X_imputed (ndarray): Imputed data.
        original_format (str): Original format of the data ('DataFrame', 'list', or 'ndarray').
        columns (list, optional): Column names for DataFrame restoration.

    Returns:
        DataFrame or list or ndarray: Restored data in the original format.

    Raises:
        ValueError: If the original format is not supported.
    """
    if original_format == "DataFrame":
        return pd.DataFrame(X_imputed, columns=columns)
    elif original_format == "list":
        return X_imputed.tolist()
    elif original_format == "ndarray":
        return X_imputed
    else:
        raise ValueError(f"Unsupported original format: {original_format}. Expected 'DataFrame', 'list', or 'ndarray'.")

def _missing_to_nan(X: np.ndarray, missing_values: float | str | int | list) -> np.ndarray:
    """
    Set all instances of 'missing_values' to NaN.

    Parameters:
        X (ndarray): Input data.
        missing_values (float, str, int, list): Values to be replaced with NaN.

    Returns:
        ndarray: Data with specified missing values replaced by NaN.
    """
    X_nan = X.copy()

    # Assure missing_values is a list
    is_list = isinstance(missing_values, list)
    missing_values = missing_values if is_list else [missing_values]

    try:
        # If the data is numerical, set np.nan
        X_nan = X_nan.astype(float)
        for missing_value in missing_values:
            X_nan[X == missing_value] = np.nan 
    except (ValueError, TypeError):
        # If the data is not numerical, set string 'nan'
        X_nan = X_nan.astype(str)  # Convert to string for processing
        for missing_value in missing_values:
            X_nan[X == str(missing_value)] = "nan"

    return X_nan

def _all_numeric(X: np.ndarray) -> bool:
    """Checks if all values in a 1D array are numerical."""
    try:
        np.asarray(X, dtype=float)
        return True
    except (ValueError, TypeError):
        return False

def is_valid_integer(val) -> bool:
    """Checks if a value is an integer or NaN"""
    try:
        x = float(val)
        return np.isnan(x) or (x.is_integer() and np.isfinite(x))
    except (ValueError, TypeError):
        return False

def _integer_encoding(X: np.ndarray, ordinal_features: dict = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Encodes non-numeric features in each column into integers, optionally using provided ordinal mappings.

    Parameters:
        X (ndarray): Input 2D array.
        ordinal_features (dict or None, optional): Dictionary mapping column indices to lists of ordered values.

    Returns:
        Tuple[ndarray, ndarray, dict]: 
            The encoded array.
            Boolean mask indicating which columns were encoded.
            Dictionary of encoding maps per encoded colmn.
    """
    X_encoded = X.copy()
    n_cols = X.shape[1]
    encoding_mask = np.full(n_cols, False)
    encoding_info = {}
        
    try:
        # If all values are numerical, return the input array.
        X_encoded = X_encoded.astype(float)
        return X_encoded, encoding_mask, encoding_info
    except ValueError:
        X_encoded = X_encoded.astype(str)  # Convert to string for processing
        # Look at each column seperately
        for i in range(n_cols):
            col = X_encoded[:, i]
            if _all_numeric(col):  # If the column is already numeric, continue
                continue

            # Get unique values in column
            unique_values = np.unique(col)
            unique_values = unique_values[unique_values != "nan"]      # Remove NaN from unique values
                        
            if ordinal_features and i in ordinal_features:
                order = ordinal_features[i]
                value_map = {val: j for j, val in enumerate(order)}
            else:
                value_map = {val: idx for idx, val in enumerate(unique_values)}   # Create value map for unique values

            # Apply endoding  
            encoding_mask[i] = True
            encoding_info[i] = value_map
            X_encoded[:, i] = [value_map.get(val, np.nan) for val in col]   # Apply value map to array

    return X_encoded.astype(float), encoding_mask, encoding_info

def _restore_encoding(X: np.ndarray, mask: np.ndarray, info: dict) -> np.ndarray:
    """
    Restores encoded features to the original type.
    
    Parameters:
        X (ndarray): Encoded data.
        mask (ndarray): Boolean mask indicating which columns were encoded.
        info (dict): Dictionary of encoding maps per encoded column.

    Returns:
        ndarray: Restored data.
    """
    X_restored = X.astype(str)
        
    for i in range(X.shape[1]):
        if not mask[i]:
            continue  # Skip non-encoded columns

        reverse_map = {v: k for k, v in info[i].items()}  # Reverse integer mapping
        X_restored[:, i] = [
            reverse_map.get(int(round(val)), np.nan) if not np.isnan(val) else np.nan
            for val in X[:, i]
        ]

    # Try converting back to float, else keep as strings
    try:
        X_restored = X_restored.astype(float)
        return X_restored
    except (ValueError, TypeError):
        return X_restored
        
def _binary_encoding(X: np.ndarray, mask: np.ndarray, info: dict, ordinal_features: dict = None) -> tuple[np.ndarray, list]:
    """
    Converts integer encoded features into multiple binary features.
    
    Parameters:
        X (ndarray): Input data.
        mask (ndarray): Boolean mask indicating which columns were encoded.
        info (dict): Dictionary of encoding maps per encoded column.
        ordinal_features (dict or None, optional): Dictionary mapping column indices to lists of ordered values.

    Returns:
        Tuple[ndarray, list]: 
            The encoded array.
            List of binary encoding info for each column.
    """
    n_rows, n_cols = X.shape
    replacing = {}
    bin_info = []
    # Look at each column seperately
    for i in range(n_cols):
        # If the column is integer encoded and not an ordinal feature, continue
        if mask[i] and (not ordinal_features or not i in ordinal_features):     
            n_unique = max(info[i].values()) + 1  # Get number of unique values
            num_bits = max(1, math.ceil(math.log2(n_unique)))  # Compute bit length
                
            # Create binary mappings
            bin_vals = {val: list(map(int, f"{val:0{num_bits}b}")) for val in range(n_unique)}

            # Convert integer feature into binary representation
            col = X[:, i]
            X_bin = np.empty((n_rows, num_bits))
            for j in range(n_rows):
                val = col[j]
                X_bin[j] = np.nan if np.isnan(val) else bin_vals[int(val)]

            replacing[i] = X_bin  # Store transformed binary columns
            bin_info.append([num_bits, n_unique-1])   # Store binary encoding info
        else:
            bin_info.append(-1)  # Store -1 for non-encoded columns

    # Construct final transformed dataset
    X_encoded = np.hstack([
        replacing[i] if i in replacing else X[:, i].reshape(-1, 1)
        for i in range(n_cols)
    ])

    return X_encoded, bin_info
    
def _restore_binary_encoding(X: np.ndarray, info: dict, X_prob: np.ndarray) -> np.ndarray:
    """
    Restores the binary encoding for encoded features.
    
    Parameters:
        X (ndarray): Encoded data.
        info (dict): Dictionary of encoding maps per encoded column.
        X_prob (ndarray): Probabilities for binary encoding.

    Returns:
        ndarray: Restored data.
    """
    X_str = X.astype(str)
    n_samples = X.shape[0]
    n_features = len(info)
    restored = np.empty((n_samples, n_features))  # Initialize empty array for restored data

    column_index = 0       # Encoding index
    for i in range(n_features):
        if info[i] != -1:       # If the column is binary encoded, continue
            bit_length, max_val = info[i]   # Get bit length and max value
            
            # Create reverse binary mappings
            bin_map = {format(val, f"0{bit_length}b"): val for val in range(2 ** bit_length)}

            for j in range(n_samples):     # Look at each row seperately
                # Extract bitstring from encoded columns
                bitstring = "".join(X_str[j, column_index + k][0] for k in range(bit_length))
                int_value = bin_map.get(bitstring, None)

                # Fix values above max_val using probabilities
                while int_value is not None and int_value > max_val:   
                    probs = X_prob[j, column_index:column_index + bit_length]
                    one_bits = np.where(probs >= 0.5)[0]
                    if len(one_bits) == 0:
                        break  # No bits to flip

                    # Flip the most uncertain bit (closest to 0.5)
                    least_certain = one_bits[np.argmin(probs[one_bits])]
                    X_str[j, column_index + least_certain] = "0.0"
                    X_prob[j, column_index + least_certain] = 0.0

                    # Rebuild bitstring and decode again
                    bitstring = "".join(X_str[j, column_index + k][0] for k in range(bit_length))
                    int_value = bin_map.get(bitstring, None)

                restored[j, i] = int_value if int_value is not None else np.nan  # Insert the decoded value            
            column_index += bit_length  # Update index for next binary column
        else:
            restored[:, i] = X_str[:, column_index] # Keep original column
            column_index += 1
        
    # Try converting back to float, else keep as strings
    try:
        restored = restored.astype(float)
        return restored
    except (ValueError, TypeError):
        return restored
