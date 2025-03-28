import math
import numpy as np
import pandas as pd

def _load_file(filepath: str, sep=",", decimal=".") -> pd.DataFrame:
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
        
def _to_numpy(X):
    """Converts input data to NumPy array for internal processing."""
    if isinstance(X, pd.DataFrame):
        return X.to_numpy(), "DataFrame", X.columns
    elif isinstance(X, list):
        return np.array(X), "list", None
    elif isinstance(X, np.ndarray):
        return X, "ndarray", None
    else:
        raise ValueError("Unsupported data type. Please provide a NumPy array, pandas DataFrame or list.")
        
def _restore_format(X_imputed, original_format="ndarray", columns=None):
    """Restore the format of the imputed data based on the original input format."""
    if original_format == "DataFrame":
        return pd.DataFrame(X_imputed, columns=columns)
    elif original_format == "list":
        return X_imputed.tolist()
    return X_imputed

def _missing_to_nan(X: np.ndarray, missing_values):
    """ Set all instances of 'missing_values' to NaN."""
    if missing_values is not np.nan:
        try:
            # If the data is numerical, set np.nan
            X = X.astype(float)
            X[X == missing_values] = np.nan
        except ValueError:
            # If the data is not numerical, set string 'nan'
            X[X == missing_values] = "nan"
    return X.copy()

def _all_numeric(X: np.ndarray):
    """Checks if all values in a 1D array are numerical."""
    for x in X:
        try:
            float(x)
        except ValueError:
            return False
    return True

def is_valid_integer(val):
    """Checks if a value is an integer (ignores NaN)"""
    try:
        x = float(val)
    except ValueError:
        return False
    return np.isnan(x) or (np.isfinite(x) and x == np.floor(x))

def _integer_encoding(X: np.ndarray, ordinal_features=None):
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
            if not _all_numeric(X[:, i]):
                # Get unique values in column
                unique_values = np.unique(X[:, i])
                if "nan" in unique_values:      # Remove NaN from unique values
                    unique_values = np.delete(unique_values, np.argwhere(unique_values=="nan"))
                        
                if ordinal_features and i in ordinal_features:
                    order = ordinal_features[i]
                    value_map = {val: j for j, val in enumerate(order)}
                else:
                    value_map = {unique_values[i]: i for i in range(len(unique_values))}    # Create value map for unique values
                    
                encoding_mask[i] = True
                encoding_info[i] = value_map
                X[:, i] = [value_map[val] if val in value_map else np.nan for val in X[:, i]]   # Apply value map to array

    return X.astype(float).copy(), encoding_mask, encoding_info

def _restore_encoding(X: np.ndarray, mask, info):
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
        
def _binary_encoding(X: np.ndarray, mask, info, ordinal_features=None):
    """Converts integer encoded features into multiple binary features."""
    replacing = {}
    bin_info = []
    for i in range(X.shape[1]):
        # If the column is integer encoded and not an ordinal feature, continue
        if mask[i] and (not ordinal_features or not i in ordinal_features):     
            n_unique = max(info[i].values()) + 1  # Get number of unique values
                
            num_cols = math.ceil(math.log2(n_unique))  # Compute bit length
            X_new = np.zeros((X.shape[0], num_cols))  # Initialize binary column array
                
            # Create binary mappings
            bin_vals = {val: list(map(int, format(val, f'0{num_cols}b'))) for val in range(n_unique)}

            # Convert integer feature into binary representation
            for j in range(X.shape[0]):
                X_new[j] = np.nan if np.isnan(X[j, i]) else bin_vals[int(X[j, i])]

            replacing[i] = X_new  # Store transformed binary columns
            bin_info.append([num_cols, n_unique-1])   # Store binary encoding info
        else:
            bin_info.append(-1)

    # Construct final transformed dataset
    X_transformed = []
    for i in range(X.shape[1]):
        if i in replacing:
            X_transformed.append(replacing[i])  # Append binary columns
        else:
            X_transformed.append(X[:, i].reshape(-1, 1))  # Keep original column

    X_encoded = np.hstack(X_transformed)  # Combine into final array
    return X_encoded, bin_info
    
def _restore_binary_encoding(X: np.ndarray, info, X_prob: np.ndarray):
    """Restores the binary encoding for encoded features."""
    X = X.astype(str)
    restored = np.zeros((X.shape[0], len(info)))

    for i in range(len(info)):
        if info[i] != -1:       # If the column is binary encoded, continue
            # Create reverse binary mappings
            bin_map = {format(val, f'0{info[i][0]}b'): val for val in range(2**info[i][0])}

            for j in range(X.shape[0]):     # Look at each row seperately
                bin_value = ""
                for n in range(info[i][0]):       # Obtain the binary value from multiple columns
                    bin_value += X[j, i+n][0]
                int_value = bin_map.get(bin_value)
            
                while int_value > info[i][1]:   # Check if the value exceeds the maximum value for this feature
                    one_indices = np.where(X_prob[j] >= 0.5)[0]     # Get the indices of values that are rounded to 1
                    min_pos = one_indices[np.argmin(X_prob[j][one_indices])]    # Get the position of the lowest probability that is rounded to 1
                    X[j, min_pos] = "0.0"       # Set the lowest probability to 0 to reduce the integer value
                    X_prob[j, min_pos] = 0.0    # Update for future loops

                    bin_value = ""      # Same steps for obtaining integer value
                    for n in range(info[i][0]):     
                        bin_value += X[j, i+n][0]
                    int_value = bin_map.get(bin_value)

                restored[j, i] = int_value
                
            for _ in range(1, info[i][0]):        # Remove binary rows for future loops
                X = np.delete(X, i+1, 1)
        else:
            restored[:, i] = X[:, i]
        
    try:
        restored = restored.astype(float)
        return restored
    except ValueError:
        return restored
