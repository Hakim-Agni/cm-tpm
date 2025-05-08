import pytest
import numpy as np
import pandas as pd
from cm_tpm._helpers import (
    _load_file, _to_numpy, _restore_format, _missing_to_nan, _all_numeric, is_valid_integer,
    _integer_encoding, _restore_encoding, _binary_encoding, _restore_binary_encoding, _convert_json, _convert_numpy
)

class TestLoadFiletypes:
    def test_load_csv_file(self):
        """Test loading a CSV file."""
        df = _load_file("tests/data/test_data.csv", sep=";", decimal=",")
        assert df.shape == (10, 3)
        assert df.columns.tolist() == ["A", "B", "C"]
        assert df.dtypes.tolist() == [float, float, float]

    def test_load_xlsx_file(self):
        """Test loading a XLSX file."""
        df = _load_file("tests/data/test_data.xlsx")
        assert df.shape == (10, 3)
        assert df.columns.tolist() == ["A", "B", "C"]
        assert df.dtypes.tolist() == [float, float, float]

    def test_load_parquet_file(self):
        """Test loading a Parquet file."""
        df = _load_file("tests/data/test_data.parquet")
        assert df.shape == (10, 3)
        assert df.columns.tolist() == ["A", "B", "C"]
        assert df.dtypes.tolist() == [float, float, float]

    def test_load_feather_file(self):
        """Test loading a Feather file."""
        df = _load_file("tests/data/test_data.feather")
        assert df.shape == (10, 3)
        assert df.columns.tolist() == ["A", "B", "C"]
        assert df.dtypes.tolist() == [float, float, float]

    def test_unsupported_file_format(self):
        """Test loading an unsupported filetype."""
        try:
            _load_file("tests/data/test_data.txt")
            assert False
        except ValueError as e:
            assert str(e) == "Unsupported file format. Please provide a CSV, Excel, Parquet, or Feather file."

    def test_file_not_exists(self):
        """Test loading a file that does not exist."""
        try:
            _load_file("tests/data/non_existent_file.csv")
            assert False
        except FileNotFoundError as e:
            assert str(e) == "[Errno 2] No such file or directory: 'tests/data/non_existent_file.csv'"

class TestToNumpy():
    def test_dataframe_to_numpy(self):
        """Test converting a pandas DataFrame to a NumPy array."""
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        X_np, original_format, columns = _to_numpy(df)
        assert isinstance(X_np, np.ndarray)
        assert X_np.shape == (3, 2)
        assert original_format == "DataFrame"
        assert columns[0] == "A"
        assert columns[1] == "B"

    def test_list_to_numpy(self):
        """Test converting a list to a NumPy array."""
        X_np, original_format, columns = _to_numpy([1, 2, 3])
        assert isinstance(X_np, np.ndarray)
        assert X_np.shape == (3,)
        assert original_format == "list"
        assert columns is None

    def test_numpy_to_numpy(self):
        """Test converting a NumPy array to a NumPy array."""
        X_np, original_format, columns = _to_numpy(np.array([1, 2, 3]))
        assert isinstance(X_np, np.ndarray)
        assert X_np.shape == (3,)
        assert original_format == "ndarray"
        assert columns is None

    def test_unsupported_to_numpy(self):
        """Test converting unsupported data types to a NumPy array."""
        try:
            _, _, _ = _to_numpy("test")
            assert False
        except TypeError as e:
            assert str(e) == "Unsupported data type: str. Expected NumPy ndarray, pandas DataFrame or list."
     
class TestRestoreFormat():
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup method for the test class."""
        self.X_imputed = np.array([[1, 2, 3], [4, 5, 6]])

    def test_restore_dataframe(self):
        """Test restoring data to DataFrame format."""
        columns = ["A", "B", "C"]
        restored = _restore_format(self.X_imputed, original_format="DataFrame", columns=columns)
        assert isinstance(restored, pd.DataFrame)
        assert restored.shape == (2, 3)
        assert restored.columns[0] == columns[0]
        assert restored.columns[1] == columns[1]
        assert restored.columns[2] == columns[2]

    def test_restore_list(self):
        """Test restoring data to list format."""
        restored = _restore_format(self.X_imputed, original_format="list")
        assert isinstance(restored, list)
        assert len(restored) == 2
        assert len(restored[0]) == 3

    def test_restore_numpy(self):
        """Test restoring data to NumPy array."""
        restored = _restore_format(self.X_imputed, original_format="ndarray")
        assert isinstance(restored, np.ndarray)
        assert restored.shape == (2, 3)

    def test_restore_default(self):
        """Test that data is restored to NumPy array by default."""
        restored = _restore_format(self.X_imputed)
        assert isinstance(restored, np.ndarray)
        assert restored.shape == (2, 3)

class TestMissingToNaN():
    def test_nan(self):
        """Test an instance where the missing value is NaN."""
        X = np.array([[np.nan, 0, 1, 2, 3]])
        missing_value = np.nan
        X_nan = _missing_to_nan(X, missing_value)
        assert X_nan.shape == (1, 5)
        assert np.array_equal(X[:, 1:], X_nan[:, 1:])
        assert np.isnan(X_nan[0, 0])

    def test_int(self):
        """Test an instance where the missing value is an integer."""
        X = np.array([[-1, 0, 1, 2, 3]])
        missing_value = -1
        X_nan = _missing_to_nan(X, missing_value)
        assert X_nan.shape == (1, 5)
        assert np.array_equal(X[:, 1:], X_nan[:, 1:])
        assert np.isnan(X_nan[0, 0])

    def test_float(self):
        """Test an instance where the missing value is a float."""
        X = np.array([[0.5, 0, 1, 2, 3]])
        missing_value = 0.5
        X_nan = _missing_to_nan(X, missing_value)
        assert X_nan.shape == (1, 5)
        assert np.array_equal(X[:, 1:], X_nan[:, 1:])
        assert np.isnan(X_nan[0, 0])

    def test_string(self):
        """Test an instance where the missing value is a string."""
        X = np.array([["", 0, 1, 2, 3]])
        missing_value = ""
        X_nan = _missing_to_nan(X, missing_value)
        assert X_nan.shape == (1, 5)
        assert np.array_equal(X[:, 1:], X_nan[:, 1:])
        assert X_nan[0, 0] == "nan" 

    def test_list(self):
        """Test an instance where there are multiple missing values."""
        X = np.array([[-1, 0.5, 1, 2, 3]])
        missing_value = [-1, 0.5]
        X_nan = _missing_to_nan(X, missing_value)
        assert X_nan.shape == (1, 5)
        assert np.array_equal(X[:, 2:], X_nan[:, 2:])
        assert np.isnan(X_nan[0, 0])
        assert np.isnan(X_nan[0, 1])

    def test_list_all(self):
        """Test an instance where there are all types of missing values."""
        X = np.array([[-1, 0.5, "", np.nan, 3]])
        missing_value = [-1, 0.5, "", np.nan]
        X_nan = _missing_to_nan(X, missing_value)
        assert X_nan.shape == (1, 5)
        assert np.array_equal(X[:, 4:], X_nan[:, 4:])
        assert X_nan[0, 0] == "nan" 
        assert X_nan[0, 1] == "nan"
        assert X_nan[0, 2] == "nan"
        assert X_nan[0, 3] == "nan"

class TestAllNumeric():
    def test_all_numeric(self):
        """Test on numerical data."""
        X = np.array([0.5, 0.8, 0.2])
        assert _all_numeric(X)
    
    def test_mixed(self):
        """Test on mixed data."""
        X = np.array(["Yes", 0.8, 0.2])
        assert not _all_numeric(X)

    def test_all_non_numeric(self):
        """Test on non-numerical data."""
        X = np.array(["Yes", "No", "Yes"])
        assert not _all_numeric(X)

    def test_nan(self):
        """Test the function with a nan value."""
        X = np.array([0.5, 0.8, np.nan])
        assert _all_numeric(X)

    def test_nan(self):
        """Test the function with a string nan value."""
        X = np.array([0.5, 0.8, "nan"])
        assert _all_numeric(X)

    def test_non_numeric_nan(self):
        """Test the function with non-numerical data and a nan value."""
        X = np.array(["Yes", 0.8, np.nan])
        assert not _all_numeric(X)

class TestIsValidInteger():
    def test_integer(self):
        """Test on an integer."""
        X = 1
        assert is_valid_integer(X)

    def test_string_integer(self):
        """Test on a string integer."""
        X = "1"
        assert is_valid_integer(X)

    def test_float_integer(self):
        """Test on a float integer."""
        X = 1.0
        assert is_valid_integer(X)

    def test_string_float_integer(self):
        """Test on a string float integer."""
        X = "1.0"
        assert is_valid_integer(X)

    def test_float(self):
        """Test on a float."""
        X = 1.5
        assert not is_valid_integer(X)

    def test_nan(self):
        """Test on a NaN value."""
        X = np.nan
        assert is_valid_integer(X)

class TestIntegerEncoding():
    def test_only_numerical(self):
        """Test encoding data with only numerical values."""
        X = np.array([[0.5, 0.8, 0.2], [0.2, 0.4, 0.1], [0.9, 0.5, 0.6]])
        encoded, mask, info = _integer_encoding(X)
        assert np.array_equal(X, encoded)
        assert np.array_equal(mask, np.array([False, False, False]))
        assert info == {}

    def test_binary_features(self):
        """Test encoding data with binary features."""
        X = np.array([["Yes", 0.8, 0.2], ["No", 0.4, 0.1], ["Yes", 0.5, 0.6]])
        encoded, mask, info = _integer_encoding(X)
        assert encoded[0, 0] == 1
        assert encoded[1, 0] == 0
        assert encoded[2, 0] == 1
        assert np.array_equal(mask, np.array([True, False, False]))
        assert info == {0: {np.str_("No"): 0, np.str_("Yes"): 1}}

    def test_non_numerical_features(self):
        """Test encoding data with non-numerical features."""
        X = np.array([[0.2, "Yes", 0.8], [0.1, "No", 0.4], [0.6, "Maybe", 0.5]])
        encoded, mask, info = _integer_encoding(X)
        assert encoded[0, 1] == 2
        assert encoded[1, 1] == 1
        assert encoded[2, 1] == 0
        assert np.array_equal(mask, np.array([False, True, False]))
        assert info == {1: {np.str_("Maybe"): 0, np.str_("No"): 1, np.str_("Yes"): 2}}

    def test_ordinal_features(self):
        """Test encoding data with ordinal features."""
        X = np.array([[0.2, "Low", 0.8], [0.1, "High", 0.4], [0.6, "Medium", 0.5]])
        ordinal = {1: {"Low": 0, "Medium": 1, "High": 2}}
        encoded, mask, info = _integer_encoding(X, ordinal_features=ordinal)
        assert encoded[0, 1] == 0
        assert encoded[1, 1] == 2
        assert encoded[2, 1] == 1
        assert np.array_equal(mask, np.array([False, True, False]))
        assert info == ordinal

    def test_mixed(self):
        """Test encoding data with mixed features."""
        X = np.array([[0.2, "Low", "Blue"], [0.1, "High", "Red"], [0.6, "Medium", "Yellow"]])
        ordinal = {1: {"Low": 0, "Medium": 1, "High": 2}}
        encoded, mask, info = _integer_encoding(X, ordinal_features=ordinal)
        assert encoded[0, 1] == 0
        assert encoded[1, 1] == 2
        assert encoded[2, 1] == 1
        assert encoded[0, 2] == 0
        assert encoded[1, 2] == 1
        assert encoded[2, 2] == 2
        assert np.array_equal(mask, np.array([False, True, True]))
        assert info == {1: {"Low": 0, "Medium": 1, "High": 2}, 2: {"Blue": 0, "Red": 1, "Yellow": 2}}

class TestRestoreEncoding():
    def test_restore_none(self):
        """Test restoring unencoded data."""
        X = np.array([[0.5, 0.8, 0.2], [0.2, 0.4, 0.1], [0.9, 0.5, 0.6]])
        mask = np.array([False, False, False])
        info = {}
        restored = _restore_encoding(X, mask, info)
        assert np.array_equal(X, restored)

    def test_restore(self):
        """Test restoring encoded data."""
        X = np.array([[0.5, 0.8, 0.2], [0.2, 0.4, 0.1], [0.9, 0.5, 0.6]])
        mask = np.array([False, False, True])
        info = {2: {np.str_("No"): 0, np.str_("Yes"): 1}}
        restored = _restore_encoding(X, mask, info)
        assert restored[0, 2] == "No"
        assert restored[1, 2] == "No"
        assert restored[2, 2] == "Yes"

class TestBinaryEncoding():
    def test_only_numerical(self):
        """Test the binary encoding on numerical data."""
        X = np.array([[0.2, 2., 0.8], [0.1, 1., 0.4], [0.6, 1.4, 0.5]])
        mask = np.array([False, False, False])
        info = {}
        X_encoded, bin_info = _binary_encoding(X, mask, info)
        assert np.array_equal(X, X_encoded)
        assert bin_info == [-1, -1, -1]

    def test_binary_features(self):
        """Test the binary encoding on binary features."""
        X = np.array([[0., 2., 0.8], [0., 1., 0.4], [1., 1.4, 0.5]])
        mask = np.array([True, False, False])
        info = {0: {np.str_("No"): 0, np.str_("Yes"): 1}}
        X_encoded, bin_info = _binary_encoding(X, mask, info)
        assert np.array_equal(X, X_encoded)
        assert bin_info == [[1, 1], -1, -1]

    def test_non_numerical_features(self):
        """Test the binary encoding on non-numerical features"""
        X = np.array([[0., 2., 0.8], [1., 1., 0.4], [2., 1.4, 0.5]])
        mask = np.array([True, False, False])
        info = {0: {np.str_("Maybe"): 0, np.str_("No"): 1, np.str_("Yes"): 2}}
        X_encoded, bin_info = _binary_encoding(X, mask, info)
        assert np.array_equal(X_encoded[:, :2], np.array([[0., 0.], [0., 1.], [1., 0.]]))
        assert np.array_equal(X_encoded[:, 2:], X[:, 1:])
        assert bin_info == [[2, 2], -1, -1]

    def test_larger_non_numerical_features(self):
        """Test the binary encoding on more non-numerical features"""
        X = np.array([[0., 2., 0.8], [1., 1., 0.4], [2., 1.4, 0.5], [5., 1.2, 0.5], 
                      [7., 0.6, 1.3], [4., 0.2, 0.6], [6., 0.9, 0.2], [3., 1.1, 0.4]])
        mask = np.array([True, False, False])
        info = {0: {np.str_("Red"): 0, np.str_("Blue"): 1, np.str_("Yellow"): 2, np.str_("Green"): 3, 
                    np.str_("Orange"): 4, np.str_("Purple"): 5, np.str_("Black"): 6, np.str_("White"): 7}}
        X_encoded, bin_info = _binary_encoding(X, mask, info)
        assert np.array_equal(X_encoded[:, :3], np.array([[0., 0., 0.], [0., 0., 1.], [0., 1., 0.], [1., 0., 1.], 
                                                       [1., 1., 1.], [1., 0., 0.], [1., 1., 0.], [0., 1., 1.]]))
        assert np.array_equal(X_encoded[:, 3:], X[:, 1:])
        assert bin_info == [[3, 7], -1, -1]

    def test_ordinal_features(self):
        """Test the binary encoding on ordinal features."""
        X = np.array([[0.2, 0, 0.8], [0.1, 2, 0.4], [0.6, 1, 0.5]])
        ordinal = {1: {"Low": 0, "Medium": 1, "High": 2}}
        mask = np.array([False, True, False])
        info = ordinal
        X_encoded, bin_info = _binary_encoding(X, mask, info, ordinal_features=ordinal)
        assert np.array_equal(X, X_encoded)
        assert bin_info == [-1, -1, -1]

    def test_mixed(self):
        """Test the binary encoding on mixed features."""
        X = np.array([[0.2, 0, 0], [0.1, 2, 1], [0.6, 1, 2]])
        ordinal = {1: {"Low": 0, "Medium": 1, "High": 2}}
        mask = np.array([False, True, True])
        info = {1: {"Low": 0, "Medium": 1, "High": 2}, 2: {"Blue": 0, "Red": 1, "Yellow": 2}}
        X_encoded, bin_info = _binary_encoding(X, mask, info, ordinal_features=ordinal)
        assert np.array_equal(X[:, :2], X_encoded[:, :2])
        assert np.array_equal(X_encoded[:, 2:], np.array([[0., 0.], [0., 1.], [1., 0.]]))
        assert bin_info == [-1, -1, [2, 2]]

class TestRestoreBinary():
    def test_restore_none(self):
        """Test the restore function when nothing needs to be restored."""
        X_encoded = np.array([[0.2, 2., 0.8], [0.1, 1., 0.4], [0.6, 1.4, 0.5]])
        info = [-1, -1, -1]
        X_decoded = _restore_binary_encoding(X_encoded, info, None)
        assert np.array_equal(X_encoded, X_decoded)

    def test_restore_binary(self):
        """Test the restore function on a binary feature."""
        X_encoded = np.array([[0.2, 0., 0.8], [0.1, 1., 0.4], [0.6, 0., 0.5]])
        info = [-1, [1, 1], -1]
        X_decoded = _restore_binary_encoding(X_encoded, info, None)
        assert np.array_equal(X_encoded, X_decoded)

    def test_restore_non_numerical(self):
        """Test the restore function on a non_numerical feature."""
        X_encoded = np.array([[0.2, 0., 0., 0.8], [0.1, 1., 0., 0.4], [0.6, 1., 1., 0.5]])
        info = [-1, [2, 3], -1]
        X_decoded = _restore_binary_encoding(X_encoded, info, None)
        assert np.array_equal(X_decoded[:, 0], X_encoded[:, 0])
        assert np.array_equal(X_decoded[:, 2], X_encoded[:, 3])
        assert np.array_equal(X_decoded[:, 1], np.array([0., 2., 3.]))

    def test_restore_consecutive_non_numerical(self):
        """Test the restore function on a non_numerical feature."""
        X_encoded = np.array([[0.2, 0., 0., 0., 0., 1., 0.8], [0.1, 1., 0., 1., 0., 1., 0.4], 
                              [0.6, 1., 1., 0., 1., 1., 0.5]])
        info = [-1, [2, 3], [3, 5], -1]
        X_decoded = _restore_binary_encoding(X_encoded, info, None)
        assert np.array_equal(X_decoded[:, 0], X_encoded[:, 0])
        assert np.array_equal(X_decoded[:, 3], X_encoded[:, 6])
        assert np.array_equal(X_decoded[:, 1], np.array([0., 2., 3.]))
        assert np.array_equal(X_decoded[:, 2], np.array([1., 5., 3.]))

    def test_restore_exceeds_maximum(self):
        """Test the restore function when the decoded integer exceeds the maximum value."""
        X_encoded = np.array([[0.2, 0., 0., 1., 0.8], [0.1, 1., 0., 1., 0.4], 
                              [0.6, 0., 1., 1., 0.5]])
        X_probs = np.array([[0.2, 0.1, 0.3, 0.83, 0.8], [0.1, 0.84, 0.1, 0.56, 0.4], 
                              [0.6, 0.3, 0.9, 0.8, 0.5]])
        info = [-1, [3, 4], -1]
        X_decoded = _restore_binary_encoding(X_encoded, info, X_probs)
        assert np.array_equal(X_decoded[:, 0], X_encoded[:, 0])
        assert np.array_equal(X_decoded[:, 2], X_encoded[:, 4])
        assert np.array_equal(X_decoded[:, 1], np.array([1., 4., 3.]))

class TestConvertJSON():
    def test_convert_single_value(self):
        """Test converting sinlge values to JSON format."""
        x1 = 5
        x2 = 0.5
        x3 = "string"
        assert x1 == _convert_json(x1)
        assert x2 == _convert_json(x2)
        assert x3 == _convert_json(x3)

    def test_convert_list(self):
        """Test converting a list to JSON format."""
        x = [1, 2, 3, 4]
        assert x == _convert_json(x)

    def test_convert_np_array(self):
        """Test converting a numpy array to JSON format."""
        x = np.array([1, 2, 3, 4])
        assert _convert_json(x) == [1, 2, 3, 4]

    def test_convert_tuple(self):
        """Test converting a tuple to JSON format."""
        x = (15, np.array([5, 6, 7]))
        assert _convert_json(x) == (15, [5, 6, 7])

class TestConvertNumpy():
    def test_convert_list(self):
        """Test converting a list to a numpy array."""
        x = [1, 2, 3, 4]
        assert np.array_equal(_convert_numpy(x), np.array([1, 2, 3, 4]))

    def test_convert_np_array(self):
        """Test converting a numpy array to a numpy array."""
        x = np.array([1, 2, 3, 4])
        assert np.array_equal(x, _convert_numpy(x))

    def test_convert_tuple(self):
        """Test converting a tuple to numpy format."""
        x = (15, [5, 6, 7])
        assert _convert_numpy(x)[0] == 15
        assert np.array_equal(_convert_numpy(x)[1], np.array([5, 6, 7]))
