import pytest
import numpy as np
import pandas as pd
from cm_tpm import CMImputer

class TestClass:
    def test_instance(self):
        """
        Test the instantiation of the CMImputer class.
        """
        imputer = CMImputer()
        assert imputer is not None

    def test_random_state(self):
        """
        Test the random_state parameter.
        """
        imputer = CMImputer(random_state=42)
        assert imputer.random_state == 42

class TestLoadFiletypes:
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """
        Setup method for the test class.
        """
        self.imputer = CMImputer()

    def test_load_csv_file(self):
        """
        Test loading a CSV file.
        """
        df = self.imputer._load_file("tests/data/test_data.csv")
        assert df.shape == (10, 3)
        assert df.columns.tolist() == ["A", "B", "C"]
        assert df.dtypes.tolist() == [float, float, float]

    def test_load_xlsx_file(self):
        """
        Test loading a XLSX file.
        """
        df = self.imputer._load_file("tests/data/test_data.xlsx")
        assert df.shape == (10, 3)
        assert df.columns.tolist() == ["A", "B", "C"]
        assert df.dtypes.tolist() == [float, float, float]

    def test_load_parquet_file(self):
        """
        Test loading a Parquet file.
        """
        df = self.imputer._load_file("tests/data/test_data.parquet")
        assert df.shape == (10, 3)
        assert df.columns.tolist() == ["A", "B", "C"]
        assert df.dtypes.tolist() == [float, float, float]

    def test_load_feather_file(self):
        """
        Test loading a Feather file.
        """
        df = self.imputer._load_file("tests/data/test_data.feather")
        assert df.shape == (10, 3)
        assert df.columns.tolist() == ["A", "B", "C"]
        assert df.dtypes.tolist() == [float, float, float]

    def test_unsupported_file_format(self):
        """
        Test loading an unsupported filetype.
        """
        try:
            self.imputer._load_file("tests/data/test_data.txt")
            assert False
        except ValueError as e:
            assert str(e) == "Unsupported file format. Please provide a CSV, Excel, Parquet, or Feather file."

    def test_file_not_exists(self):
        """
        Test loading a file that does not exist.
        """
        try:
            self.imputer._load_file("tests/data/non_existent_file.csv")
            assert False
        except FileNotFoundError as e:
            assert str(e) == "[Errno 2] No such file or directory: 'tests/data/non_existent_file.csv'"

class TestToNumpy():
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """
        Setup method for the test class.
        """
        self.imputer = CMImputer()

    def test_dataframe_to_numpy(self):
        """
        Test converting a pandas DataFrame to a NumPy array.
        """
        self.imputer = CMImputer()
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        X_np, original_format, columns = self.imputer._to_numpy(df)
        assert X_np.shape == (3, 2)
        assert isinstance(X_np, np.ndarray)
        assert original_format == "DataFrame"
        assert columns[0] == "A"
        assert columns[1] == "B"

    def test_list_to_numpy(self):
        """
        Test converting a list to a NumPy array.
        """
        X_np, original_format, columns = self.imputer._to_numpy([1, 2, 3])
        assert X_np.shape == (3,)
        assert isinstance(X_np, np.ndarray)
        assert original_format == "list"
        assert columns is None

    def test_ndarray_to_numpy(self):
        """
        Test converting a NumPy array to a NumPy array.
        """
        X_np, original_format, columns = self.imputer._to_numpy(np.array([1, 2, 3]))
        assert X_np.shape == (3,)
        assert isinstance(X_np, np.ndarray)
        assert original_format == "ndarray"
        assert columns is None

    def test_unsupported_to_numpy(self):
        """
        Test converting unsupported data types to a NumPy array.
        """
        try:
            X_np, original_format, columns = self.imputer._to_numpy("test")
            assert False
        except ValueError as e:
            assert str(e) == "Unsupported data type. Please provide a NumPy array, pandas DataFrame or list."
     