import pytest
import numpy as np
import pandas as pd
import os.path
from cm_tpm import CMImputer

class TestClass:
    def test_instance(self):
        """Test the instantiation of the CMImputer class."""
        imputer = CMImputer()
        assert imputer is not None

    def test_parameters(self):
        """Test the model parameters."""
        imputer = CMImputer(
            missing_values="",
            n_components=5,
            max_depth=3,
            max_iter=100,
            tol=1e-3,
            weight_sharing=False,
            smooth=False,
            random_state=42,
            verbose=2,
            copy=False,
            keep_empty_features=False,
            )
        assert imputer.missing_values == ""
        assert imputer.n_components == 5
        assert imputer.max_depth == 3
        assert imputer.max_iter == 100
        assert imputer.tol == 1e-3
        assert imputer.weight_sharing == False
        assert imputer.smooth == False
        assert imputer.random_state == 42
        assert imputer.verbose == 2
        assert imputer.copy == False
        assert imputer.keep_empty_features == False

    def test_attributes(self):
        """Test the model attributes."""
        imputer = CMImputer(random_state=42)
        assert imputer.is_fitted_ == False
        assert imputer.n_features_ == None
        assert imputer.feature_names_in_ == None
        assert imputer.components_ == None
        assert imputer.log_likelihood_ == None
        assert np.array_equal(
            imputer.random_state_.get_state()[1], 
            np.random.RandomState(42).get_state()[1]
        )  

class TestLoadFiletypes:
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup method for the test class."""
        self.imputer = CMImputer()

    def test_load_csv_file(self):
        """Test loading a CSV file."""
        df = self.imputer._load_file("tests/data/test_data.csv", sep=";", decimal=",")
        assert df.shape == (10, 3)
        assert df.columns.tolist() == ["A", "B", "C"]
        assert df.dtypes.tolist() == [float, float, float]

    def test_load_xlsx_file(self):
        """Test loading a XLSX file."""
        df = self.imputer._load_file("tests/data/test_data.xlsx")
        assert df.shape == (10, 3)
        assert df.columns.tolist() == ["A", "B", "C"]
        assert df.dtypes.tolist() == [float, float, float]

    def test_load_parquet_file(self):
        """Test loading a Parquet file."""
        df = self.imputer._load_file("tests/data/test_data.parquet")
        assert df.shape == (10, 3)
        assert df.columns.tolist() == ["A", "B", "C"]
        assert df.dtypes.tolist() == [float, float, float]

    def test_load_feather_file(self):
        """Test loading a Feather file."""
        df = self.imputer._load_file("tests/data/test_data.feather")
        assert df.shape == (10, 3)
        assert df.columns.tolist() == ["A", "B", "C"]
        assert df.dtypes.tolist() == [float, float, float]

    def test_unsupported_file_format(self):
        """Test loading an unsupported filetype."""
        try:
            self.imputer._load_file("tests/data/test_data.txt")
            assert False
        except ValueError as e:
            assert str(e) == "Unsupported file format. Please provide a CSV, Excel, Parquet, or Feather file."

    def test_file_not_exists(self):
        """Test loading a file that does not exist."""
        try:
            self.imputer._load_file("tests/data/non_existent_file.csv")
            assert False
        except FileNotFoundError as e:
            assert str(e) == "[Errno 2] No such file or directory: 'tests/data/non_existent_file.csv'"

class TestToNumpy():
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup method for the test class."""
        self.imputer = CMImputer()

    def test_dataframe_to_numpy(self):
        """Test converting a pandas DataFrame to a NumPy array."""
        self.imputer = CMImputer()
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        X_np, original_format, columns = self.imputer._to_numpy(df)
        assert isinstance(X_np, np.ndarray)
        assert X_np.shape == (3, 2)
        assert original_format == "DataFrame"
        assert columns[0] == "A"
        assert columns[1] == "B"

    def test_list_to_numpy(self):
        """Test converting a list to a NumPy array."""
        X_np, original_format, columns = self.imputer._to_numpy([1, 2, 3])
        assert isinstance(X_np, np.ndarray)
        assert X_np.shape == (3,)
        assert original_format == "list"
        assert columns is None

    def test_numpy_to_numpy(self):
        """Test converting a NumPy array to a NumPy array."""
        X_np, original_format, columns = self.imputer._to_numpy(np.array([1, 2, 3]))
        assert isinstance(X_np, np.ndarray)
        assert X_np.shape == (3,)
        assert original_format == "ndarray"
        assert columns is None

    def test_unsupported_to_numpy(self):
        """Test converting unsupported data types to a NumPy array."""
        try:
            _, _, _ = self.imputer._to_numpy("test")
            assert False
        except ValueError as e:
            assert str(e) == "Unsupported data type. Please provide a NumPy array, pandas DataFrame or list."
     
class TestRestoreFormat():
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup method for the test class."""
        self.imputer = CMImputer()
        self.X_imputed = np.array([[1, 2, 3], [4, 5, 6]])

    def test_restore_dataframe(self):
        """Test restoring data to DataFrame format"""
        columns = ["A", "B", "C"]
        restored = self.imputer._restore_format(self.X_imputed, original_format="DataFrame", columns=columns)
        assert isinstance(restored, pd.DataFrame)
        assert restored.shape == (2, 3)
        assert restored.columns[0] == columns[0]
        assert restored.columns[1] == columns[1]
        assert restored.columns[2] == columns[2]

    def test_restore_list(self):
        """Test restoring data to list format"""
        restored = self.imputer._restore_format(self.X_imputed, original_format="list")
        assert isinstance(restored, list)
        assert len(restored) == 2
        assert len(restored[0]) == 3

    def test_restore_numpy(self):
        """Test restoring data to NumPy array"""
        restored = self.imputer._restore_format(self.X_imputed, original_format="ndarray")
        assert isinstance(restored, np.ndarray)
        assert restored.shape == (2, 3)

    def test_restore_default(self):
        """Test that data is restored to NumPy array by default"""
        restored = self.imputer._restore_format(self.X_imputed)
        assert isinstance(restored, np.ndarray)
        assert restored.shape == (2, 3)

class TestFit():
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup method for the test class."""
        self.imputer = CMImputer()

    def test_fitted(self):
        """Test the is_fitted attribute."""
        assert self.imputer.is_fitted_ == False
        self.imputer.fit(np.array([[1, 2, 3], [4, 5, 6]]))
        assert self.imputer.is_fitted_ == True

    def test_fit_numpy(self):
        """Test fitting a NumPy array."""
        X = np.array([[1, 2, 3], [4, 5, 6]])
        imputer = self.imputer.fit(X)
        assert imputer is not None

    def test_fit_dataframe(self):
        """Test fitting a pandas DataFrame."""
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        imputer = self.imputer.fit(df)
        assert imputer is not None

    def test_fit_list(self):
        """Test fitting a list."""
        X = [[1, 2, 3], [4, 5, 6]]
        imputer = self.imputer.fit(X)
        assert imputer is not None

    def test_fit_file(self):
        """Test fitting data from file."""
        imputer = self.imputer.fit("tests/data/test_data.csv")
        assert imputer is not None

    def test_fit_unsupported(self):
        """Test fitting an unsupported data type."""
        try:
            self.imputer.fit(0)
            assert False
        except ValueError as e:
            assert str(e) == "Unsupported data type. Please provide a NumPy array, pandas DataFrame or list."

class TestTransform():
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup method for the test class."""
        self.imputer = CMImputer()

    def test_transform_no_fit(self):
        """Test transforming data without fitting the imputer."""
        try:
            self.imputer.transform(np.array([[1, 2, 3], [4, 5, 6]]))
            assert False
        except ValueError as e:
            assert str(e) == "The model has not been fitted yet. Please call the fit method first."

    def test_transform_numpy(self):
        """Test the transform method on a NumPy array."""
        X = np.array([[1, 2, 3], [4, 5, 6]])
        imputer = self.imputer.fit(X)
        X_imputed = imputer.transform(X)
        assert isinstance(X_imputed, np.ndarray)
        assert X_imputed.shape == (2, 3)

    def test_transform_dataframe(self):
        """Test the transform method on a pandas DataFrame."""
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        imputer = self.imputer.fit(df)
        X_imputed = imputer.transform(df)
        assert isinstance(X_imputed, pd.DataFrame)
        assert X_imputed.shape == (3, 2)

    def test_transform_list(self):
        """Test the transform method on a list."""
        X = [[1, 2, 3], [4, 5, 6]]
        imputer = self.imputer.fit(X)
        X_imputed = imputer.transform(X)
        assert isinstance(X_imputed, list)
        assert len(X_imputed) == 2
        assert len(X_imputed[0]) == 3

    def test_transform_file(self):
        """Test the transform method on a file."""
        if os.path.isfile("tests/data/test_data_imputed.csv"):
            os.remove("tests/data/test_data_imputed.csv")
        imputer = self.imputer.fit("tests/data/test_data.csv", sep=";", decimal=",")
        X_imputed = imputer.transform("tests/data/test_data.csv", sep=';', decimal=',')
        assert isinstance(X_imputed, np.ndarray)
        assert X_imputed.shape == (10, 3)
        assert os.path.exists("tests/data/test_data_imputed.csv")

    def test_transform_save_path_from_file(self):
        """Test saving the imputed data from a file to a file."""
        if os.path.isfile("tests/data/test_data_save_path_file.parquet"):
            os.remove("tests/data/test_data_save_path_file.parquet")
        imputer = self.imputer.fit("tests/data/test_data.parquet")
        X_imputed = imputer.transform("tests/data/test_data.parquet", save_path="tests/data/test_data_save_path_file.parquet")
        assert isinstance(X_imputed, np.ndarray)
        assert X_imputed.shape == (10, 3)
        assert os.path.exists("tests/data/test_data_save_path_file.parquet")


    def test_transform_save_path_from_data(self):
        """Test saving the imputed data to a file."""
        if os.path.isfile("tests/data/test_data_save_path_data.feather"):
            os.remove("tests/data/test_data_save_path_data.feather")
        X = np.array([[1, 2, 3], [4, 5, 6]])
        imputer = self.imputer.fit(X)
        X_imputed = imputer.transform(X, save_path="tests/data/test_data_save_path_data.feather")
        assert isinstance(X_imputed, np.ndarray)
        assert X_imputed.shape == (2, 3)
        assert os.path.exists("tests/data/test_data_save_path_data.feather")

class TestParams():
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup method for the test class."""
        self.imputer = CMImputer(
            missing_values="",
            n_components=5,
            max_depth=3,
            max_iter=100,
            tol=1e-3,
            weight_sharing=False,
            smooth=False,
            random_state=42,
            verbose=2,
            copy=False,
            keep_empty_features=False,
            )

    def test_get_params(self):
        """Test getting parameters."""
        params = self.imputer.get_params()
        assert params["missing_values"] == ""
        assert params["n_components"] == 5
        assert params["max_depth"] == 3
        assert params["max_iter"] == 100
        assert params["tol"] == 1e-3
        assert params["weight_sharing"] == False
        assert params["smooth"] == False
        assert params["random_state"] == 42
        assert params["verbose"] == 2
        assert params["copy"] == False
        assert params["keep_empty_features"] == False

    def test_set_params(self):
        """Test setting parameters."""
        self.imputer.set_params(
            missing_values=np.nan, 
            n_components=10,
            max_depth=5,
            max_iter=200,
            tol=1e-4,
            weight_sharing=True,
            smooth=True,
            random_state=43,
            verbose=1,
            copy=True,
            keep_empty_features=True,
            )
        assert np.isnan(self.imputer.missing_values)
        assert self.imputer.n_components == 10
        assert self.imputer.max_depth == 5
        assert self.imputer.max_iter == 200
        assert self.imputer.tol == 1e-4
        assert self.imputer.weight_sharing == True
        assert self.imputer.smooth == True
        assert self.imputer.random_state == 43
        assert self.imputer.verbose == 1
        assert self.imputer.copy == True
        assert self.imputer.keep_empty_features == True
