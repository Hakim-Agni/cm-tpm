import pytest
import math
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
            latent_dim=8,
            pc_type="spn",
            missing_strategy="ignore",
            net=None,
            max_depth=3,
            max_iter=100,
            tol=1e-3,
            lr=0.01,
            smooth=False,
            random_state=42,
            verbose=2,
            copy=False,
            keep_empty_features=False,
            )
        assert imputer.missing_values == ""
        assert imputer.n_components == 5
        assert imputer.latent_dim == 8
        assert imputer.pc_type == "spn"
        assert imputer.missing_strategy == "ignore"
        assert imputer.net == None
        assert imputer.max_depth == 3
        assert imputer.max_iter == 100
        assert imputer.tol == 1e-3
        assert imputer.lr == 0.01
        assert imputer.smooth == False
        assert imputer.random_state == 42
        assert imputer.verbose == 2
        assert imputer.copy == False
        assert imputer.keep_empty_features == False

    def test_attributes(self):
        """Test the model attributes."""
        imputer = CMImputer(random_state=42)
        assert imputer.is_fitted_ == False
        assert imputer.n_features_in_ == None
        assert imputer.feature_names_in_ == None
        assert imputer.components_ == None
        assert imputer.log_likelihood_ == None
        assert imputer.mean_ == 0.0
        assert imputer.std_ == 1.0
        assert imputer.binary_info_ is None
        assert imputer.encoding_info_ is None
        assert imputer.categorical_info_ is None
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

class TestIntegerEncoding():
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup method for the test class."""
        self.imputer = CMImputer()

    def test_only_numerical(self):
        """Test encoding data with only numerical values."""
        X = np.array([[0.5, 0.8, 0.2], [0.2, 0.4, 0.1], [0.9, 0.5, 0.6]])
        encoded, mask, info = self.imputer._integer_encoding(X)
        assert np.array_equal(X, encoded)
        assert np.array_equal(mask, np.array([False, False, False]))
        assert info == {}

    def test_binary_features(self):
        """Test encoding data with binary features."""
        X = np.array([["Yes", 0.8, 0.2], ["No", 0.4, 0.1], ["Yes", 0.5, 0.6]])
        encoded, mask, info = self.imputer._integer_encoding(X)
        assert encoded[0, 0] == 1
        assert encoded[1, 0] == 0
        assert encoded[2, 0] == 1
        assert np.array_equal(mask, np.array([True, False, False]))
        assert info == {0: {np.str_("No"): 0, np.str_("Yes"): 1}}

    def test_non_numerical_features(self):
        """Test encoding data with non-numerical features."""
        X = np.array([[0.2, "Yes", 0.8], [0.1, "No", 0.4], [0.6, "Maybe", 0.5]])
        encoded, mask, info = self.imputer._integer_encoding(X)
        assert encoded[0, 1] == 2
        assert encoded[1, 1] == 1
        assert encoded[2, 1] == 0
        assert np.array_equal(mask, np.array([False, True, False]))
        assert info == {1: {np.str_("Maybe"): 0, np.str_("No"): 1, np.str_("Yes"): 2}}

class TestRestoreEncoding():
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup method for the test class."""
        self.imputer = CMImputer()

    def test_restore_none(self):
        """Test restoring unencoded data."""
        X = np.array([[0.5, 0.8, 0.2], [0.2, 0.4, 0.1], [0.9, 0.5, 0.6]])
        mask = np.array([False, False, False])
        info = {}
        restored = self.imputer._restore_encoding(X, mask, info)
        assert np.array_equal(X, restored)

    def test_restore(self):
        """Test restoring encoded data."""
        X = np.array([[0.5, 0.8, 0.2], [0.2, 0.4, 0.1], [0.9, 0.5, 0.6]])
        mask = np.array([False, False, True])
        info = {2: {np.str_("No"): 0, np.str_("Yes"): 1}}
        restored = self.imputer._restore_encoding(X, mask, info)
        assert restored[0, 2] == "No"
        assert restored[1, 2] == "No"
        assert restored[2, 2] == "Yes"
        
class TestPreprocess():
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup method for the test class."""
        self.imputer = CMImputer()

    def test_preprocess_ints(self):
        """Test preprocessing an array with integers."""
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        X_preprocessed, _, _, _ = self.imputer._preprocess_data(X)
        assert np.array_equal(X_preprocessed, np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]))

    def test_preprocess_non_nan(self):
        """Test preprocessing an array with a different missing value than NaN."""
        self.imputer.missing_values = -1
        X = np.array([[1., 2., -1], [4., 5., 6.], [7., 8., 9.]])
        X_preprocessed, _, _, _ = self.imputer._preprocess_data(X)
        assert X.shape == X_preprocessed.shape
        assert np.isnan(X_preprocessed[0, 2])

    def test_preprocess_remove_nan_features(self):
        """Test preprocessing removes NaN features."""
        X = np.array([[1., 2., np.nan], [4., 5., np.nan], [7., 8., np.nan]])
        X_preprocessed, _, _, _ = self.imputer._preprocess_data(X, train=True)
        assert X_preprocessed.shape[0] == 3
        assert X_preprocessed.shape[1] == 2
        assert not np.isnan(X_preprocessed).any()

    def test_preprocess_remove_missing_features(self):
        """Test preprocessing removes other missing features."""
        self.imputer.missing_values = -10
        X = np.array([[1., 2., -10], [4., 5., -10], [7., 8., -10]])
        X_preprocessed, _, _, _ = self.imputer._preprocess_data(X, train=True)
        assert X_preprocessed.shape[0] == 3
        assert X_preprocessed.shape[1] == 2
        assert not np.isnan(X_preprocessed).any()
        assert not np.any(X_preprocessed == -10)

    def test_preprocess_fill_nan_features(self):
        """Test preprocessing fills NaN features."""
        self.imputer.keep_empty_features = True
        X = np.array([[1., 2., np.nan], [4., 5., np.nan], [7., 8., np.nan]])
        X_preprocessed, _, _, _ = self.imputer._preprocess_data(X)
        assert np.array_equal(X_preprocessed, np.array([[1., 2., 0.], [4., 5., 0.], [7., 8., 0.]]))

    def test_preprocess_fill_missing_features(self):
        """Test preprocessing fills other missing features."""
        self.imputer.missing_values = -1
        self.imputer.keep_empty_features = True
        X = np.array([[1., 2., -1], [4., 5., -1], [7., 8., -1]])
        X_preprocessed, _, _, _ = self.imputer._preprocess_data(X)
        assert np.array_equal(X_preprocessed, np.array([[1., 2., 0.], [4., 5., 0.], [7., 8., 0.]]))

    def test_preprocess_mean_std(self):
        """Test updating the mean and std while preprocessing."""
        X = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
        X_preprocessed, _, _, _ = self.imputer._preprocess_data(X, train=True)
        assert np.array_equal(self.imputer.mean_, np.array([4., 5., 6.]))
        assert np.array_equal(self.imputer.std_, np.array([math.sqrt(6), math.sqrt(6), math.sqrt(6)]))

    def test_preprocess_binary_info(self):
        """Test if the binary info is set correctly during preprocessing"""
        X = np.array([[0, 2., 3.], [1, 5., 6.], [0, 8., 9.]])
        X_preprocessed, binary_mask, _, _ = self.imputer._preprocess_data(X, train=True)
        assert np.array_equal(binary_mask, np.array([True, False, False]))
        assert X_preprocessed[0, 0] == 0
        assert X_preprocessed[1, 0] == 1
        assert X_preprocessed[2, 0] == 0

    def test_preprocess_binary_string(self):
        """Test if binary values are converted to 0/1."""
        X = np.array([["Yes", 2., 3.], ["No", 5., 6.], ["Yes", 8., 9.]])
        X_preprocessed, binary_mask, _, _ = self.imputer._preprocess_data(X, train=True)
        assert np.array_equal(binary_mask, np.array([True, False, False]))
        assert X_preprocessed[0, 0] == 1
        assert X_preprocessed[1, 0] == 0
        assert X_preprocessed[2, 0] == 1

    def test_preprocess_non_numerical(self):
        """Test if non numerical feature values are converted to integers."""
        X = np.array([["Yes", "Medium", 3.], ["No", "High", 6.], ["Maybe", "Low", 9.]])
        X_preprocessed, binary_mask, (encoding_mask, _), _ = self.imputer._preprocess_data(X, train=True)
        assert np.array_equal(binary_mask, np.array([False, False, False]))
        assert np.array_equal(encoding_mask, np.array([True, True, False]))

class TestFit():
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup method for the test class."""
        self.imputer = CMImputer(n_components=1)

    def test_fitted(self):
        """Test the is_fitted_ attribute."""
        assert self.imputer.is_fitted_ == False
        self.imputer.fit(np.array([[1, 2, 3], [4, 5, 6]]))
        assert self.imputer.is_fitted_ == True

    def test_n_features_in(self):
        """Test the n_features_in_ attribute."""
        assert self.imputer.n_features_in_ is None
        self.imputer.fit(np.array([[1, 2, 3], [4, 5, 6]]))
        assert self.imputer.n_features_in_ == 3

    def test_feature_names_in(self):
        """Test the feature_names_in_ attribute."""
        assert self.imputer.feature_names_in_ is None
        self.imputer.fit(pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}))
        assert np.array_equal(self.imputer.feature_names_in_, ["A", "B"])

    def test_no_feature_names(self):
        """Test the feature_names_in_ attribute without feature names."""
        assert self.imputer.n_features_in_ is None
        self.imputer.fit(np.array([[1, 2, 3], [4, 5, 6]]))
        assert self.imputer.feature_names_in_ is None

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
        self.imputer = CMImputer(n_components=1)

    def test_transform_no_fit(self):
        """Test transforming data without fitting the imputer."""
        try:
            self.imputer.transform(np.array([[1, 2, 3], [4, 5, 6]]))
            assert False
        except ValueError as e:
            assert str(e) == "The model has not been fitted yet. Please call the fit method first."

    def test_transform_no_missing(self):
        """Test transforming data without missing values"""
        self.imputer.missing_values = -1
        X = np.array([[1, 2, 3], [4, 5, 6]])
        imputer = self.imputer.fit(X)
        with pytest.warns(UserWarning, match="No missing values detected in input data, transformation has no effect. Did you set the correct missing value: '-1'?"):
            X_imputed = imputer.transform(X)
            assert np.array_equal(X_imputed, X)

    def test_transform_numpy(self):
        """Test the transform method on a NumPy array."""
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        imputer = self.imputer.fit(X)
        X_missing = np.array([[np.nan, 2., 3.], [4., 5., 6.]])
        X_imputed = imputer.transform(X_missing)
        assert isinstance(X_imputed, np.ndarray)
        assert X_imputed.shape == (2, 3)
        assert not np.isnan(X_imputed).any()

    def test_transform_dataframe(self):
        """Test the transform method on a pandas DataFrame."""
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        imputer = self.imputer.fit(df)
        df_missing = pd.DataFrame({"A": [np.nan, 2., 3.], "B": [4., 5., 6.]})
        X_imputed = imputer.transform(df_missing)
        assert isinstance(X_imputed, pd.DataFrame)
        assert X_imputed.shape == (3, 2)
        assert X_imputed.columns[0] == "A"
        assert X_imputed.columns[1] == "B"
        assert not X_imputed.isnull().values.any()

    def test_transform_list(self):
        """Test the transform method on a list."""
        X = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        imputer = self.imputer.fit(X)
        X_missing = [[np.nan, 2., 3.], [4., 5., 6.]]
        X_imputed = imputer.transform(X_missing)
        assert isinstance(X_imputed, list)
        assert len(X_imputed) == 2
        assert len(X_imputed[0]) == 3
        assert not np.isnan(X_imputed).any()

    def test_transform_file(self):
        """Test the transform method on a file."""
        if os.path.isfile("tests/data/test_data_imputed.csv"):
            os.remove("tests/data/test_data_imputed.csv")
        imputer = self.imputer.fit("tests/data/test_data.csv", sep=";", decimal=",")
        with pytest.warns(UserWarning, match="No missing values detected in input data, transformation has no effect. Did you set the correct missing value: 'nan'?"):
            X_imputed = imputer.transform("tests/data/test_data.csv", sep=';', decimal=',')
            assert isinstance(X_imputed, np.ndarray)
            assert X_imputed.shape == (10, 3)
            assert os.path.exists("tests/data/test_data_imputed.csv")

    def test_transform_save_path_from_file(self):
        """Test saving the imputed data from a file to a file."""
        if os.path.isfile("tests/data/test_data_save_path_file.parquet"):
            os.remove("tests/data/test_data_save_path_file.parquet")
        imputer = self.imputer.fit("tests/data/test_data.parquet")
        with pytest.warns(UserWarning, match="No missing values detected in input data, transformation has no effect. Did you set the correct missing value: 'nan'?"):
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
        with pytest.warns(UserWarning, match="No missing values detected in input data, transformation has no effect. Did you set the correct missing value: 'nan'?"):
            X_imputed = imputer.transform(X, save_path="tests/data/test_data_save_path_data.feather")
            assert isinstance(X_imputed, np.ndarray)
            assert X_imputed.shape == (2, 3)
            assert os.path.exists("tests/data/test_data_save_path_data.feather")

    def test_transform_non_nan(self):
        """Test the transform method with a different missing value than nan."""
        self.imputer.missing_values = -1
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        imputer = self.imputer.fit(X)
        X_missing = np.array([[-1, 2., 3.], [4., 5., 6.]])
        X_imputed = imputer.transform(X_missing)
        assert isinstance(X_imputed, np.ndarray)
        assert X_imputed.shape == (2, 3)
        assert not np.any(X_imputed == -1)

    def test_transform_seed(self):
        imputer1 = CMImputer(n_components=1, random_state=42)
        imputer2 = CMImputer(n_components=1, random_state=42)
        X = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
        imputer1.fit(X)
        imputer2.fit(X)
        X_missing = np.array([[np.nan, 2., 3.], [4., 5., 6.]])
        X_imputed1 = imputer1.transform(X_missing)
        X_imputed2 = imputer2.transform(X_missing)
        assert np.array_equal(X_imputed1, X_imputed2)

    def test_transform_binary(self):
        """Test the transform method with a binary feature"""
        X = np.array([[1, 2, 3], [0, 5, 6], [0, 3, 2]])
        imputer = self.imputer.fit(X)
        X_missing = np.array([[np.nan, 2., 3.], [1, 5., 6.]])
        X_imputed = imputer.transform(X_missing)
        assert isinstance(X_imputed, np.ndarray)
        assert X_imputed.shape == (2, 3)
        assert not np.isnan(X_imputed).any()
        assert X_imputed[0, 0] == 0 or X_imputed[0, 0] == 1

class TestFitTransform():
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup method for the test class."""
        self.imputer = CMImputer(n_components=1)

    def test_fit_transform(self):
        """Test the fit transform function"""
        X_missing = np.array([[1., 2., np.nan], [4., 5., 6.], [7., 8., 9.]])
        X_imputed = self.imputer.fit_transform(X_missing)
        assert X_imputed.shape[0] == 3
        assert X_imputed.shape[1] == 3
        assert not np.isnan(X_imputed).any()

class TestParams():
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup method for the test class."""
        self.imputer = CMImputer(
            missing_values="",
            n_components=5,
            latent_dim=8,
            pc_type="spn",
            missing_strategy="ignore",
            net=None,
            max_depth=3,
            max_iter=100,
            tol=1e-3,
            lr=0.01,
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
        assert params["latent_dim"] == 8
        assert params["pc_type"] == "spn"
        assert params["missing_strategy"] == "ignore"
        assert params["net"] is None
        assert params["max_depth"] == 3
        assert params["max_iter"] == 100
        assert params["tol"] == 1e-3
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
            latent_dim=4,
            pc_type="clt",
            missing_strategy="integration",
            max_depth=5,
            max_iter=200,
            tol=1e-4,
            lr=0.001,
            smooth=True,
            random_state=43,
            verbose=1,
            copy=True,
            keep_empty_features=True,
            )
        assert np.isnan(self.imputer.missing_values)
        assert self.imputer.n_components == 10
        assert self.imputer.latent_dim == 4
        assert self.imputer.pc_type == "clt"
        assert self.imputer.missing_strategy == "integration"
        assert self.imputer.max_depth == 5
        assert self.imputer.max_iter == 200
        assert self.imputer.tol == 1e-4
        assert self.imputer.lr == 0.001
        assert self.imputer.smooth == True
        assert self.imputer.random_state == 43
        assert self.imputer.verbose == 1
        assert self.imputer.copy == True
        assert self.imputer.keep_empty_features == True
