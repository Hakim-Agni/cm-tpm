import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.datasets import load_diabetes
from sklearn.impute import KNNImputer, SimpleImputer
from cm_tpm import CMImputer

def introduce_missingness(data, missing_rate=0.1, random_state=42):
    rng = np.random.RandomState(random_state)  # Ensures reproducibility
    mask = rng.rand(*data.shape) < missing_rate  # Create mask for missing values
    data_missing = data.mask(mask)  # Apply mask
    return data_missing, mask

# Load dataset as a pandas DataFrame
data = load_diabetes(as_frame=True).frame
data.to_csv("evaluation/data/diabetes_complete.csv", index=False)

# Introduce 10% missing values
data_missing, mask = introduce_missingness(data, missing_rate=0.1)

# Display missing value summary
# print(data_missing.isnull().sum())

# Save to CSV for testing
data_missing.to_csv("evaluation/data/diabetes_with_missing.csv", index=False)
print("Dataset with missing values saved as 'diabetes_with_missing.csv'")

imputer = CMImputer(
    missing_values=np.nan,
    n_components=1024,
    latent_dim=32,
    pc_type="factorized",
    missing_strategy="mean",
    ordinal_features=None,
    max_depth=5,
    custom_net=None,
    max_iter=100,
    tol=0.0001,
    lr=0.001,
    smooth=0.000001,
    verbose=0,
    copy=True,
    keep_empty_features=True,
)
#imputer.fit(data)
#data_imputed = imputer.transform(data_missing)
data_imputed = imputer.fit_transform(data_missing)

# Save the imputed dataset
data_imputed.to_csv("evaluation/data/diabetes_imputed_cm.csv", index=False)
print("Imputed dataset saved as 'diabetes_imputed_cm.csv'")

# Select only the originally missing values for comparison
imputed_values = data_imputed.values[mask]
true_values = data.values[mask]

# Compute Error Metrics
mae = mean_absolute_error(true_values, imputed_values)
rmse = np.sqrt(mean_squared_error(true_values, imputed_values))
mape = np.mean(np.abs((true_values - imputed_values) / true_values)) * 100
correlation = np.corrcoef(true_values.flatten(), imputed_values.flatten())[0, 1]

print("CM Imputer (factorized):")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"Correlation between true and imputed values: {correlation:.4f}")


# imputer_spn = CMImputer(
#     missing_values=np.nan,
#     n_components=8,
#     latent_dim=32,
#     pc_type="spn",
#     missing_strategy="integration",
#     ordinal_features=None,
#     max_depth=5,
#     custom_net=None,
#     max_iter=100,
#     tol=0.0001,
#     lr=0.001,
#     smooth=0.000001,
#     random_state=42,
#     verbose=0,
#     copy=True,
#     keep_empty_features=True,
# )
# spn_imputed = imputer_spn.fit_transform(data_missing)

# # Select only the originally missing values for comparison
# spn_values = spn_imputed.values[mask]

# # Compute Error Metrics
# mae = mean_absolute_error(true_values, spn_values)
# rmse = np.sqrt(mean_squared_error(true_values, spn_values))
# mape = np.mean(np.abs((true_values - spn_values) / true_values)) * 100
# correlation = np.corrcoef(true_values.flatten(), spn_values.flatten())[0, 1]

# print("___________________________________________________")
# print("CM Imputer (SPN):")
# print(f"Mean Absolute Error (MAE): {mae:.4f}")
# print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
# print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
# print(f"Correlation between true and imputed values: {correlation:.4f}")


knn_imputer = KNNImputer()
knn_imputed = pd.DataFrame(knn_imputer.fit_transform(data_missing))

knn_values = knn_imputed.values[mask]

# Compute Error Metrics
mae = mean_absolute_error(true_values, knn_values)
rmse = np.sqrt(mean_squared_error(true_values, knn_values))
mape = np.mean(np.abs((true_values - knn_values) / true_values)) * 100
correlation = np.corrcoef(true_values.flatten(), knn_values.flatten())[0, 1]

print("___________________________________________________")
print("KNN Imputer:")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"Correlation between true and imputed values: {correlation:.4f}")

mean_imputer = SimpleImputer()
mean_imputed = pd.DataFrame(mean_imputer.fit_transform(data_missing))

mean_values = mean_imputed.values[mask]

# Compute Error Metrics
mae = mean_absolute_error(true_values, mean_values)
rmse = np.sqrt(mean_squared_error(true_values, mean_values))
mape = np.mean(np.abs((true_values - mean_values) / true_values)) * 100
correlation = np.corrcoef(true_values.flatten(), mean_values.flatten())[0, 1]

print("___________________________________________________")
print("Simple Imputer:")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"Correlation between true and imputed values: {correlation:.4f}")