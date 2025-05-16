import os
import numpy as np
import pandas as pd
import time
from torch import nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.datasets import load_diabetes, load_breast_cancer, load_digits, load_iris, load_linnerud, load_wine
import torchvision
from sklearn.impute import KNNImputer, SimpleImputer
from ucimlrepo import fetch_ucirepo
from cm_tpm import CMImputer

# CMImputer Settings
settings = 3                    # 0 is high fidelity, 1 is medium, 2 is fast, 3 is image, 4 is custom
random_state = 42

# Dataset Settings
# Complete datasets
diabetes = True                 # Medium sized; numerical and integer
breast_cancer = False            # Large sized; numerical and binary
digits = False                   # Very large sized; integer; image
fashion = False                  # Very large sized; numerical, image
iris = False                     # Small sized; numerical and binary
linnerud = False                 # Small sized; integer
mushroom = False                  # Very large sized; categorical and binary
wine = False                     # Medium sized; numerical and binary

# Datasets with missing values
credit = False                     # Large sized; numerical, integer and categorical
titanic = False                   # Large sized; numerical, categorical and binary

datasets = {
    "diabetes": diabetes,
    "breast_cancer": breast_cancer,
    "digits": digits,
    "fashion": fashion,
    "iris": iris,
    "linnerud": linnerud,
    "mushroom": mushroom,
    "wine": wine,
    "credit": credit,
    "titanic": titanic,
    }

# Imputer Settings
use_cm_imputer = True
use_knn_imputer = False
use_simple_imputer = False
imputers = {
    "cm_imputer": use_cm_imputer,
    "knn_imputer": use_knn_imputer,
    "simple_imputer": use_simple_imputer,
    }


# Function to introduce missingness in the dataset
def introduce_missingness(data, missing_rate=0.1, random_state=42):
    rng = np.random.RandomState(random_state)  # Ensures reproducibility
    mask = rng.rand(*data.shape) < missing_rate  # Create mask for missing values
    data_missing = data.mask(mask)  # Apply mask
    return data_missing, mask

def run_evaluation(cm_imputer=CMImputer(), print_results=True):
    scores = {}
    for dataset_name, use_dataset in datasets.items():
        if not use_dataset:
            continue

        categorical = False
        missing = False

        if dataset_name == "diabetes":
            # Load dataset as a pandas DataFrame
            data = load_diabetes(as_frame=True).frame
            os.makedirs("evaluation/data/diabetes", exist_ok=True)  # Create directory if it doesn't exist
            path = "evaluation/data/diabetes/diabetes_"
        elif dataset_name == "breast_cancer":
            data = load_breast_cancer(as_frame=True).frame
            os.makedirs("evaluation/data/breast_cancer", exist_ok=True)  # Create directory if it doesn't exist
            path = "evaluation/data/breast_cancer/breast_cancer_"
        elif dataset_name == "digits":
            data = load_digits(as_frame=True).frame
            os.makedirs("evaluation/data/digits", exist_ok=True)
            path = "evaluation/data/digits/digits_"
        elif dataset_name == "fashion":
            fashion_mnist = torchvision.datasets.FashionMNIST(
                root="/data", train=True, download=True, transform=torchvision.transforms.ToTensor()
            )
            images = [img for img, label in list(fashion_mnist)[:5000]]  # First 5000 samples
            data_np = np.stack([img.squeeze().numpy().flatten() for img in images])
            data = pd.DataFrame(data_np)
            data = data[:5000]  # Reduce amount of samples
            os.makedirs("evaluation/data/fashion", exist_ok=True)
            path = "evaluation/data/fashion/fashion_"
        elif dataset_name == "iris":
            data = load_iris(as_frame=True).frame
            os.makedirs("evaluation/data/iris", exist_ok=True)
            path = "evaluation/data/iris/iris_"
        elif dataset_name == "linnerud":
            data = load_linnerud(as_frame=True).frame
            os.makedirs("evaluation/data/linnerud", exist_ok=True)
            path = "evaluation/data/linnerud/linnerud_"
        elif dataset_name == "mushroom":
            categorical = True
            mushroom = fetch_ucirepo(id=73)
            data = pd.DataFrame(mushroom.data.features, columns=mushroom.feature_names)
            # Fix column with missing values (stalk-root)
            #data["stalk-root"] = data["stalk-root"].astype("string")
            #data = data.drop(columns=["stalk-root"])
            os.makedirs("evaluation/data/mushroom", exist_ok=True)
            path = "evaluation/data/mushroom/mushroom_"
        elif dataset_name == "wine":
            data = load_wine(as_frame=True).frame
            os.makedirs("evaluation/data/wine", exist_ok=True)
            path = "evaluation/data/wine/wine_"
        elif dataset_name == "credit":
            categorical = True
            missing = True
            credit = fetch_ucirepo(id=27)
            data = pd.DataFrame(credit.data.features, columns=credit.feature_names)
            data_missing = data.copy()
            os.makedirs("evaluation/data/credit", exist_ok=True)
            path = "evaluation/data/credit/credit_"
        elif dataset_name == "titanic":
            categorical = True
            missing = True
            data = pd.read_csv("evaluation/data/titanic/titanic_complete.csv")
            data_missing = data.copy()
            path = "evaluation/data/titanic/titanic_"

        data.to_csv(path + "complete.csv", index=False)

        # Introduce 10% missing values
        if not missing:
            data_missing, mask = introduce_missingness(data, missing_rate=0.1)

        # Display missing value summary
        # print(data_missing.isnull().sum())

        # Save to CSV for testing
        data_missing.to_csv(path + "with_missing.csv", index=False)
        if print_results:
            print(f"Dataset with missing values saved as '{dataset_name}_with_missing.csv'")

        # Impute missing values using chosen imputers
        for imputer_name, use_imputer in imputers.items():
            if not use_imputer:
                continue
            if imputer_name != "cm_imputer" and categorical:
                continue

            if imputer_name == "cm_imputer":
                # CM Imputer
                imputer = cm_imputer
                name = "CM Imputer"
            elif imputer_name == "knn_imputer":
                # KNN Imputer
                imputer = KNNImputer()
                name = "KNN Imputer"
            elif imputer_name == "simple_imputer":
                # Simple Imputer (Mean)
                imputer = SimpleImputer()
                name = "Simple Imputer"

            start_time = time.time()
            imputer.fit(data_missing)
            train_time = time.time()
            data_imputed = imputer.transform(data_missing)
            impute_time = time.time()
            if imputer_name != "cm_imputer":
                data_imputed = pd.DataFrame(data_imputed)
            #end_time = time.time()

            # Save the imputed dataset (only for CM Imputer)
            if imputer_name == "cm_imputer":
                data_imputed.to_csv(path + "imputed_cm.csv", index=False)
                if print_results:
                    print(f"Imputed dataset saved as '{dataset_name}_imputed_cm.csv'")

            # Print evaluation results
            if print_results:
                print("___________________________________________________")
                print(f"{name} evaluation on {dataset_name} dataset:")
                print(f"Time taken for training: {train_time - start_time:.2f} seconds")
                print(f"Time taken for imputation: {impute_time - train_time:.2f} seconds")
            if not missing:
                if not categorical:     # For non-categorical datasets
                    # Select only the originally missing values for comparison
                    imputed_values = data_imputed.values[mask]
                    true_values = data.values[mask]
                    # Compute Error Metrics
                    mae = mean_absolute_error(true_values, imputed_values)
                    rmse = np.sqrt(mean_squared_error(true_values, imputed_values))
                    correlation = np.corrcoef(true_values.flatten(), imputed_values.flatten())[0, 1]
                    if print_results:
                        print(f"Mean Absolute Error (MAE): {mae:.4f}")
                        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
                        print(f"Correlation between true and imputed values: {correlation:.4f}")
                        print("___________________________________________________")

                    scores[dataset_name, imputer_name] = {
                        "MAE": mae,
                        "RMSE": rmse,
                        "Correlation": correlation,
                        "Training Time": train_time - start_time,
                        "Imputation Time": impute_time - train_time,
                    }
                else:       # For categorical datasets
                    data = data.to_numpy()
                    data = data.astype(str)
                    try:
                        if np.any(np.isnan(data)):
                            mask = np.logical_and(mask, ~np.isnan(data))
                    except TypeError:
                        if np.any(data == "nan"):
                            mask = np.logical_and(mask, data != "nan")
                    # Select only the originally missing values for comparison
                    imputed_values = data_imputed.values[mask]
                    true_values = data[mask]
                    # Compute Accuracy
                    accuracy = (imputed_values == true_values).mean()
                    if print_results:
                        print(f"Accuracy of imputed values: {accuracy:.4f}")
                        print("___________________________________________________")
                    
                    scores[dataset_name, imputer_name] = {
                        "Accuracy": accuracy,
                        "Training Time": train_time - start_time,
                        "Imputation Time": impute_time - train_time,
                    }
    return scores


if __name__ == "__main__":

    if settings == 0:
        # "High fidelity"
        cm_imputer = CMImputer(
            settings="precise",
            random_state=random_state,
            verbose=1,
        )

    elif settings == 2:
        # "Fast"
        cm_imputer = CMImputer(
            setting="fast",
            random_state=random_state,
            verbose=1,
        )

    elif settings == 3:
        # "Image"
        cm_imputer = CMImputer(
            settings="image",
            image_dimension=(11, 1),
            random_state=random_state,
            verbose=1,
        )

    elif settings == 4:
        # "Custom"
        cm_imputer = CMImputer(
            settings="custom",
            skip_layers=False,
            random_state=random_state,
            verbose=0,
        )

    else:
        # "Balanced"
        cm_imputer = CMImputer(
            settings="balanced",
            random_state=random_state,
            verbose=1,
        )

    run_evaluation(cm_imputer=cm_imputer, print_results=True)
        
