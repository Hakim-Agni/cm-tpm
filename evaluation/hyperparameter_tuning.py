import os
import numpy as np
import pandas as pd
import time
from torch import nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.datasets import load_diabetes, load_breast_cancer, load_digits, load_iris, load_linnerud, load_wine
from sklearn.impute import KNNImputer, SimpleImputer
from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt
from cm_tpm import CMImputer


# Dataset Settings
# Complete datasets
diabetes = True                 # Medium sized; numerical and integer
breast_cancer = False            # Large sized; numerical and binary
digits = False                   # Very large sized; integer; image
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
    likelihoods = {}
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

            # Store the likelihoods during trainging and imputation (only for CM Imputer)
            if imputer_name == "cm_imputer":
                likelihoods[dataset_name] = {
                    "Training": imputer.training_likelihoods_,
                    "Inference": imputer.imputing_likelihoods_,
                }

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
    return scores, likelihoods


if __name__ == "__main__":
    # CMImputer Settings
    hidden_layers = 5
    neurons_per_layer = 1024
    activation = "LeakyReLU"
    batch_norm = True
    dropout_rate = 0.1

    random_state = None
    rng = np.random.default_rng(random_state) 
    nr_of_runs = 5

    maes = {}
    accs = {}
    tt = {}
    ti = {}
    ll_train = {}
    ll_inference = {}

    # Run once with each random state
    for i in range(nr_of_runs):
        rand_state = rng.integers(1e9)
        cm_imputer = CMImputer(
            settings="Fast",
            skip_layers=False,
            random_state=rand_state,
            verbose=0,
        )

        score, likelihood = run_evaluation(cm_imputer=cm_imputer, print_results=False)

        # Store the likelihoods for the first loop only
        for dataset_name in likelihood.keys():
            ll_train[dataset_name, i] = likelihood[dataset_name].get("Training", [])
            ll_inference[dataset_name, i] = likelihood[dataset_name].get("Inference", [])
        
        for dataset_name, imputer_name in score.keys():
            if imputer_name != "cm_imputer":
                continue

            if dataset_name not in maes:
                maes[dataset_name] = []
                accs[dataset_name] = []
                tt[dataset_name] = []
                ti[dataset_name] = []

            mae = score[dataset_name, imputer_name].get("MAE", 0)
            accuracy = score[dataset_name, imputer_name].get("Accuracy", 0)
            time_train = score[dataset_name, imputer_name].get("Training Time", 0)
            time_impute = score[dataset_name, imputer_name].get("Imputation Time", 0)

            maes[dataset_name].append(mae)
            accs[dataset_name].append(accuracy)
            tt[dataset_name].append(time_train)
            ti[dataset_name].append(time_impute)

    # Print results
    for dataset_name in datasets.keys():
        if dataset_name not in maes:
            continue
        print(f"Dataset: {dataset_name}")
        print(f"\tMean Absolute Error (MAE): {np.mean(maes[dataset_name]):.4f}")
        print(maes[dataset_name])
        #print(f"\tAccuracy: {np.mean(accs[dataset_name]):.4f}")
        print(f"\tTraining Time: {np.mean(tt[dataset_name]):.4f} seconds")
        print(f"\tImputation Time: {np.mean(ti[dataset_name]):.4f} seconds")

    # Plot likelihood results
    n_datasets = sum(datasets.values())
    if n_datasets > 1:
        fig, axes = plt.subplots(n_datasets, 2, sharex=True, sharey=False, figsize=(9, 4 * n_datasets))
        mult_data = True
        n_rows = n_datasets
    else:
        fig, axes = plt.subplots(nr_of_runs, 2, sharex=True, sharey=False, figsize=(9, 4 * nr_of_runs))
        mult_data = False
        n_rows = nr_of_runs

    index = 0
    for name, i in ll_train.keys():
        if mult_data and i > 0:
            print(mult_data)
            continue

        if n_rows == 1:
            axes[0].plot(ll_train[name, i])
            axes[0].set_title(name + " - training")
            axes[1].plot(ll_inference[name, i])
            axes[1].set_title(name + " - inference")
        else:
            axes[index, 0].plot(ll_train[name, i])
            axes[index, 0].set_title(name + " - training - run " + str(i))
            axes[index, 1].plot(ll_inference[name, i])
            axes[index, 1].set_title(name + " - inference - run " + str(i))

            index += 1

    fig.supxlabel('Epoch')
    fig.supylabel('Log-likelihood')

    plt.show()
