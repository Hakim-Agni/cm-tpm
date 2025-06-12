import os
import numpy as np
import pandas as pd
import time
from torch import nn
import torchvision
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes, load_breast_cancer, load_digits, load_iris, load_linnerud, load_wine
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt
from cm_tpm import CMImputer

# Set a custom cm_imputer for hyperparameter tuning
cm_imputer = CMImputer(
            settings="custom",
            # Add other hyperparameters as needed
            verbose=1,
        )

# General Settings
nr_of_runs = 1
random_state = 42
missing_rate = 0.1
show_plots = False        # Whether to show plots of log-likelihoods during training or not
save_datasets = True     # Whether to save datasets with missing values and imputed values or not
split_data = True        # Whether to split the data into training and test sets or not

# Dataset Settings
diabetes = True                 # Medium sized; numerical and integer
breast_cancer = False            # Large sized; numerical and binary
digits = False                   # Very large sized; integer; image
fashion = False                  # Very large sized; numerical, image
iris = False                     # Small sized; numerical and binary
linnerud = False                 # Small sized; integer
mushroom = False                 # Very large sized; categorical and binary
wine = False                     # Medium sized; numerical and binary

# Imputer Settings
use_cm_imputer = True
use_knn_imputer = True
use_simple_imputer = True

datasets = {
    "diabetes": diabetes,
    "breast_cancer": breast_cancer,
    "digits": digits,
    "fashion": fashion,
    "iris": iris,
    "linnerud": linnerud,
    "mushroom": mushroom,
    "wine": wine,
    }

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

def run_evaluation(cm_imputer=CMImputer(), print_results=True, rand_state=None):
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
            data = data.drop(['target'], axis=1)
            os.makedirs("evaluation/data/diabetes", exist_ok=True)  # Create directory if it doesn't exist
            path = "evaluation/data/diabetes/diabetes_"
        elif dataset_name == "breast_cancer":
            data = load_breast_cancer(as_frame=True).frame
            data = data.drop(['target'], axis=1)
            os.makedirs("evaluation/data/breast_cancer", exist_ok=True)  # Create directory if it doesn't exist
            path = "evaluation/data/breast_cancer/breast_cancer_"
        elif dataset_name == "digits":
            data = load_digits(as_frame=True).frame
            data = data.drop(['target'], axis=1)
            os.makedirs("evaluation/data/digits", exist_ok=True)
            path = "evaluation/data/digits/digits_"
        elif dataset_name == "fashion":
            fashion_mnist = torchvision.datasets.FashionMNIST(
                root="/data", train=True, download=True, transform=torchvision.transforms.ToTensor()
            )
            images = [img for img, label in list(fashion_mnist)]
            data_np = np.stack([img.squeeze().numpy().flatten() for img in images])
            data = pd.DataFrame(data_np)
            data = data[:5000]  # Reduce amount of samples
            os.makedirs("evaluation/data/fashion", exist_ok=True)
            path = "evaluation/data/fashion/fashion_"
        elif dataset_name == "iris":
            data = load_iris(as_frame=True).frame
            data = data.drop(['target'], axis=1)
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
            os.makedirs("evaluation/data/mushroom", exist_ok=True)
            path = "evaluation/data/mushroom/mushroom_"
        elif dataset_name == "wine":
            data = load_wine(as_frame=True).frame
            data = data.drop(['target'], axis=1)
            os.makedirs("evaluation/data/wine", exist_ok=True)
            path = "evaluation/data/wine/wine_"

        # Save the complete dataset to CSV
        if save_datasets:
            data.to_csv(path + "complete.csv", index=False)

        if split_data:
            # Split the data into training and test sets
            data_train, data_test = train_test_split(data, test_size=0.2, shuffle=True, random_state=rand_state)

            if save_datasets:
                # Save the training and test sets to CSV
                data_train.to_csv(path + "train.csv", index=False)
                data_test.to_csv(path + "test.csv", index=False)
                if print_results:
                    print(f"Training set saved as '{dataset_name}_train.csv'")
                    print(f"Test set saved as '{dataset_name}_test.csv'")
        else:
            # Use the complete dataset for training and testing
            data_test = data.copy()
            
        # Introduce missing values
        if not missing:
            data_missing, mask = introduce_missingness(data_test, missing_rate=missing_rate, random_state=rand_state)

        # Use the missing dataset for training and testing
        if not split_data:
            data_train = data_missing.copy()

        # Save to CSV for testing
        if save_datasets:
            data_missing.to_csv(path + "with_missing.csv", index=False)
            if print_results:
                print(f"Dataset with missing values saved as '{dataset_name}_with_missing.csv'")

        # Impute missing values using chosen imputers
        for imputer_name, use_imputer in imputers.items():
            if not use_imputer:
                continue
            
            train_data = data_train.copy()
            test_data = data_missing.copy()

            if imputer_name != "cm_imputer" and categorical:
                # Apply encoding for categorical datasets
                encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
                encoder.fit(data)
                train_data = encoder.transform(data_train)
                test_data = encoder.transform(data_missing)

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
            imputer.fit(train_data)
            train_time = time.time()
            data_imputed = imputer.transform(test_data)
            impute_time = time.time()
            if imputer_name != "cm_imputer":
                if categorical:
                    data_imputed = encoder.inverse_transform(data_imputed)
                data_imputed = pd.DataFrame(data_imputed)

            # Store the likelihoods during trainging and imputation (only for CM Imputer)
            if imputer_name == "cm_imputer":
                likelihoods[dataset_name] = {
                    "Training": imputer.training_likelihoods_,
                    "Inference": imputer.imputing_likelihoods_,
                }

            # Save the imputed dataset (only for CM Imputer)
            if imputer_name == "cm_imputer" and save_datasets:
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
                    true_values = data_test.values[mask]
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
                    if not isinstance(data_test, np.ndarray):
                        data_test = data_test.to_numpy()
                    data_test = data_test.astype(str)
                    try:
                        if np.any(np.isnan(data_test)):
                            mask = np.logical_and(mask, ~np.isnan(data_test))
                    except TypeError:
                        if np.any(data_test == "nan"):
                            mask = np.logical_and(mask, data_test != "nan")
                    # Select only the originally missing values for comparison
                    imputed_values = data_imputed.values[mask]
                    true_values = data_test[mask]
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
    rng = np.random.default_rng(random_state) 

    maes = {}
    rmses = {}
    corrs = {}
    accs = {}
    tt = {}
    ti = {}
    ll_train = {}
    ll_inference = {}

    # Run a number of times
    for i in range(nr_of_runs):
        rand_state = rng.integers(1e9)
        cm_imputer.set_params(random_state=rand_state)

        score, likelihood = run_evaluation(cm_imputer=cm_imputer, print_results=False, rand_state=rand_state)

        # Store the likelihoods for the first loop only
        for dataset_name in likelihood.keys():
            ll_train[dataset_name, i] = likelihood[dataset_name].get("Training", [])
            ll_inference[dataset_name, i] = likelihood[dataset_name].get("Inference", [])
        
        for dataset_name, imputer_name in score.keys():
            if (dataset_name, imputer_name) not in maes:
                maes[dataset_name, imputer_name] = []
                rmses[dataset_name, imputer_name] = []
                corrs[dataset_name, imputer_name] = []
                accs[dataset_name, imputer_name] = []
                tt[dataset_name, imputer_name] = []
                ti[dataset_name, imputer_name] = []

            mae = score[dataset_name, imputer_name].get("MAE", -1)
            rmse = score[dataset_name, imputer_name].get("RMSE", -1)
            correlation = score[dataset_name, imputer_name].get("Correlation", -1)
            accuracy = score[dataset_name, imputer_name].get("Accuracy", -1)
            time_train = score[dataset_name, imputer_name].get("Training Time", -1)
            time_impute = score[dataset_name, imputer_name].get("Imputation Time", -1)

            maes[dataset_name, imputer_name].append(mae)
            rmses[dataset_name, imputer_name].append(rmse)
            corrs[dataset_name, imputer_name].append(correlation)
            accs[dataset_name, imputer_name].append(accuracy)
            tt[dataset_name, imputer_name].append(time_train)
            ti[dataset_name, imputer_name].append(time_impute)

    # Print results
    for dataset_name in datasets.keys():
        for imputer_name in imputers.keys():
            if (dataset_name, imputer_name) not in maes:
                continue
        
            # Compute mean and confidence intervals for MAE
            mae_mean = np.mean(maes[dataset_name, imputer_name])
            mae_std = np.std(maes[dataset_name, imputer_name])
            mae_int_coef = 1.96 * mae_std / np.sqrt(nr_of_runs)
            mae_int_top = mae_mean + mae_int_coef
            mae_int_bottom = mae_mean - mae_int_coef

            # Compute mean and confidence intervals for RMSE
            rmse_mean = np.mean(rmses[dataset_name, imputer_name])
            rmse_std = np.std(rmses[dataset_name, imputer_name])
            rmse_int_coef = 1.96 * rmse_std / np.sqrt(nr_of_runs)
            rmse_int_top = rmse_mean + rmse_int_coef
            rmse_int_bottom = rmse_mean - rmse_int_coef

            # Compute mean and confidence intervals for Correlation
            corr_mean = np.mean(corrs[dataset_name, imputer_name])
            corr_std = np.std(corrs[dataset_name, imputer_name])
            corr_int_coef = 1.96 * corr_std / np.sqrt(nr_of_runs)
            corr_int_top = corr_mean + corr_int_coef
            corr_int_bottom = corr_mean - corr_int_coef

            # Compute mean and confidence intervals for Accuracy
            acc_mean = np.mean(accs[dataset_name, imputer_name])
            acc_std = np.std(accs[dataset_name, imputer_name])
            acc_int_coef = 1.96 * acc_std / np.sqrt(nr_of_runs)
            acc_int_top = acc_mean + acc_int_coef
            acc_int_bottom = acc_mean - acc_int_coef

            print("___________________________________________________")
            print(f"Dataset: {dataset_name}, Imputer: {imputer_name}")
            if mae_mean != -1:
                print(f"\tMean Absolute Error (MAE): {mae_mean:.4f} (± {mae_int_top-mae_mean:.4f}). 95% Confidence Interval: [{mae_int_bottom:.4f}, {mae_int_top:.4f}]")
                print(f"\tRoot Mean Squared Error (RMSE): {rmse_mean:.4f} (± {rmse_int_top-rmse_mean:.4f}). 95% Confidence Interval: [{rmse_int_bottom:.4f}, {rmse_int_top:.4f}]")
                print(f"\tCorrelation: {corr_mean:.4f} (± {corr_int_top-corr_mean:.4f}). 95% Confidence Interval: [{corr_int_bottom:.4f}, {corr_int_top:.4f}]")
            if acc_mean != -1:
                print(f"\tAccuracy: {acc_mean:.4f} (± {acc_int_top-acc_mean:.4f}). 95% Confidence Interval: [{acc_int_bottom:.4f}, {acc_int_top:.4f}]")
            #print(f"\tAccuracy: {np.mean(accs[dataset_name]):.4f}")
            print(f"\tTraining Time: {np.mean(tt[dataset_name, imputer_name]):.4f} seconds")
            print(f"\tImputation Time: {np.mean(ti[dataset_name, imputer_name]):.4f} seconds")
    print("___________________________________________________")

    # Plot likelihood results
    if show_plots:
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
