# User Guide
### Welcome to the User Guide of **cm-tpm**!  

In this guide, you can find the following:
- [Installation Instructions](https://github.com/Hakim-Agni/cm-tpm/wiki/User-Guide#installation) for the package.
- [A Quick Guide](https://github.com/Hakim-Agni/cm-tpm/wiki/User-Guide#quick-guide) to using the package.
- [Some Examples](https://github.com/Hakim-Agni/cm-tpm/wiki/User-Guide#examples) for usage.
- [Frequently Asked Questions](https://github.com/Hakim-Agni/cm-tpm/wiki/User-Guide#frequently_asked_questions) and answers.

---
## Installation

### Installing the Package
You can install the package via pip or clone it directly from GitHub.

#### Option 1: Using pip (recommended)
```bash
pip install cm-tpm
```
This command installs the latest version of **cm-tpm**, including all dependencies.  
If you also want to install all optional dependencies, use:
```bash
pip install cm-tpm[all]
```

#### Option 2: Install from source
```bash
git clone https://github.com/Hakim-Agni/cm-tpm.git
cd cm-tpm
pip install .
```
Once installed locally, you can install all requirements using:
```bash
pip install -r 'requirements.txt'
```

### Installing a GPU
To install PyTorch with GPU (CUDA) support, follow the instructions at: https://pytorch.org/get-started/locally/.  
For example, install CUDA 12.6 using:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

To verify whether CUDA has been installed correctly, run the following in Python:
```python
import torch

print(torch.cuda.is_available())
```
If installed correctly, the output will show `True`.

---
## Quick Start

The most basic way to use **cm-tpm** looks like this:
```python
import numpy as np
from cm_tpm import CMImputer

# Example data with missing values
X = np.array([[1.0, np.nan], [2.0, 3.0], [np.nan, 4.0]])

# Initialize and fit imputer
imputer = CMImputer()
X_imputed = imputer.fit_transform(X)

# Print the imputed data
print(X_imputed)
```
This example shows how to initialze a *CMImputer*, train it on a dataset, and impute missing values.  

It is also possible to use different data for training and imputation. For example:
```python
import numpy as np
from cm_tpm import CMImputer

# Example data with and without missing values
X_train = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
X_missing = np.array([[1.0, np.nan], [2.0, 3.0], [np.nan, 4.0]])

# Initialize and fit imputer
imputer = CMImputer()
imputer.fit(X_train)

# Impute missing values
X_imputed = imputer.transform(X_missing)

# Print the imputed data
print(X_imputed)
```
Here, *X_train* is used for training a model, and the trained model is used to impute missing values in *X_missing*.

---
## Examples

### Impute a real-world dataset
```python
import numpy as np
from cm_tpm import CMImputer
from sklearn.datasets import load_diabetes

# Load the real-world data
X, y = load_diabetes(return_X_y=True)

# Function to introduce missingness in the dataset
def introduce_missingness(data, missing_rate=0.1, random_state=42):
    rng = np.random.RandomState(random_state)  # Ensures reproducibility
    mask = rng.rand(*data.shape) < missing_rate  # Create mask for missing values
    data_missing = data.copy() 
    data_missing[mask] = np.nan  # Apply mask
    return data_missing, mask

# Add missing values
X_missing = introduce_missingness(X)

# Initialize and fit imputer
imputer = CMImputer()
X_imputed = imputer.fit_transform(X_missing)
```

### Using custom hyperparameters
```python
from cm_tpm import CMImputer

imputer = CMImputer(n_components_train=32, n_components_impute=128, top_k=5)
```
These include just a few hyperparameters, the full list is found in the [API Documentation](API-Documentation).

---

## Frequently Asked Questions

### Q: Does **cm-tpm** support categorical data?
A: Yes, **cm-tpm** supports categorical data by automatically encoding categorical features in the data. Naturally, the encoded features are reverted back to their original after imputation.

### Q: Do I need the pre-process my data before using **cm-tpm**?
A: No, all relevant pre-processing steps are automatically performed. These include encoding categorical data and scaling numerical data.

### Q: Is **cm-tpm** compatible with scikit-learn pipelines?
A: Yes, **cm-tpm** follows sklearn's API with `fit`, `transform`, and `fit_transform` methods.

### Q: Can I use **cm-tpm** for image data?
A: Yes, **cm-tpm** works on all 2-dimensional datasets, this also includes image data.

### Q: Does **cm-tpm** support ordinal data?
A: Partially. **cm-tpm** can support ordinal data if the user explicitly specifies which features are ordinal and provides their ordering. This can be done via the `ordinal_features` parameter in *CMImputer*. See the [API Documentation](API-Documentation) for details.

---
For other questions, please reach out via [GitHub Discussions](https://github.com/Hakim-Agni/cm-tpm/discussions).