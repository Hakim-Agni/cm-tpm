User Guide
------------

### Quickstart
After installation, you can easily start imputing missing data, for example:
```python
from cm-tpm import CMImputer
import numpy as np

# Example numpy dataset with missing values (NaNs)
data = np.array([[1, 2, np.nan], [4, np.nan, 6], [7, 8, 9]])

# Initialize the imputer with default settings
imputer = CMImputer()

# Fit the model and impute missing values
imputed_data = imputer.fit_transform(data)

print(imputed_data)
```

<!-- Preprocessing -->

<!-- Seperate fit and transform -->

<!-- Supported input data types and resulting output -->

<!-- Hyperparameters? (Missing data value(s)), link to API -->
