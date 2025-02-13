from cm_tpm import CMImputer
import numpy as np

data = [
    [1, 2, 3], 
    [4, 5, 6], 
    [7, 8, 9],
    [10, 11, 12],
    [13, 14, 15],
    [16, 17, 18],
    [19, 20, 21],
    [22, 23, 24],
    [25, 26, 27],
    [28, 29, 30],
]
X = [
    [1, 2, 3], 
    [4, 5, 6], 
    [7, 8, 9],
    [np.nan, 11, 12],
    [13, 14, 15],
    [16, 17, 18],
    [19, 20, 21],
    [22, 23, np.nan],
    [25, 26, 27],
    [28, 29, 30],
]
imputer = CMImputer(missing_values=np.nan, n_components=5)
imputer.fit(data)
X_imputed = imputer.transform(X)

print(X_imputed)
