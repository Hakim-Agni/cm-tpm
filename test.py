from cm_tpm import CMImputer
from cm_tpm._helpers import _convert_json
import numpy as np

X = [
    [1, 2, 3, 4], 
    [4, 5, 6, 7], 
    [7, 8, 9, 10],
    [np.nan, 11, 12, 13],
    [13, 14, 15, 16],
    [16, 17, 18, 19],
    [19, 20, 21, 22],
    [22, 23, np.nan, 25],
    [25, 26, 27, 28],
    [28, 29, 30, 31],
]

imputer = CMImputer(settings="image")

imputer.fit(X, image_dimension=(3, 2))

imputed = imputer.transform(X)

print(imputed)
