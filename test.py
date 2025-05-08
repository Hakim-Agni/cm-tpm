from cm_tpm import CMImputer
from cm_tpm._helpers import _convert_json
import numpy as np

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

imputer = CMImputer(missing_values=np.nan, n_components_train=64, random_state=0)
imputer.fit(X, save_model_path="saved_models/cm_tpm_test/")

#imputer.save_model("saved_models/cm_tpm_test")

loaded_imputer = CMImputer.load_model("saved_models/cm_tpm_test/")

imputed = loaded_imputer.transform(X)
print(imputed)

a = CMImputer.transform_from_file(X, load_model_path="saved_models/cm_tpm_test/")
print(a)
