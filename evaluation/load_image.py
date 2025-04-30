from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from matplotlib import colors
import math
import numpy as np
import pandas as pd
from cm_tpm import CMImputer

# TODO: Add option for multiple samples
random_state = 42
remove = "Bottom"   # "top" or "bottom" or "random"
n_outputs = 5

# Function to introduce random missingness in the dataset
def introduce_missingness(data, missing_rate=0.1, random_state=42):
    rng = np.random.RandomState(random_state)  # Ensures reproducibility
    mask = rng.rand(*data.shape) < missing_rate  # Create mask for missing values
    data_missing = data.mask(mask)  # Apply mask
    return data_missing, mask

# Funtion to remove the bottom part of the image
def remove_bottom(data):
    data = data.copy()
    num_cols = data.shape[0]
    halfway = math.floor(num_cols / 2)
    data.iloc[halfway:] = np.nan  # Set the right half to NaN
    return data

# Funtion to remove the top part of the image
def remove_top(data):
    data = data.copy()
    num_cols = data.shape[0]
    halfway = math.floor(num_cols / 2)
    data.iloc[:halfway] = np.nan  # Set the left half to NaN
    return data


def show_image(image_data, ax, title):
    # Convert the 1D array into an 8x8 2D array
    image_array = np.array(image_data).reshape((8, 8))

    # Create a masked array: mask NaNs
    masked_array = np.ma.masked_invalid(image_array)

    # Create a custom colormap: grayscale for values, red for NaNs
    cmap = plt.cm.gray_r
    cmap = cmap.copy()
    cmap.set_bad(color='red')  # color for NaNs

    # Plot
    ax.imshow(masked_array, cmap=cmap, vmin=0, vmax=15)
    ax.axis('off')
    ax.set_title(title)

data = pd.DataFrame(load_digits(as_frame=True).frame)
data = data.drop("target", axis=1)

train_data = data[:1500]
test_data = data[1500:]

if remove == "random":
    # Remove random parts of the image
    test_data_missing, _ = introduce_missingness(test_data, missing_rate=0.5, random_state=random_state)
elif remove == "top":
    # Remove the top part of the image
    test_data_missing = test_data.apply(remove_top, axis=1)    
else: # Default to bottom
    # Remove the bottom part of the image
    test_data_missing = test_data.apply(remove_bottom, axis=1)


test_data = test_data
test_data_missing = test_data_missing

hidden_layers = 5
neurons_per_layer = 512
activation = "LeakyReLU"
batch_norm = True
dropout_rate = 0.3

model = CMImputer(
    n_components_train=256,
    n_components_impute=1024,
    latent_dim=4,
    imputation_method="EM",
    random_state=random_state,
    verbose=1,
    # hidden_layers=hidden_layers,
    # neurons_per_layer=neurons_per_layer,
    # activation=activation,
    # batch_norm=batch_norm,
    # dropout_rate=dropout_rate,
    # max_iter=100,
    # tol=0.0001,
    # lr=0.001,
    # weight_decay=0.01,
)

model.fit(train_data)

test_samples = test_data_missing.shape[0]

# Create the figure and axes
fig, axes = plt.subplots(n_outputs, 3, figsize=(9, 4*n_outputs + 2))

rng = np.random.default_rng(random_state)
for i in range(n_outputs):
    np.random.seed(rng.integers(1e9))
    index = np.random.randint(low=0, high=test_samples)

    test_sample = test_data_missing.iloc[index].to_numpy()
    imputed_sample = model.transform(test_sample)

    if n_outputs == 1:
        show_image(test_data.iloc[index], axes[0], "Full image")
        show_image(test_sample, axes[1], "Image with missing")
        show_image(imputed_sample, axes[2], "Imputed image")
    else:
        show_image(test_data.iloc[index], axes[i][0], "Full image")
        show_image(test_sample, axes[i][1], "Image with missing")
        show_image(imputed_sample, axes[i][2], "Imputed image")

plt.show()
