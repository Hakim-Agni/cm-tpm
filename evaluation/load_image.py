import os
from sklearn.datasets import load_digits
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from cm_tpm import CMImputer

# Configurations
dataset = "digits"      # "digits" or "fashion"
random_state = 42
remove = "bottom"          # "top", "bottom", "left", "right", or "random"
missing_rate = 0.5     # Missing rate for the images
n_outputs = 5           # The number of output images to show
train_new = False       # Whether to train a new model or use a saved model (if available)
imputer = "cm"          # "cm", "knn", or "simple"

# Function to introduce random missingness in the dataset
def introduce_missingness(data, missing_rate=0.1, random_state=42):
    assert 0 < missing_rate <= 1, "The missing rate must be between 0 and 1."
    rng = np.random.RandomState(random_state)  # Ensures reproducibility
    mask = rng.rand(*data.shape) < missing_rate  # Create mask for missing values
    data_missing = data.mask(mask)  # Apply mask
    return data_missing, mask

# Function to remove a specific part of the image (top, bottom, right or left)
def remove_side(data, image_shape, side="bottom", missing_rate=0.5):
    data = data.copy().to_numpy()
    w, h = image_shape      # Get the image shape
    num_cols = data.shape[0]
    assert num_cols == w * h, "Mismatch between sample length and shape."
    assert 0 < missing_rate <= 1, "The missing rate must be between 0 and 1."

    # Reshape as 2d array
    data_2d = data.reshape(w, h).copy()
    # Compute the amount of columns to remove
    cols_to_mask = int(w * missing_rate)

    if cols_to_mask == 0:
        return data     # Nothing to remove
    
    # Remove the selected side from the array
    if side == "bottom":
        data_2d[-cols_to_mask:, :] = np.nan
    elif side == "top":
        data_2d[:cols_to_mask, :] = np.nan
    elif side == "left":
        data_2d[:, :cols_to_mask] = np.nan
    elif side == "right":
        data_2d[:, -cols_to_mask:] = np.nan

    # Return a flattened array
    return pd.Series(data_2d.flatten())

def show_image(image_data, ax, title, image_shape):
    # Convert the 1D array into an 8x8 2D array
    image_array = np.array(image_data).reshape(image_shape)

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

if dataset == "digits":
    data = pd.DataFrame(load_digits(as_frame=True).frame)
    data = data.drop("target", axis=1)

    train_data = data[:1500]
    test_data = data[1500:]

    image_shape = (8, 8)

    save_str = "evaluation/models/digits/"
elif dataset == "fashion":
    # Load FashionMNIST (only the data, no labels)
    fashion_mnist = torchvision.datasets.FashionMNIST(
        root="/data", train=True, download=True, transform=torchvision.transforms.ToTensor()
    )

    # Limit the size if needed (e.g., for speed)
    images = [img for img, label in list(fashion_mnist)[:5000]]  # First 5000 samples
    data_np = np.stack([img.squeeze().numpy().flatten() * 15 for img in images])  # Scale to 0-255

    data = pd.DataFrame(data_np)

    train_data = data[:4000]
    test_data = data[4000:]

    image_shape = (28, 28)

    save_str = "evaluation/models/fashion/"
else:
    raise ValueError(f"Unsupported dataset: {dataset}")

if remove == "random":
    # Remove random parts of the image
    test_data_missing, _ = introduce_missingness(test_data, missing_rate=missing_rate, random_state=random_state)
else:
    assert remove in ["top", "bottom", "left", "right"], "Side to remove must be 'top', 'bottom', 'left', 'right', or 'random'."
    # Remove the selected side of the image
    test_data_missing = test_data.apply(lambda x: remove_side(x, image_shape, side=remove, missing_rate=missing_rate), axis=1)

# test_data = test_data
# test_data_missing = test_data_missing

if imputer == "cm":
    model = CMImputer(
        settings="balanced",
        random_state=random_state,
        verbose=1,
    )
elif imputer == "knn":
    model = KNNImputer()
elif imputer == "simple":
    model = SimpleImputer()
else:
    raise ValueError(f"Unsupported imputer type: {imputer}")

if imputer == "cm" and not train_new and os.path.exists(save_str):
    model = CMImputer.load_model(save_str)
elif imputer == "cm":
    model.fit(train_data, save_model_path=save_str)
else:
    model.fit(train_data)

test_samples = test_data_missing.shape[0]

# Create the figure and axes
fig, axes = plt.subplots(n_outputs, 3, figsize=(9, 4*n_outputs + 2))

rng = np.random.default_rng(random_state)
for i in range(n_outputs):
    np.random.seed(rng.integers(1e9))
    index = np.random.randint(low=0, high=test_samples)

    test_sample = test_data_missing.iloc[index].to_numpy()
    imputed_sample = model.transform([test_sample])

    if n_outputs == 1:
        show_image(test_data.iloc[index], axes[0], "Full image", image_shape)
        show_image(test_sample, axes[1], "Image with missing", image_shape)
        show_image(imputed_sample, axes[2], "Imputed image", image_shape)
    else:
        show_image(test_data.iloc[index], axes[i][0], "Full image", image_shape)
        show_image(test_sample, axes[i][1], "Image with missing", image_shape)
        show_image(imputed_sample, axes[i][2], "Imputed image", image_shape)

plt.show()
