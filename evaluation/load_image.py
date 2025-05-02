from sklearn.datasets import load_digits
import torchvision
import matplotlib.pyplot as plt
from matplotlib import colors
import math
import numpy as np
import pandas as pd
from cm_tpm import CMImputer

# TODO: Add option for multiple samples
dataset = "fashion"
random_state = 42
remove = "bottom"   # "top" or "bottom" or "random"
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
elif dataset == "fashion":
    # Load FashionMNIST (only the data, no labels)
    fashion_mnist = torchvision.datasets.FashionMNIST(
        root="/data", train=True, download=True, transform=torchvision.transforms.ToTensor()
    )

    # Limit the size if needed (e.g., for speed)
    images = [img for img, label in list(fashion_mnist)[:5000]]  # First 2000 samples
    data_np = np.stack([img.squeeze().numpy().flatten() * 255 for img in images])  # Scale to 0-255

    data = pd.DataFrame(data_np)

    train_data = data[:4000]
    test_data = data[4000:]

    image_shape = (28, 28)
else:
    raise ValueError(f"Unsupported dataset: {dataset}")

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

model = CMImputer(
    settings="balanced",
    random_state=0,
    verbose=1,
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
        show_image(test_data.iloc[index], axes[0], "Full image", image_shape)
        show_image(test_sample, axes[1], "Image with missing", image_shape)
        show_image(imputed_sample, axes[2], "Imputed image", image_shape)
    else:
        show_image(test_data.iloc[index], axes[i][0], "Full image", image_shape)
        show_image(test_sample, axes[i][1], "Image with missing", image_shape)
        show_image(imputed_sample, axes[i][2], "Imputed image", image_shape)

plt.show()
