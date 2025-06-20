# API Documentation
### Welcome to the API documentation of **cm-tpm**!

This document contains the following:
- [An API Overview](#api-overview) of the package.
- [The CMImputer Class](#cmimputer), including a full list of parameters.
- [All Methods/Functions](#methods) that can be called.
- [Several Examples](#examples) showcasing various features of the package.

---
## API Overview

The main entry point is the `CMImputer` class. This class is used for all functionalities of the package.


### Features

The `CMImputer` class offers the following core features:
- Fitting a CM-TPM
- Performing imputation on missing data using a trained CM-TPM
- Scikit-learn compatible
- Various configurable hyperparameters

The `CMImpter` class also included several additional features:
- Automatic pre-processing, including support for categorical and ordinal data
- Saving and loading CM-TPMs
- Support for common datatypes (ndarrays, pandas dataframes, Python lists) and common file types (csv, xlsx, parquet, feather)
- Built-in evaluation functions
- Debugging options


---
## CMImputer

```python
class cm_tpm.CMImputer(*, settings='custom', missing_values=nan, n_components_train=256, n_components_impute=2048, latent_dim=4, top_k=None, lo=False, 
 pc_type='factorized', imputation_method='expectation', ordinal_features=None, max_depth=5, custom_net=None, hidden_layers=4, neurons_per_layer=512, 
 activation='LeakyReLU', batch_norm=True, dropout_rate=0.1, skip_layers=True, max_iter=100, batch_size_train=1024, batch_size_impute=256, tol=1e-4, 
 patience=10, lr=0.001, weight_decay=0.01, use_gpu=True, random_state=None, verbose=0, copy=True, keep_empty_features=False)
``` 
<p align="right"><a href="https://github.com/Hakim-Agni/cm-tpm/blob/master/src/cm_tpm/_cm.py#L15">[source code]</a></p>

Imputation for completing missing values using Continuous Mixtures of Tractable Probabilistic Models (CM-TPM), a framework created by [Correia et al. (2023)](https://arxiv.org/abs/2209.10584).

A CM-TPM is trained using input data, which involves learning the parameters of a neural network that outputs TPM structures and parameters. The trained CM-TPM is used to impute missing values by either optimizing the missing values or by using the properties of the TPM components. 

For more details, please refer to the original article or the thesis (link coming soon) corresponding to this package.


### Parameters:

- **settings** : *{'custom', 'fast', 'balanced', 'precise'}, default='custom'*  
&emsp; The hyperparameters settings to use for the model.  
&emsp; - 'custom' : Allows custom hyperparameters by setting them manually.  
&emsp; - 'fast' : Prioritizes speed over accuracy. Use when speed is most important.  
&emsp; - 'balanced' : Balances accuracy and speed. The default parameter settings. Suitable for general use.  
&emsp; - 'precise' : Focuses on accuracy, with less regard to speed. Use when accuracy matters the most.  
- **missing_values** : *int, float, str, np.nan or list, default=np.nan*  
&emsp; The placeholder(s) for missing values in the input data. All instances of **missing_values** will be imputed.  
- **n_components_train** : *int, default=256*  
&emsp; Number of components to use in the mixture model during training. Values that are a power of 2 are preferred.  
- **n_components_impute** : *int or None, default=2048*  
&emsp; Number of components to use in the mixture model during imputation. Values that are a power of 2 are preferred.  
&emsp; If None, it uses the same number of components as used during training.  
- **latent_dim** : *int, default=4*  
&emsp; Dimensionality of the latent variable.  
- **top_k** : *int or None, default=None*  
&emsp; The number of components to use for efficient learning. If None, all components are used.  
- **lo** : *bool, default=False*  
&emsp; Whether to use latent optimization after training.  
- **pc_type** : *{'factorized'}, default='factorized'*  
&emsp; The type of PC to use in the model. Currently, only 'factorized' is supported.  
- **imputation_method** : *{'expectation', 'optimization'}, default='expectation'*  
&emsp; The imputation method to use during inferece.  
&emsp; - 'expectation' : Imputes values by maximizing the expected values. Faster method.  
&emsp; - 'optimization' : Imputes values by finding the optimal values using an optimizer. More accurate method.  
- **ordinal_features** : *dict or None, default=None*  
&emsp; A dictionary containing information on which features have ordinal data and how the values are ordered.  
- **max_depth** : *int, default=5*  
&emsp; Maximum depth of the PC, if applicable. Currently not used.  
- **custom_net** : *nn.Sequential or None, default=None*  
&emsp; A custom neural network to use in the model.  
- **hidden_layers** : *int, default=4*  
&emsp; The number of hidden layers to use in the neural network. Only used if **custom_net**=None.  
- **neurons_per_layer** : *int or list of ints, default=512*  
&emsp; The number of neurons in each layer in the neural network. Only used if **custom_net**=None.  
- **activation** : *{'ReLU', 'LeakyReLU', 'Tanh', 'Sigmoid', 'Identity'}, default='LeakyReLU'*  
&emsp; The activation function to use in the neural network. Only used if **custom_net**=None.  
- **batch_norm** : *bool, default=True*  
&emsp; Whether to use batch normalization in the neural network. Only used if **custom_net**=None. 
- **dropout_rate** : *float, default=0.1*  
&emsp; The dropout rate to use in the neural network. Only used if **custom_net**=None.   
- **skip_layers** : *bool, default=True*  
&emsp; Whether to use skip layers in the neural network.  
- **max_iter** : *int, default=100*  
&emsp; Maximum number of iteration to perform during training.  
- **batch_size_train** : *int or None, default=1024*  
&emsp; The batch size to use for training. If None, the entire dataset is used.  
- **batch_size_impute** : *int or None, default=256*  
&emsp; The batch size to use for imputing. If None, the entire dataset is used.  
- **tol** : *float, default=1e-4*  
&emsp; Tolerance for the convergence criterion.  
- **patience** : *int, default=10*  
&emsp; Number of iterations to wait if no improvement and then stop the training.  
- **lr** : *float, default=0.001*  
&emsp; The learning rate for the optimizer.  
- **weight_decay** : *flaot, default=0.01*  
&emsp; The weight decay (L2 penalty) for the optimizer.  
- **use_gpu** : *bool, default=True*  
&emsp; Whether to use GPU for training and imputation if available. If False, CPU is used.  
- **random_state** : *int, RandomState instance or None, default=None*  
&emsp; Random seed for reproducibility.  
- **verbose** : *int, default=0*  
&emsp; Verbosity level, controls debug messages.  
- **copy** : *bool, default=True*  
&emsp; Whether to copy the input data or modify it in place.  
- **keep_empty_features** : *bool, default=False*  
&emsp; Whether to keep features that only have missing values in the imputed dataset.


### Attributes:

- **is_fitted_**: *bool*  
&emsp; Whether the model is fitted.  
- **n_features_in_**: *int*  
&emsp; The number of features in the input data.  
- **feature_names_in_**: *list of str*  
&emsp; Names of the input features.  
- **input_dimension**: *int*  
&emsp; The number of features in the input data after pre-processing.  
- **log_likelihood_**: *float*  
&emsp; Log-likelihood of the data under the model.  
- **training_likelihoods_**: *list of floats*  
&emsp; List of recorded log-likelihoods during training.  
- **imputing_likelihoods_**: *list of floats*  
&emsp; List of recorded log-likelihoods during imputation.  
- **mean_**: *list of floats*  
&emsp; The mean values for each feature observed during training.  
- **std_**: *list of floats*  
&emsp; The standard deviations for each feature observed during training.  
- **binary_info_**: *list*  
&emsp; The information about binary features observed during training.  
- **integer_info_**: *list*  
&emsp; The information about integer features observed during training.  
- **categorical_info_**: *list*  
&emsp; The information about categorical features observed during training.  
- **random_state_**: *RandomState instance*  
&emsp; RandomState that is generated from a seed or a random number generator.


### References:

[Alvara Correia, Gennaro Gala, Erik Quaeghebeur, Cassio de Campos and Robert Peharz, (2023). Continuous Mixtures of Tractable Probabilistic Models. Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 37, No. 6, Pages 7244-7252.](https://arxiv.org/abs/2209.10584) 


### Basic Example:

```python
>>> from cm_tpm import CMImputer
>>> import numpy as np
>>> X = [[1, 2, np.nan], [4, 5, 6], [np.nan, 8, 9]]
>>> imputer = CMImputer(random_state=0)
>>> imputer.fit_transform(X)
array([[1., 2., 3.],
       [4., 5., 6.],
       [1., 8., 9.]])
```
For more detailed examples, see [several examples](#examples) in this document.


---
## Methods

#### `fit(X, save_model_path=None, sep=',', decimal='.')`  
&emsp; Fits the imputer on *X*. In other words, this functions trains a CM-TPM using training data *X*. &emsp;&emsp;&emsp;&emsp;&emsp;&emsp; [[source code]](https://github.com/Hakim-Agni/cm-tpm/blob/master/src/cm_tpm/_cm.py#L233)  
<br></br>
&emsp; **Parameters**:  
&emsp; &emsp; - **X** : *array-like of shape (n_samples, n_features) or a filepath (csv, xlsx, parquet, or feather)*  
&emsp; &emsp; &emsp; Input data with 2 dimensions used to train the model.  
&emsp; &emsp; - **save_model_path** : *str or None, default=None*  
&emsp; &emsp; &emsp; File location to save the trained model. If None, the model is not saved to a file.  
&emsp; &emsp; - **sep** : *str, default=','*  
&emsp; &emsp; &emsp; Delimiter for csv files.  
&emsp; &emsp; - **decimal** : *str, default='.'*  
&emsp; &emsp; &emsp; Decimal seperator for csv files.  
<br></br>
&emsp; **Returns**:  
&emsp; &emsp; - **self** : *object*  
&emsp; &emsp; &emsp; The fitted `CMImputer` instance.  
<br></br>
&emsp; **Raises**:  
&emsp; &emsp; - **ImportError** : If one of {xlsx, parquet, feather} files is used without the required dependency.  

<br></br>
#### `transform(X, save_output_path=None, sep=',', decimal='.', return_format='auto')`  
&emsp; Imputes all missing values in the input dataset. &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; &emsp;&emsp;&emsp;&emsp;&emsp; [[source code]](https://github.com/Hakim-Agni/cm-tpm/blob/master/src/cm_tpm/_cm.py#L301)  
<br></br>
&emsp; **Parameters**:  
&emsp; &emsp; - **X** : *array-like of shape (n_samples, n_features) or a filepath (csv, xlsx, parquet, or feather)*  
&emsp; &emsp; &emsp; Input data with 2 dimensions, the missing values in this data are imputed.  
&emsp; &emsp; - **save_output_path** : *str or None, default=None*  
&emsp; &emsp; &emsp; If a file location is provided, saves the output data to a file in that location.  
&emsp; &emsp; &emsp; If this parameter is not set and **X** is a file location, the output is saved to '**X** + _imputed'.  
&emsp; &emsp; &emsp; Otherwise, the output is not saved to a file.  
&emsp; &emsp; - **sep** : *str, default=','*  
&emsp; &emsp; &emsp; Delimiter for csv files.  
&emsp; &emsp; - **decimal** : *str, default='.'*  
&emsp; &emsp; &emsp; Decimal seperator for csv files.  
&emsp; &emsp; - **return_format** : {'auto', 'ndarray', 'dataframe'}, default='auto'  
&emsp; &emsp; &emsp; Format of the returned imputed data. 'auto' returns the same format as **X**, 'ndarray' always returns a `NumPy` array,  
&emsp; &emsp; &emsp; and 'dataframe' always returns a `pandas` dataframe.  
<br></br>
&emsp; **Returns**:  
&emsp; &emsp; - **X_imputed** : *array-like of shape (n_samples, n_features)*  
&emsp; &emsp; &emsp; The imputed dataset.  
<br></br>
&emsp; **Raises**:  
&emsp; &emsp; - **ValueError** : If the model is not fitted.  
&emsp; &emsp; - **ValueError** : If an unknown file format is provided as a save path.  
&emsp; &emsp; - **ImportError** : If one of {xlsx, parquet, feather} files is used without the required dependency.  

<br></br>
#### `fit_transform(X, save_model_path=None, save_output_path=None, sep=',', decimal='.', return_format='auto')`  
&emsp; Fits the imputer and then imputes the missing values in the input data. &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; &emsp;&emsp;&emsp;&emsp;&emsp; [[source code]](https://github.com/Hakim-Agni/cm-tpm/blob/master/src/cm_tpm/_cm.py#L435)  
<br></br>
&emsp; **Parameters**:  
&emsp; &emsp; - **X** : *array-like of shape (n_samples, n_features) or a filepath (csv, xlsx, parquet, or feather)*  
&emsp; &emsp; &emsp; Input data with 2 dimensions, the missing values in this data are imputed.  
&emsp; &emsp; - **save_model_path** : *str or None, default=None*  
&emsp; &emsp; &emsp; File location to save the trained model. If None, the model is not saved to a file.  
&emsp; &emsp; - **save_output_path** : *str or None, default=None*  
&emsp; &emsp; &emsp; If a file location is provided, saves the output data to a file in that location.  
&emsp; &emsp; &emsp; If this parameter is not set and **X** is a file location, the output is saved to '**X** + _imputed'.  
&emsp; &emsp; &emsp; Otherwise, the output is not saved to a file.  
&emsp; &emsp; - **sep** : *str, default=','*  
&emsp; &emsp; &emsp; Delimiter for csv files.  
&emsp; &emsp; - **decimal** : *str, default='.'*  
&emsp; &emsp; &emsp; Decimal seperator for csv files.  
&emsp; &emsp; - **return_format** : {'auto', 'ndarray', 'dataframe'}, default='auto'  
&emsp; &emsp; &emsp; Format of the returned imputed data. 'auto' returns the same format as **X**, 'ndarray' always returns a `NumPy` array,  
&emsp; &emsp; &emsp; and 'dataframe' always returns a `pandas` dataframe.  
<br></br>
&emsp; **Returns**:  
&emsp; &emsp; - **X_imputed** : *array-like of shape (n_samples, n_features)*  
&emsp; &emsp; &emsp; The imputed dataset.  
<br></br>
&emsp; **Raises**:  
&emsp; &emsp; - **ValueError** : If an unknown file format is provided as a save path.  
&emsp; &emsp; - **ImportError** : If one of {xlsx, parquet, feather} files is used without the required dependency.  

<br></br>
#### `save_model(path)`  
&emsp; Saves a trained model to a specified file location. &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; &emsp;&emsp;&emsp;&emsp; [[source code]](https://github.com/Hakim-Agni/cm-tpm/blob/master/src/cm_tpm/_cm.py#L453)  
<br></br>
&emsp; **Parameters**:  
&emsp; &emsp; - **path** : *str*    
&emsp; &emsp; &emsp; The file location to save the model to.   
<br></br> 
&emsp; **Returns**:  
&emsp; &emsp; - **None**  
<br></br>
&emsp; **Raises**:  
&emsp; &emsp; - **ValueError** : If the model is not fitted.  

<br></br>
#### `load_model(path)`  
&emsp; Loads a trained model from a specified file location. This is a class method, meaning it can be called without initializing  
&emsp; a `CMImputer` instance, see [Examples](#model-saving-and-loading). &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; [[source code]](https://github.com/Hakim-Agni/cm-tpm/blob/master/src/cm_tpm/_cm.py#L505)  
<br></br>
&emsp; **Parameters**:  
&emsp; &emsp; - **path** : *str*    
&emsp; &emsp; &emsp; The file location where the model is stored.   
<br></br> 
&emsp; **Returns**:  
&emsp; &emsp; - **CMImputer** : *object*  
&emsp; &emsp; &emsp; The loaded `CMImputer` instance.   
<br></br>
&emsp; **Raises**:  
&emsp; &emsp; - **FileNotFoundError** : If the file location does not contain model files.  

<br></br>
#### `transform_from_file(X, load_model_path, )`  
&emsp; Imputes all missing values in the input data using a `CMImputer` loaded from a file. This is a class method, meaning  
&emsp; it can be called without initializing a `CMImputer` instance, see [Examples](#model-saving-and-loading). &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; &emsp;&emsp;&emsp;&emsp; [[source code]](https://github.com/Hakim-Agni/cm-tpm/blob/master/src/cm_tpm/_cm.py#L401)  
<br></br>
&emsp; **Parameters**:  
&emsp; &emsp; - **X** : *array-like of shape (n_samples, n_features) or a filepath (csv, xlsx, parquet, or feather)*  
&emsp; &emsp; &emsp; Input data with 2 dimensions, the missing values in this data are imputed.  
&emsp; &emsp; - **load_model_path** : *str*    
&emsp; &emsp; &emsp; The file location where the model is stored.   
&emsp; &emsp; - **save_output_path** : *str or None, default=None*  
&emsp; &emsp; &emsp; If a file location is provided, saves the output data to a file in that location.  
&emsp; &emsp; &emsp; If this parameter is not set and **X** is a file location, the output is saved to '**X** + _imputed'.  
&emsp; &emsp; &emsp; Otherwise, the output is not saved to a file.  
&emsp; &emsp; - **sep** : *str, default=','*  
&emsp; &emsp; &emsp; Delimiter for csv files.  
&emsp; &emsp; - **decimal** : *str, default='.'*  
&emsp; &emsp; &emsp; Decimal seperator for csv files.  
&emsp; &emsp; - **return_format** : {'auto', 'ndarray', 'dataframe'}, default='auto'  
&emsp; &emsp; &emsp; Format of the returned imputed data. 'auto' returns the same format as **X**, 'ndarray' always returns a `NumPy` array,  
&emsp; &emsp; &emsp; and 'dataframe' always returns a `pandas` dataframe.  
<br></br>
&emsp; **Returns**:  
&emsp; &emsp; - **X_imputed** : *array-like of shape (n_samples, n_features)*  
&emsp; &emsp; &emsp; The imputed dataset.  
<br></br>
&emsp; **Raises**:  
&emsp; &emsp; - **ValueError** : If a model load path is not specified.  
&emsp; &emsp; - **FileNotFoundError** : If the model file location does not contain model files.  
&emsp; &emsp; - **ValueError** : If the model is not fitted.  
&emsp; &emsp; - **ValueError** : If an unknown file format is provided as a save path.  
&emsp; &emsp; - **ImportError** : If one of {xlsx, parquet, feather} files is used without the required dependency.   

<br></br>
#### `get_feature_names_out(input_features=None)`  
&emsp; Returns the feature names of the input data. In none exist, generates standard feature names. &emsp;&emsp;&emsp;&emsp;&emsp; [[source code]](https://github.com/Hakim-Agni/cm-tpm/blob/master/src/cm_tpm/_cm.py#L583)  
<br></br>
&emsp; **Parameters**:  
&emsp; &emsp; - **input_features** : *list of str, default=None*    
&emsp; &emsp; &emsp; List of feature names. If None, uses feature names seen during **fit()**.  
<br></br> 
&emsp; **Returns**:  
&emsp; &emsp; - **feature_names_out** : *list of str*  
&emsp; &emsp; &emsp; A list containing the output feature names.  
<br></br>
&emsp; **Raises**:  
&emsp; &emsp; - **ValueError** : If the model is not fitted.  
&emsp; &emsp; - **ValueError** : If input_features is not equal to feauture_names_in.  
&emsp; &emsp; - **ValueError** : If the length of input_features is not equal to n_features_in_.  
&emsp; &emsp; - **ValueError** : If n_features_in_ is not set.  

<br></br>
#### `get_params()`  
&emsp; Gets the hyperparameters for the `CMImputer`. &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; &emsp;&emsp;&emsp;&emsp;&emsp; [[source code]](https://github.com/Hakim-Agni/cm-tpm/blob/master/src/cm_tpm/_cm.py#L622)   
<br></br>
&emsp; **Parameters**:  
&emsp; &emsp; - **None**  
<br></br> 
&emsp; **Returns**:  
&emsp; &emsp; - **params** : *dict*    
&emsp; &emsp; &emsp; Parameter names mapped to their values.  

<br></br>
#### `set_params(**params)`  
&emsp; Sets the hyperparameters for the `CMImputer`. &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; &emsp;&emsp;&emsp;&emsp;&emsp; [[source code]](https://github.com/Hakim-Agni/cm-tpm/blob/master/src/cm_tpm/_cm.py#L662)    
<br></br>
&emsp; **Parameters**:  
&emsp; &emsp; - **params** : *dict*    
&emsp; &emsp; &emsp; Parameter names mapped to their updated values.   
<br></br> 
&emsp; **Returns**:  
&emsp; &emsp; - **self** : *object*   
&emsp; &emsp; &emsp; An updated `CMImputer` instance.  
<br></br>
&emsp; **Raises**:  
&emsp; &emsp; - **ValueError** : If an invalid parameter is passed.  

<br></br>
#### `evaluate(X)`   
&emsp; Evaluates how well the trained model explains the input data using log-likleihood. &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; [[source code]](https://github.com/Hakim-Agni/cm-tpm/blob/master/src/cm_tpm/_cm.py#L682)  
<br></br>
&emsp; **Parameters**:  
&emsp; &emsp; - **X** : *array-like of shape (n_samples, n_features) or a filepath (csv, xlsx, parquet, or feather)*  
&emsp; &emsp; &emsp; Input data with 2 dimensions, the missing values in this data are imputed.   
<br></br> 
&emsp; **Returns**:  
&emsp; &emsp; - **log_likelihood** : *float*   
&emsp; &emsp; &emsp; The log-likelihood of the input data under the trained model.  
<br></br>
&emsp; **Raises**:  
&emsp; &emsp; - **ValueError** : If the model is not fitted.  

---
## Examples

Here, you can find various example of uses of the package. These examples are sorted by category:
- [Core Functionality](#core-functionality)  
&emsp; - Fit and Transform Usage  
&emsp; - Fit_transform Usage  
&emsp; - Categorical Data  
- [Hyperparameters](#hyperparameters)  
&emsp; - Fast Settings  
&emsp; - Balanced Settings  
&emsp; - Precise Settings  
&emsp; - Custom Settings  
&emsp; - Getting Hyperparameters  
&emsp; - Setting Hyperparameters  
&emsp; - Setting a Custom Neural Network  
- [Related to Data](#related-to-data)  
&emsp; - Multiple Missing Value Types  
&emsp; - Ordinal Data  
&emsp; - CSV Files  
&emsp; - XSLX Files  
&emsp; - Parquet Files  
&emsp; - Feather Files  
&emsp; - Outputting to a File  
- [Model Saving and Loading](#model-saving-and-loading)  
&emsp; - Saving a Trained Model  
&emsp; - Saving Directly using Fit  
&emsp; - Loading a Trained Model  
&emsp; - Loading and Using a Trained Model Directly  
- [Evaluation](#evaluation)  
&emsp; - Evaluating the Model Fit  
&emsp; - Evaluation Imputation Quality  
- [Debugging](#debugging)  
&emsp; - Verbose Settings  
&emsp; - Verbose Example  
&emsp; - Plotting Log-Likelihoods throughout Training  
- [Miscellaneous](#miscellaneous)  
&emsp; - Setting Feature Names
  
---
### Core Functionality

---
#### Fit and Transform Usage
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
**Expected output**:
```python
array([[1., 2.],
       [2., 3.],
       [3., 4.]])
```
---
#### Fit_transform Usage
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
**Expected output**:
```python
array([[1., 4.],
       [2., 3.],
       [1., 4.]])
```

---
#### Categorical Data
```python
import numpy as np
from cm_tpm import CMImputer

# Example data with and without missing values
X_train = np.array([['Red', 'Blue'], ['Green', 'Blue'], ['Blue', 'Red'], ['Red', 'Yellow']])
X_missing = np.array([['Blue', np.nan], ['Green', np.nan], [np.nan, 'Blue']])

# Initialize and fit imputer
imputer = CMImputer()
imputer.fit(X_train)

# Impute missing values
X_imputed = imputer.transform(X_missing)

# Print the imputed data
print(X_imputed)
```
**Expected output**:
```python
array([['Blue', 'Red'],
       ['Green', 'Blue'],
       ['Red', 'Blue']])
```

---
### Hyperparameters

---
#### Fast Settings
```python
import numpy as np
from cm_tpm import CMImputer

# Example data with missing values
X = np.array([[1.0, np.nan], [2.0, 3.0], [np.nan, 4.0]])

# Initialize imputer with 'fast' settings
imputer = CMImputer(settings='fast')

# Fit the imputer
X_imputed = imputer.fit_transform(X)

# Print the imputed data
print(X_imputed)
```
**Notes**:
- Parameters set by the 'fast' setting **cannot** be overridden.

**Expected output**:
```python
array([[1., 5.],
       [2., 3.],
       [1., 4.]])
```
Using this setting option, you can expect quick results, but less accurate than the other settings.

---
#### Balanced Settings
```python
import numpy as np
from cm_tpm import CMImputer

# Example data with missing values
X = np.array([[1.0, np.nan], [2.0, 3.0], [np.nan, 4.0]])

# Initialize imputer with 'balanced' settings
imputer = CMImputer(settings='balanced')    # Note: This is the same as using 'imputer = CMImputer()'

# Fit the imputer
X_imputed = imputer.fit_transform(X)

# Print the imputed data
print(X_imputed)
```
**Notes**:
- Parameters set by the 'balanced' setting **cannot** be overridden.

**Expected output**:
```python
array([[1., 3.],
       [2., 3.],
       [1., 4.]])
```
Using this setting option, you can expect better results than using the 'fast' setting, but slightly slower.

---
#### Precise Settings
```python
import numpy as np
from cm_tpm import CMImputer

# Example data with missing values
X = np.array([[1.0, np.nan], [2.0, 3.0], [np.nan, 4.0]])

# Initialize imputer with 'precise' settings
imputer = CMImputer(settings='precise')

# Fit the imputer
X_imputed = imputer.fit_transform(X)

# Print the imputed data
print(X_imputed)
```
**Notes**:
- Parameters set by the 'precise' setting **cannot** be overridden.

**Expected output**:
```python
array([[1., 4.],
       [2., 3.],
       [1., 4.]])
```
Using this setting option, you can expect accurate results, but slower than the other settings.

---
#### Custom Settings
```python
import numpy as np
from cm_tpm import CMImputer

# Example data with missing values
X = np.array([[1.0, np.nan], [2.0, 3.0], [np.nan, 4.0]])

# Initialize imputer with custom settings
imputer = CMImputer(settings='custom',
                    n_components_train=256,
                    n_components_impute=1024,
                    latent_dim=8,
                    top_k=3,
                    imputation_method="optimization",
                    max_iter=50)

# Fit the imputer
X_imputed = imputer.fit_transform(X)

# Print the imputed data
print(X_imputed)
```
**Notes**:
- Here, we use a random selection of hyperparameter for demonstration purposed. In practice, you can set every [hyperparameter](#parameters) as you wish.  

**Expected output**:
```python
array([[1., 4.],
       [2., 3.],
       [1., 4.]])
```

---
#### Getting Hyperparameters
```python
from cm_tpm import CMImputer

# Initialize imputer with custom settings
imputer = CMImputer()

# Get the hyperparameters of the imputer
params = imputer.get_params()

# Print the parameters
print(params)
```
**Expected output**:
```python
{'settings': 'custom', 'missing_values': nan, 'n_components_train': 256, 'n_components_impute': 2048, 'latent_dim': 4, 'top_k': None,   
'lo': False, 'pc_type': 'factorized', 'imputation_method': 'EM', 'ordinal_features': None, 'max_depth': 5, 'custom_net': None,   
'hidden_layers': 4, 'neurons_per_layer': 512, 'activation': 'LeakyReLU', 'batch_norm': True, 'dropout_rate': 0.1,   
'skip_layers': False, 'max_iter': 100, 'batch_size_train': 1024, 'batch_size_impute': 256, 'tol': 0.0001, 'patience': 10,   
'lr': 0.001, 'weight_decay': 0.01, 'use_gpu': True, 'random_state': None, 'verbose': 0, 'copy': True, 'keep_empty_features': False}  
```
---
#### Setting Hyperparameters (after initialing)
```python
from cm_tpm import CMImputer

# Initialize imputer with custom settings
imputer = CMImputer()

# Set some hyperparameters of the imputer
imputer.set_params(
    missing_values='',
    n_components_train=128,
    n_components_impute=128,
)

# Get the relevant hyperparameters
params = imputer.get_params()
params = {key: params[key] for key in ["missing_values", "n_components_train", "n_components_impute"]}

# Print the parameters
print(params)
```
**Expected output**:
```python
{'missing_values': '', 'n_components_train': 128, 'n_components_impute': 128}
```

---
#### Setting a Custom Neural Network
```python
import numpy as np
import torch.nn as nn
from cm_tpm import CMImputer

# Example data with missing values
X = np.array([[1.0, np.nan], [2.0, 3.0], [np.nan, 4.0]])

# Create a custom (sequential) neural network
neural_network = nn.Sequential(
    nn.Linear(4, 64),
    nn.ReLU(),
    nn.Linear(64, 256),
    nn.ReLU(),
    nn.Linear(256, 4),
)

# Initialize imputer with the custom neural network
imputer = CMImputer(custom_net=neural_network)

# Fit the imputer
X_imputed = imputer.fit_transform(X)

# Print the imputed data
print(X_imputed)
```
**Notes**:
- Assert that the number of input features in the first layer is equal to the latent dimension (in this case 4).  
- Assert that the number of output features in the last layer is equal to 2 * the number of  features (in this case also 4).  
- Currently, only Sequential neural networks from `torch` are supported.  

**Expected output**:
```python
array([[1., 4.]
       [2., 3.]
       [1., 4.]])
```
---
### Related to Data
---
#### Multiple Missing Value Types
```python
import numpy as np
from cm_tpm import CMImputer

# Example data with missing values
X = np.array([[1.0, np.nan], [2.0, 3.0], ["", 4.0]])

# Initialize and fit imputer
imputer = CMImputer(missing_values=[np.nan, ""])   # Set the correct missing values
X_imputed = imputer.fit_transform(X)

# Print the imputed data
print(X_imputed)
```
**Expected output**:
```python
array([['1.0', '4.0'],
       ['2.0', '3.0'],
       ['1.0', '4.0']])
```
---
#### Ordinal Data
```python
import numpy as np
from cm_tpm import CMImputer

# Example data with and without missing values
X_train = np.array([["Low", 1], ["Medium", 3], ["High", 10], ["Low", 2], ["High", 8]])
X_missing = np.array([["Low", np.nan], [np.nan, 6], [np.nan, 10]])

# Set the correct ordinal data information
ordinal_info = {0: {"Low": 0, "Medium": 1, "High": 2}}

# Initialize and fit imputer
imputer = CMImputer(ordinal_features=ordinal_info)   
imputer.fit(X_train)

# Impute missing values
X_imputed = imputer.transform(X_missing)

# Print the imputed data
print(X_imputed)
```
**Expected output**:
```python
array([['Low', '2.0'],
       ['Medium', '6'],
       ['High', '10']])
```
---
#### CSV Files
```python
from cm_tpm import CMImputer

# Example file path pointing to a CSV file
filepath = 'path/to/data.csv'      # Replace with an actual file path!

# Initialize and fit imputer
imputer = CMImputer()
X_imputed = imputer.fit_transform(filepath, sep=',', decimal='.')

# Print the imputed data
print(X_imputed)
```
**Notes**:
- For CSV files specifically, make sure to enter the correct delimiter and decimal separator in the `fit`, `transform` or `fit_transform` function.

**Expected output**:
- Depends on the data stored in 'path/to/data.csv'.
- The output is also saved to a file: 'path/to/data_imputed.csv'. For another save location, set the `save_output_path` parameter in the `transform` or `fit_transform` function.
---
#### XLSX Files
```python
from cm_tpm import CMImputer

# Example file path pointing to a XLSX file
filepath = 'path/to/data.xlsx'      # Replace with an actual file path!

# Initialize and fit imputer
imputer = CMImputer()
X_imputed = imputer.fit_transform(filepath)

# Print the imputed data
print(X_imputed)
```
**Notes**:
- To use XLSX files, make sure to install the corresponding optional dependency, using: `pip install cm-tpm[excel]`.

**Expected output**:
- Depends on the data stored in 'path/to/data.xlsx'.
- The output is also saved to a file: 'path/to/data_imputed.xlsx'. For another save location, set the `save_output_path` parameter in the `transform` or `fit_transform` function.
---
#### Parquet Files
```python
from cm_tpm import CMImputer

# Example file path pointing to a Parquet file
filepath = 'path/to/data.parquet'      # Replace with an actual file path!

# Initialize and fit imputer
imputer = CMImputer()
X_imputed = imputer.fit_transform(filepath)

# Print the imputed data
print(X_imputed)
```
**Notes**:
- To use Parquet files, make sure to install the corresponding optional dependency, using: `pip install cm-tpm[parquet]`.

**Expected output**:
- Depends on the data stored in 'path/to/data.parquet'.
- The output is also saved to a file: 'path/to/data_imputed.parquet'. For another save location, set the `save_output_path` parameter in the `transform` or `fit_transform` function.
---
#### Feather Files
```python
from cm_tpm import CMImputer

# Example file path pointing to a Feather file
filepath = 'path/to/data.feather'      # Replace with an actual file path!

# Initialize and fit imputer
imputer = CMImputer()
X_imputed = imputer.fit_transform(filepath)

# Print the imputed data
print(X_imputed)
```
**Notes**:
- To use Parquet files, make sure to install the corresponding optional dependency, using: `pip install cm-tpm[parquet]`.

**Expected output**:
- Depends on the data stored in 'path/to/data.feather'.
- The output is also saved to a file: 'path/to/data_imputed.feather'. For another save location, set the `save_output_path` parameter in the `transform` or `fit_transform` function.
---
#### Outputting to a File
```python
import numpy as np
from cm_tpm import CMImputer

# Example data with missing values
X = np.array([[1.0, np.nan], [2.0, 3.0], [np.nan, 4.0]])

# Example filepath where the output must be saved
out_filepath = 'path/to/output/data.csv'     # Replace with an actual file path!

# Initialize and fit imputer
imputer = CMImputer()
X_imputed = imputer.fit_transform(X, save_output_path=out_filepath)

# Print the imputed data
print(X_imputed)
```
**Notes**:
- The file path can point to one of the following file types: {'.csv', '.xlsx', '.parquet', '.feather'}. In this example, we save to a CSV file.
- To save to a XSLS, Parquet, or Feather file, make sure to install the relevant optional dependency: `pip install cm-tpm[excel]` or `pip install cm-tpm[parquet]`.

**Expected output**:
```python
array([['1.0', '4.0'],
       ['2.0', '3.0'],
       ['1.0', '4.0']])
```
---
### Model Saving and Loading
---
#### Saving a Trained Model
```python
import numpy as np
from cm_tpm import CMImputer

# Example data without missing values
X_train = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])

# Example file path where the model must be saved
filepath = 'path/to/model'      # Replace with an actual file path!

# Initialize and fit imputer
imputer = CMImputer()
imputer.fit(X_train)

# Save the model to a file
imputer.save_model(filepath)
```
**Notes**:
- The file path points to a folder where the model will be saved. Two files will be created in this folder: 'config.json' and 'model.pt'.
- If a custom neural network is used, a third file will be created: 'custom_net.pt'.

**Expected output**:
```
Model has been successfully saved at 'path/to/model'.
```
---
#### Saving Directly using `fit`
```python
import numpy as np
from cm_tpm import CMImputer

# Example data without missing values
X_train = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])

# Example file path where the model must be saved
filepath = 'path/to/model'      # Replace with an actual file path!

# Initialize imputer
imputer = CMImputer()

# Fit imputer with the file path to save the model specified
imputer.fit(X_train, save_model_path=filepath)
```
**Notes**:
- The file path points to a folder where the model will be saved. Two files will be created in this folder: 'config.json' and 'model.pt'.
- If a custom neural network is used, a third file will be created: 'custom_net.pt'.

**Expected output**:
```
Model has been succesfully saved at 'path/to/model'.
```
---
#### Loading a Trained Model
```python
import numpy as np
from cm_tpm import CMImputer

# Example data with missing values
X_missing = np.array([[1.0, np.nan], [2.0, 3.0], [np.nan, 4.0]])

# Example file path where the model must be saved
filepath = 'path/to/model'      # Replace with an actual file path!

# load a saved imputer
imputer = CMImputer.load_model(filepath)

# Impute missing values
X_imputed = imputer.transform(X_missing)

# Print the imputed data
print(X_imputed)
```
**Expected output**:
```python
array([[1., 1.],
       [2., 3.],
       [3., 4.]])
```
---
#### Loading and Using a Trained Model Directly using `transform_from_file`
```python
import numpy as np
from cm_tpm import CMImputer

# Example data with missing values
X_missing = np.array([[1.0, np.nan], [2.0, 3.0], [np.nan, 4.0]])

# Example file path where the model must be saved
filepath = 'path/to/model'      # Replace with an actual file path!

# Impute missing values using a saved imputer
X_imputed = CMImputer.transform_from_file(X_missing, filepath)

# Print the imputed data
print(X_imputed)
```
**Notes**:
- This method does not require explicitly initiating a `CMImputer` instance. 

**Expected output**:
```python
array([[1., 1.],
       [2., 3.],
       [3., 4.]])
```
---
### Evaluation

---
#### Evaluating the Model Fit
```python
import numpy as np
from cm_tpm import CMImputer

# Example data without missing values
X_train = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])

# Initialize and fit first imputer
imputer1 = CMImputer()
imputer1.fit(X_train)

# Initialize and fit second imputer
imputer2 = CMImputer()
imputer2.fit(X_train)

# Evaluate the likelihood of the training data under each model
likelihood1 = imputer1.evaluate(X_train)
likelihood2 = imputer2.evaluate(X_train)

# Print the likelihoods
print(likelihood1)
print(likelihood2)
```
**Notes**:
- The higher the log-likelihood, the better.
- **Tip** : Try using different hyperparameter settings for the two imputer to see which settings work best on the training data.

**Expected output**:
```python
-0.9929150938987732
-1.2049541473388672
```
Since a higher log-likelihood is better, we conclude that the first imputer is better fit to the training data in this case.

---
#### Evaluation Imputation Quality (using log-likelihood)
```python
import numpy as np
from cm_tpm import CMImputer

# Example data with and without missing values
X_train = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
X_missing = np.array([[1.0, np.nan], [2.0, 3.0], [np.nan, 4.0]])

# Initialize and fit two imputers
imputer1 = CMImputer()
imputer1.fit(X_train)
imputer2 = CMImputer()
imputer2.fit(X_train)

# Impute the missing data twice for comparing
X_imputed1 = imputer1.transform(X_missing)
X_imputed2 = imputer2.transform(X_missing)

# Evaluate the likelihood of the imputed data under each model
likelihood1 = imputer1.evaluate(X_imputed1)
likelihood2 = imputer2.evaluate(X_imputed2)

# Print the imputed data with their respective likelihoods
print(X_imputed1, likelihood1)
print(X_imputed2, likelihood2)
```
**Notes**:
- The higher the log-likelihood, the better.
- **Tip** : Try using different hyperparameter settings for the two imputer to see which settings work best on the training data.

**Expected output**:
```python
array([[1., 3.],
       [2., 3.],
       [0., 4.]]), -2.0776379108428955
array([[1., 2.],
       [2., 3.],
       [2., 4.]]), -1.42733895778656
```
In this case, we see that the second imputation is better than the first.

---
### Debugging

---
#### Verbose settings
The package offers four levels of verbosity:
- **0 - Silent** (default): No output is shown. Suitable for production or when debugging is not
needed.  
- **1 - Basic**: Displays essential information, including:  
&emsp; – Progress bars for training and imputation.  
&emsp; &emsp; **Note**: This functionality requires and additional dependency, which can be installed using: `pip install cm-tpm[tqdm]`.    
&emsp; – Whether the model is using GPU or CPU.  
&emsp; – Timestamps for the start of training and imputation.  
&emsp; – Confirmation of successful completion of training and imputation.  
&emsp; – Final log-likelihood values for training and imputation.  
- **2 - Detailed**: Adds the following information to level 1 (progress bars are replaced by
textual updates):  
&emsp; – Confirmation when a CM-TPM is successfully created.  
&emsp; – Time taken for key steps (e.g., pre-processing, training, imputation).  
&emsp; – Log-likelihood values every 10 epochs during training and imputation.  
&emsp; – Notification when early stopping is triggered during training.  
- **3 - Full**: Extends level 2 by also displaying:  
&emsp; – Log-likelihood values at every epoch during training and imputation.  

The verbosity level can be selected when initializing a `CMImputer` instance:
```python
imputer = CMImputer(verbose=3)      # Replace with desired verbose level
```

---
#### Verbose Example
```python
import numpy as np
from cm_tpm import CMImputer

# Example data with missing values
X = np.array([[1.0, np.nan], [2.0, 3.0], [np.nan, 4.0]])

# Initialize and fit imputer
imputer = CMImputer(verbose=1)      # Set verbose to a value > 0
X_imputed = imputer.fit_transform(X)

# Print the imputed data
print(X_imputed)
```
**Notes**
- Setting **verbose** equal to 1 enables the use of `tqdm` progress bars. To display the progress bars correctly, make sure to install the optional dependency, using: `pip install cm-tpm[tqdm]`.
- When using the 'optimization' imputation method, a progress bar is also shown for the imputation process.

**Expected output**:
```
Starting training with 100 epochs...
Using device: cpu
Training:  37%|███▋      | 37/100 [00:00<00:00, 75.70it/s]
Training complete.
Final Training Log-Likelihood: -0.6828761100769043
Starting with imputing data...
Using device: cpu
Finished imputing data.
Successfully imputed 2 missing values across 2 samples.
Final imputed data log-likelihood: -1.5252711772918701
```
```python
array([[1., 4.],
       [2., 3.],
       [1., 4.]])
```

---
#### Plotting Log-Likelihoods throughout Training
```python
import numpy as np
import matplotlib.pyplot as plt
from cm_tpm import CMImputer

# Example data with missing values
X = np.array([[1.0, np.nan], [2.0, 3.0], [np.nan, 4.0]])

# Initialize and fit imputer
imputer = CMImputer()
X_imputed = imputer.fit_transform(X)

# Get the log-likelihood throughout training
train_likelihoods = imputer.training_likelihoods_

# Plot the log-likelihoods in a graph (using matplotlib)
plt.plot(train_likelihoods)
plt.xlabel('Epoch')
plt.ylabel('Log-Likelihood')
plt.show()
```
**Notes**:
- In this example we use `matplotlib` for plotting the graph. This package is **not** included `cm-tpm`, it must be installed separately: `pip install matplotlib`.
- When using the 'optimization' imputation method, the log-likelihoods during imputation can also be plotted. These values are found in the `imputing_likelihoods_` attribute of `CMImputer`.

**Expected output**:  
![LL-plot](https://github.com/user-attachments/assets/bd839419-be7c-4b5f-805c-350ed9018c6b)  
This plot displays unstable training, which is caused by using a tiny dataset for training.

---
### Miscellaneous

---
#### Setting Feature Names
```python
import numpy as np
from cm_tpm import CMImputer

# Example data with and without missing values
X_train = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
X_missing = np.array([[1.0, np.nan], [2.0, 3.0], [np.nan, 4.0]])

# Initialize and fit imputer
imputer = CMImputer()
imputer.fit(X_train)

# Set custom feature names to the data
imputer.get_feature_names_out(input_features=['column_1', 'column_2'])

# Impute missing values
X_imputed = imputer.transform(X_missing, return_format='dataframe')

# Print the imputed data
print(X_imputed)
```
**Notes**:
- To display the new feature names, we choose the return format 'dataframe'.
- When the function `get_feature_names_out()` is called without custom *input_features*, the feature names are defaulted to 'x0' and 'x1'.

**Expected Output**:
```python
   column_1  column_2
0       1.0       3.0
1       2.0       3.0
2       5.0       4.0
```

---
