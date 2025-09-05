################################################################################
#                                metrics                                       #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8

from typing import Union, Callable
import numpy as np
import pandas as pd
import inspect
from functools import wraps
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_log_error,
)

def _get_metric(metric: str) -> Callable:
    """
    Get the corresponding scikit-learn function to calculate the metric.

    Parameters
    ----------
    metric : str
        Metric used to quantify the goodness of fit of the model.

    Returns
    -------
    metric : Callable
        scikit-learn function to calculate the desired metric.

    """
    allowed_metrics = [
        "mean_squared_error",
        "mean_absolute_error",
        "mean_absolute_percentage_error",
        "mean_squared_log_error",
        "mean_absolute_scaled_error",
        "root_mean_squared_scaled_error",
    ]

    if metric not in allowed_metrics:
        raise ValueError((f"Allowed metrics are: {allowed_metrics}. Got {metric}."))

    metrics = {
        "mean_squared_error": mean_squared_error,
        "mean_absolute_error": mean_absolute_error,
        "mean_absolute_percentage_error": mean_absolute_percentage_error,
        "mean_squared_log_error": mean_squared_log_error,
        "mean_absolute_scaled_error": mean_absolute_scaled_error,
        "root_mean_squared_scaled_error": root_mean_squared_scaled_error,
    }

    metric = add_y_train_argument(metrics[metric])

    return metric


def add_y_train_argument(func):
    """
    Add `y_train` argument to a function if it is not already present.
    Also creates a NaN-aware wrapper for scikit-learn metrics.

    Parameters
    ----------
    func : callable
        Function to which the argument is added.

    Returns
    -------
    wrapper : callable
        Function with `y_train` argument added and NaN handling.
    """
    sig = inspect.signature(func)
    
    if "y_train" in sig.parameters:
        return func

    new_params = list(sig.parameters.values()) + [
        inspect.Parameter("y_train", inspect.Parameter.KEYWORD_ONLY, default=None)
    ]
    new_sig = sig.replace(parameters=new_params)

    @wraps(func)
    def wrapper(*args, y_train=None, **kwargs):
        # Handle NaN values in y_true and y_pred for scikit-learn metrics
        
        # Get y_true and y_pred from either args or kwargs
        if len(args) >= 2:
            y_true, y_pred = args[0], args[1]
            remaining_args = args[2:]
        elif len(args) == 1 and 'y_pred' in kwargs:
            y_true = args[0]
            y_pred = kwargs.pop('y_pred')
            remaining_args = ()
        elif len(args) == 0 and 'y_true' in kwargs and 'y_pred' in kwargs:
            y_true = kwargs.pop('y_true')
            y_pred = kwargs.pop('y_pred')
            remaining_args = ()
        else:
            # Fallback for edge cases - can't identify y_true/y_pred
            return func(*args, **kwargs)
        
        # Convert to numpy arrays for easier handling
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        # Create mask for valid (non-NaN) pairs
        valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        
        # If no valid pairs, return NaN
        if not np.any(valid_mask):
            return np.nan
        
        # Filter to only valid pairs
        y_true_valid = y_true[valid_mask]
        y_pred_valid = y_pred[valid_mask]
        
        # Also filter remaining_args if they have the same length (e.g. sample_weight)
        filtered_remaining_args = []
        for arg in remaining_args:
            if hasattr(arg, '__len__') and len(arg) == len(y_true):
                # This looks like it should be filtered along with y_true/y_pred
                filtered_arg = np.asarray(arg)[valid_mask]
                filtered_remaining_args.append(filtered_arg)
            else:
                # Keep as is
                filtered_remaining_args.append(arg)
        
        # Call the original function with filtered data
        return func(y_true_valid, y_pred_valid, *filtered_remaining_args, **kwargs)
    
    wrapper.__signature__ = new_sig
    
    return wrapper


def mean_absolute_scaled_error(
    y_true: Union[pd.Series, np.array],
    y_pred: Union[pd.Series, np.array],
    y_train: Union[list, pd.Series, np.array],
) -> float:
    """
    Mean Absolute Scaled Error (MASE)
    MASE is a scale-independent error metric that measures the accuracy of
    a forecast. It is the mean absolute error of the forecast divided by the
    mean absolute error of a naive forecast in the training set. The naive
    forecast is the one obtained by shifting the time series by one period.
    If y_train is a list of numpy arrays or pandas Series, it is considered
    that each element is the true value of the target variable in the training
    set for each time series. In this case, the naive forecast is calculated
    for each time series separately.
    
    NaN values in y_train are ignored when calculating the naive forecast.
    If y_true or y_pred contain NaN values, only the non-NaN pairs are used
    for the calculation.

    Parameters
    ----------
    y_true : pd.Series, np.array
        True values of the target variable.
    y_pred : pd.Series, np.array
        Predicted values of the target variable.
    y_train : list, pd.Series, np.array
        True values of the target variable in the training set. If list, it
        is consider that each element is the true value of the target variable
        in the training set for each time series.

    Returns
    -------
    float
        MASE value.
    """
    if not isinstance(y_true, (pd.Series, np.ndarray)):
        raise TypeError("y_true must be a pandas Series or numpy array")
    if not isinstance(y_pred, (pd.Series, np.ndarray)):
        raise TypeError("y_pred must be a pandas Series or numpy array")
    if not isinstance(y_train, (list, pd.Series, np.ndarray)):
        raise TypeError("y_train must be a list, pandas Series or numpy array")
    if isinstance(y_train, list):
        for x in y_train:
            if not isinstance(x, (pd.Series, np.ndarray)):
                raise TypeError(
                    "When `y_train` is a list, each element must be a pandas Series "
                    "or numpy array"
                )
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("y_true and y_pred must have at least one element")

    # Handle NaN values in y_true and y_pred - only use valid pairs
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    
    if not np.any(valid_mask):
        return np.nan
    
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    
    if isinstance(y_train, list):
        # Handle list of arrays/series - filter NaN values from each series
        naive_forecast_diffs = []
        for x in y_train:
            x_array = np.asarray(x)
            x_valid = x_array[~np.isnan(x_array)]
            if len(x_valid) > 1:  # Need at least 2 values to compute diff
                naive_forecast_diffs.append(np.diff(x_valid))
        
        if not naive_forecast_diffs:
            return np.nan
            
        naive_forecast = np.concatenate(naive_forecast_diffs)
        if len(naive_forecast) == 0:
            return np.nan
            
        mase = np.mean(np.abs(y_true_valid - y_pred_valid)) / np.mean(np.abs(naive_forecast))
    else:
        # Handle single array/series - filter NaN values
        y_train_array = np.asarray(y_train)
        y_train_valid = y_train_array[~np.isnan(y_train_array)]
        
        if len(y_train_valid) < 2:  # Need at least 2 values to compute diff
            return np.nan
            
        naive_forecast = np.diff(y_train_valid)
        mase = np.mean(np.abs(y_true_valid - y_pred_valid)) / np.mean(np.abs(naive_forecast))

    return mase


def root_mean_squared_scaled_error(
    y_true: Union[pd.Series, np.array],
    y_pred: Union[pd.Series, np.array],
    y_train: Union[list, pd.Series, np.array],
) -> float:
    """
    Root Mean Squared Scaled Error (RMSSE)
    RMSSE is a scale-independent error metric that measures the accuracy of
    a forecast. It is the root mean squared error of the forecast divided by
    the root mean squared error of a naive forecast in the training set. The
    naive forecast is the one obtained by shifting the time series by one period.
    If y_train is a list of numpy arrays or pandas Series, it is considered
    that each element is the true value of the target variable in the training
    set for each time series. In this case, the naive forecast is calculated
    for each time series separately.
    
    NaN values in y_train are ignored when calculating the naive forecast.
    If y_true or y_pred contain NaN values, only the non-NaN pairs are used
    for the calculation.

    Parameters
    ----------
    y_true : pd.Series, np.array
        True values of the target variable.
    y_pred : pd.Series, np.array
        Predicted values of the target variable.
    y_train : list, pd.Series, np.array
        True values of the target variable in the training set. If list, it
        is consider that each element is the true value of the target variable
        in the training set for each time series.

    Returns
    -------
    float
        RMSSE value.
    """

    if not isinstance(y_true, (pd.Series, np.ndarray)):
        raise TypeError("y_true must be a pandas Series or numpy array")
    if not isinstance(y_pred, (pd.Series, np.ndarray)):
        raise TypeError("y_pred must be a pandas Series or numpy array")
    if not isinstance(y_train, (list, pd.Series, np.ndarray)):
        raise TypeError("y_train must be a list, pandas Series or numpy array")
    if isinstance(y_train, list):
        for x in y_train:
            if not isinstance(x, (pd.Series, np.ndarray)):
                raise TypeError(
                    "When `y_train` is a list, each element must be a pandas Series "
                    "or numpy array"
                )
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("y_true and y_pred must have at least one element")

    # Handle NaN values in y_true and y_pred - only use valid pairs
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    
    if not np.any(valid_mask):
        return np.nan
    
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    
    if isinstance(y_train, list):
        # Handle list of arrays/series - filter NaN values from each series
        naive_forecast_diffs = []
        for x in y_train:
            x_array = np.asarray(x)
            x_valid = x_array[~np.isnan(x_array)]
            if len(x_valid) > 1:  # Need at least 2 values to compute diff
                naive_forecast_diffs.append(np.diff(x_valid))
        
        if not naive_forecast_diffs:
            return np.nan
            
        naive_forecast = np.concatenate(naive_forecast_diffs)
        if len(naive_forecast) == 0:
            return np.nan
            
        rmsse = np.sqrt(np.mean((y_true_valid - y_pred_valid) ** 2)) / np.sqrt(np.mean(naive_forecast ** 2))
    else:
        # Handle single array/series - filter NaN values
        y_train_array = np.asarray(y_train)
        y_train_valid = y_train_array[~np.isnan(y_train_array)]
        
        if len(y_train_valid) < 2:  # Need at least 2 values to compute diff
            return np.nan
            
        naive_forecast = np.diff(y_train_valid)
        rmsse = np.sqrt(np.mean((y_true_valid - y_pred_valid) ** 2)) / np.sqrt(np.mean(naive_forecast ** 2))
    
    return rmsse
