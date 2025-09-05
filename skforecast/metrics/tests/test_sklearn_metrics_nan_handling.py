# Unit test NaN handling in scikit-learn metrics wrapped by add_y_train_argument
# ==============================================================================
import pytest
import numpy as np
import pandas as pd
from skforecast.metrics import _get_metric
from sklearn.metrics import mean_squared_error, mean_absolute_error


def test_sklearn_metrics_nan_handling_mean_squared_error():
    """Test that wrapped MSE ignores NaN values."""
    metric = _get_metric('mean_squared_error')
    
    # Test with NaN values in y_true and y_pred
    y_true = np.array([1.0, np.nan, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, np.nan, 3.8])
    
    # Expected calculation ignoring NaN pairs
    y_true_clean = np.array([1.0, 4.0])  # Only positions 0 and 3 are valid
    y_pred_clean = np.array([1.1, 3.8])
    expected_mse = mean_squared_error(y_true_clean, y_pred_clean)
    
    result = metric(y_true=y_true, y_pred=y_pred)
    assert np.isclose(result, expected_mse)


def test_sklearn_metrics_nan_handling_mean_absolute_error():
    """Test that wrapped MAE ignores NaN values."""
    metric = _get_metric('mean_absolute_error')
    
    # Test with NaN values in y_true and y_pred
    y_true = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 1.9, np.nan, 3.8, np.nan])
    
    # Expected calculation ignoring NaN pairs
    y_true_clean = np.array([1.0, 4.0])  # Only positions 0 and 3 are valid
    y_pred_clean = np.array([1.1, 3.8])
    expected_mae = mean_absolute_error(y_true_clean, y_pred_clean)
    
    result = metric(y_true=y_true, y_pred=y_pred)
    assert np.isclose(result, expected_mae)


def test_sklearn_metrics_all_nan_values():
    """Test that wrapped metrics return NaN when all values are NaN."""
    metric = _get_metric('mean_squared_error')
    
    y_true = np.array([np.nan, np.nan, np.nan, np.nan])
    y_pred = np.array([np.nan, np.nan, np.nan, np.nan])
    
    result = metric(y_true=y_true, y_pred=y_pred)
    assert np.isnan(result)


def test_sklearn_metrics_pandas_series_with_nan():
    """Test that wrapped metrics work with pandas Series containing NaN."""
    metric = _get_metric('mean_absolute_error')
    
    y_true = pd.Series([1.0, np.nan, 3.0, 4.0])
    y_pred = pd.Series([1.1, 1.9, np.nan, 3.8])
    
    # Expected calculation ignoring NaN pairs
    expected_mae = mean_absolute_error([1.0, 4.0], [1.1, 3.8])
    
    result = metric(y_true=y_true, y_pred=y_pred)
    assert np.isclose(result, expected_mae)


def test_sklearn_metrics_no_nan_values():
    """Test that wrapped metrics work correctly with no NaN values (baseline test)."""
    metric = _get_metric('mean_squared_error')
    
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])
    
    expected_mse = mean_squared_error(y_true, y_pred)
    result = metric(y_true=y_true, y_pred=y_pred)
    
    assert np.isclose(result, expected_mse)


def test_sklearn_metrics_mean_absolute_percentage_error_with_nan():
    """Test MAPE with NaN handling."""
    metric = _get_metric('mean_absolute_percentage_error')
    
    y_true = np.array([1.0, np.nan, 3.0, 4.0])
    y_pred = np.array([1.1, 2.0, np.nan, 3.8])
    
    # Expected calculation ignoring NaN pairs
    from sklearn.metrics import mean_absolute_percentage_error
    expected_mape = mean_absolute_percentage_error([1.0, 4.0], [1.1, 3.8])
    
    result = metric(y_true=y_true, y_pred=y_pred)
    assert np.isclose(result, expected_mape)