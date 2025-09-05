# Unit test root_mean_squared_scaled_error
# ==============================================================================
from skforecast.metrics import root_mean_squared_scaled_error
import pytest
import numpy as np
import pandas as pd

def test_root_mean_squared_scaled_error_output():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])
    y_train = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    expected_rmsse = np.sqrt(np.mean((y_true - y_pred) ** 2)) / np.sqrt(np.mean(np.diff(y_train) ** 2))
    assert np.isclose(root_mean_squared_scaled_error(y_true, y_pred, y_train), expected_rmsse)

def test_input_types():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])
    y_train = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    with pytest.raises(TypeError):
        root_mean_squared_scaled_error([1, 2, 3], y_pred, y_train)
    with pytest.raises(TypeError):
        root_mean_squared_scaled_error(y_true, [1, 2, 3], y_train)
    with pytest.raises(TypeError):
        root_mean_squared_scaled_error(y_true, y_pred, [1, 2, 3])

def test_input_length():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2])
    y_train = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    with pytest.raises(ValueError):
        root_mean_squared_scaled_error(y_true, y_pred, y_train)
    with pytest.raises(ValueError):
        root_mean_squared_scaled_error(y_true, y_pred, y_train)

def test_empty_input():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])
    y_train = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    with pytest.raises(ValueError):
        root_mean_squared_scaled_error(np.array([]), y_pred, y_train)
    with pytest.raises(ValueError):
        root_mean_squared_scaled_error(y_true, np.array([]), y_train)
    
def test_pandas_series_input():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])
    y_train = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    y_true_series = pd.Series(y_true)
    y_pred_series = pd.Series(y_pred)
    y_train_series = pd.Series(y_train)
    expected_rmsse = np.sqrt(np.mean((y_true - y_pred) ** 2)) / np.sqrt(np.mean(np.diff(y_train) ** 2))
    assert np.isclose(root_mean_squared_scaled_error(y_true_series, y_pred_series, y_train_series), expected_rmsse)

def test_root_mean_squared_scaled_error_with_nan_in_y_train():
    """Test RMSSE with NaN values in y_train - should ignore NaN values."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])
    y_train = np.array([0.5, np.nan, 1.5, 2.5, 3.5, np.nan, 4.5])
    
    # Expected calculation ignoring NaN values in y_train
    y_train_clean = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    expected_rmsse = np.sqrt(np.mean((y_true - y_pred) ** 2)) / np.sqrt(np.mean(np.diff(y_train_clean) ** 2))
    
    result = root_mean_squared_scaled_error(y_true, y_pred, y_train)
    assert np.isclose(result, expected_rmsse)

def test_root_mean_squared_scaled_error_with_nan_in_y_true_y_pred():
    """Test RMSSE with NaN values in y_true and y_pred - should ignore NaN pairs."""
    y_true = np.array([1.0, np.nan, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, np.nan, 3.8])
    y_train = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    
    # Expected calculation ignoring NaN pairs
    y_true_clean = np.array([1.0, 4.0])  # Only positions 0 and 3 are valid
    y_pred_clean = np.array([1.1, 3.8])
    expected_rmsse = np.sqrt(np.mean((y_true_clean - y_pred_clean) ** 2)) / np.sqrt(np.mean(np.diff(y_train) ** 2))
    
    result = root_mean_squared_scaled_error(y_true, y_pred, y_train)
    assert np.isclose(result, expected_rmsse)

def test_root_mean_squared_scaled_error_with_list_y_train_nan():
    """Test RMSSE with list y_train containing NaN values."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])
    y_train = [
        np.array([0.5, np.nan, 1.5, 2.5]),
        np.array([np.nan, 3.5, 4.5, 5.5])
    ]
    
    # Expected calculation with NaN values filtered out
    diffs_1 = np.diff([0.5, 1.5, 2.5])  # Ignore NaN
    diffs_2 = np.diff([3.5, 4.5, 5.5])  # Ignore NaN
    naive_forecast = np.concatenate([diffs_1, diffs_2])
    expected_rmsse = np.sqrt(np.mean((y_true - y_pred) ** 2)) / np.sqrt(np.mean(naive_forecast ** 2))
    
    result = root_mean_squared_scaled_error(y_true, y_pred, y_train)
    assert np.isclose(result, expected_rmsse)

def test_root_mean_squared_scaled_error_all_nan():
    """Test RMSSE when all values are NaN - should return NaN."""
    y_true = np.array([np.nan, np.nan, np.nan, np.nan])
    y_pred = np.array([np.nan, np.nan, np.nan, np.nan])
    y_train = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    
    result = root_mean_squared_scaled_error(y_true, y_pred, y_train)
    assert np.isnan(result)

def test_root_mean_squared_scaled_error_y_train_all_nan():
    """Test RMSSE when y_train is all NaN - should return NaN."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])
    y_train = np.array([np.nan, np.nan, np.nan, np.nan])
    
    result = root_mean_squared_scaled_error(y_true, y_pred, y_train)
    assert np.isnan(result)