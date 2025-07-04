# src/FIT_python/scaler_utils.py

from sklearn.preprocessing import StandardScaler, RobustScaler
from typing import Any, Dict

def get_standard_scaler(params: Dict[str, Any] = None) -> StandardScaler:
    """
    Returns a StandardScaler instance with given parameters.
    """
    params = params or {}
    return StandardScaler(**params)

def get_robust_scaler(params: Dict[str, Any] = None) -> RobustScaler:
    """
    Returns a RobustScaler instance with given parameters.
    """
    params = params or {}
    return RobustScaler(**params)
