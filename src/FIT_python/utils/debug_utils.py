import numpy as np
import pandas as pd

import FIT_python.config as config


def debug_report(df_or_arr, step_name: str) -> None:
    """Print debug statistics if ``config.DEBUG_MODE`` is True."""
    if not config.DEBUG_MODE:
        return

    if isinstance(df_or_arr, pd.DataFrame):
        arr = df_or_arr.to_numpy()
        shape = df_or_arr.shape
    else:
        arr = np.asarray(df_or_arr)
        shape = arr.shape

    n_nan = int(np.isnan(arr).sum())
    n_inf = int(np.isinf(arr).sum())
    print(f"[DEBUG] {step_name}: shape={shape}, NaN={n_nan}, Inf={n_inf}")
