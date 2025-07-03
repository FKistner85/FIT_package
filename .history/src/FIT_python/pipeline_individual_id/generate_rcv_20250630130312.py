# generate_rcv.py

try:
    import pandas as pd
    _pd_err = None
except Exception as exc:
    pd = None
    _pd_err = exc

def generate_rcv(train_df: pd.DataFrame) -> pd.DataFrame:
    """Create the **R**ecaptured **C**ontrol **V**ariation dataset.

    Used by :func:`fit_otter.models.pairwise_analysis.process_pair` when
    projecting test trails together with an artificial reference class.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training rows containing ``'individual'`` and ``'Trail'`` columns.

    Returns
    -------
    pd.DataFrame
        Copy of ``train_df`` with ``'individual'`` and ``'Trail'`` set to ``"RCV"``.
    """
    df_rcv = train_df.copy()
    df_rcv['individual'] = 'RCV'
    df_rcv['Trail'] = 'RCV'
    return df_rcv


if __name__ == "__main__":
    if pd is None:
        print("pandas not installed, skipping self-test")
    else:
        sample = pd.DataFrame({"individual": ["A", "B"], "Trail": ["t1", "t2"]})
        res = generate_rcv(sample)
        print(res)
