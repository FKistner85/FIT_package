# generate_rcv.py

try:
    import pandas as pd

    _pd_err = None
except Exception as exc:
    pd = None
    _pd_err = exc

import pandas as pd


def generate_rcv(full_df: pd.DataFrame, exclude_ids: list) -> pd.DataFrame:
    """Create the **R**ecaptured **C**ontrol **V**ariation dataset.

    Used to project comparison samples into a reference space created from all other data.

    Parameters
    ----------
    full_df : pd.DataFrame
        Complete dataset containing all samples.
    exclude_ids : list
        List of ``id`` values that should be excluded (e.g. used in the
        current comparison).

    Returns
    -------
    pd.DataFrame
        Subset of ``full_df`` excluding ``exclude_ids`` and with
        ``'individual_id'`` and ``'Trail'`` set to ``"RCV"``.
    """
    full_df = full_df.copy()
    full_df["id"] = full_df["id"].astype(str)
    exclude_ids = [str(i) for i in exclude_ids]
    df_rcv = full_df.set_index("id").drop(exclude_ids).copy()
    df_rcv = df_rcv.reset_index()
    df_rcv["individual_id"] = "RCV"
    if "Trail" in df_rcv.columns:
        df_rcv["Trail"] = "RCV"
    return df_rcv


if __name__ == "__main__":
    sample = pd.DataFrame(
        {"individual_id": ["A", "B"], "Trail": ["t1", "t2"]}, index=[0, 1]
    )
    full = pd.concat([sample] * 5, ignore_index=True)
    res = generate_rcv(full, exclude_ids=["0", "1"])
    print(res.head())
