"""
Data handling tools
"""

from typing import Union, Dict
from pathlib import Path
import pandas as pd
import numpy as np


# ===================================================================== #
# Dataframes and Series
# ===================================================================== #

def summary_df(obj) -> pd.DataFrame:
    """
    Summarize a DataFrame or Series.
    :param obj: pd.Series or pd.DataFrame
        A pandas DataFrame or Series.
    :returns: pd.DataFrame
        Summary DataFrame with metrics per column (or single row for Series).
    """
    # Converting to df
    if isinstance(obj, pd.Series):
        df = obj.to_frame()
    else:
        df = obj.copy()
    # Getting shape
    n_rows = df.shape[0]
    res = []
    # Getting dtypes
    for col in df.columns:
        ser = df[col]
        is_num = pd.api.types.is_numeric_dtype(ser)
        # basic counts
        n_missing = ser.isna().sum()
        n_inf = np.isinf(ser).sum()
        n_unique = ser.nunique(dropna=False)
        mem = ser.memory_usage(deep=True)
        # percentages
        pct_missing = n_missing / n_rows * 100
        pct_inf = n_inf / n_rows * 100
        row = {
            'dtype': ser.dtype,
            'n_missing': int(n_missing),
            'pct_missing': str(round(pct_missing, 2)) + '%',
            'n_inf': int(n_inf),
            'pct_inf': str(round(pct_inf, 2)) + '%',
            'n_unique': int(n_unique),
            'memory_usage': mem,
        }
        # numeric stats
        if is_num:
            desc = ser.replace([np.inf, -np.inf], np.nan).describe()
            row.update({
                'mean': desc['mean'],
                'std': desc['std'],
                'min': desc['min'],
                '25%': desc['25%'],
                '50%': desc['50%'],
                '75%': desc['75%'],
                'max': desc['max'],
            })
        res.append(row)
    summary = pd.DataFrame(res, index=df.columns)
    stats = ['mean', 'std', 'min', '25%', '50%', '75%', 'max']
    present = [c for c in stats if c in summary.columns]
    summary[present] = summary[present].round(2)
    # add shape for DataFrame (once)
    if isinstance(obj, pd.DataFrame):
        summary.attrs['shape'] = df.shape
    else:
        summary.attrs['shape'] = (n_rows,)
    return summary


def ffill_until_last_valid(
    df: Union[pd.Series, pd.DataFrame]
) -> Union[pd.Series, pd.DataFrame]:
    """
    Forward-fill values up to each column's last non-NaN entry, leaving any rows
    after the final valid value as NaN. Works for both Series and DataFrame.
    :param df: pd.Series or pd.DataFrame 
        Object to ffill (if df, operation is done column-wise)
    :return out_df: pd.Series or pd.DataFrame
        df or series ffilled
    """
    # Handle the Series case by converting it to a 1-column df and then back
    if isinstance(df, pd.Series):
        # Find last valid index in the series
        last_valid = df.last_valid_index()
        if last_valid is None:
            # Entire series is NaN; nothing to fill
            return df.copy()
        # Copy to avoid modifying original
        out = df.copy()
        # Forward-fill only up to last_valid (inclusive)
        out.loc[:last_valid] = df.loc[:last_valid].ffill()
        return out
    # DataFrame case (create a copy so we don’t modify the original)
    out_df = df.copy()
    # Iterate over each column independently
    for col in out_df.columns:
        series = out_df[col]
        last_valid = series.last_valid_index()
        if last_valid is None:
            # If the entire column is NaN, skip
            continue
        # Ffill only up to the last valid index (inclusive)
        filled = series.loc[:last_valid].ffill()
        out_df.loc[:last_valid, col] = filled
    return out_df


# ===================================================================== #
# Parquet
# ===================================================================== #

def save_dict_to_parquet(dfs: Dict[str, pd.DataFrame], folder: str | Path) -> None:
    """
    Saves each DataFrame in *dfs* to <folder>/<key>.parquet.
    :param dfs: dict containing pd.DataFrames
    :param folder: Path or str referring to the folder I want to target
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    # For each df save it to a different file
    for key, df in dfs.items():
        df.to_parquet(
            folder / f"{key}.parquet",
            index=True,
            engine="pyarrow"
        )


def load_dict_from_parquet(folder: str | Path) -> Dict[str, pd.DataFrame]:
    """
    Return {stem: DataFrame} for every *.parquet file in *folder*.
    :param folder: Path or str referring to the folder I want to target
    """
    folder = Path(folder)
    return {
        p.stem: pd.read_parquet(p)
        for p in folder.glob("*.parquet")
    }