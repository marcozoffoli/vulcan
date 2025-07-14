"""
Time series analysis tools
"""

from typing import Literal, Optional, Tuple, Sequence, Iterable, Dict, List
from scipy import signal
from scipy.stats import norm
from scipy.cluster.hierarchy import linkage, leaves_list
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from matplotlib.figure import Figure
from pprint import pprint
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math


# ===================================================================== #
# Correlation
# ===================================================================== #

def correlation_heatmap(
    df: pd.DataFrame,
    method: Literal["pearson", "kendall", "spearman"] = "pearson",
    ewma: bool = False,
    com: float = 5.0,
    mask_limits: Optional[Tuple[float, float]] = None,
    cluster: bool = False,
    figsize: Tuple[int, int] = (10, 8)
) -> Axes:
    """
    Plot a correlation heatmap with optional EWMA, masking, and clustering.
    :param df: pd.DataFrame
        DataFrame with numeric columns to correlate.
    :param method: str
        Correlation method for direct corr (pearson, kendall, spearman).
    :param ewma: bool
        If True, use exponentially weighted correlation (only pearson).
    :param com: int
        Center of mass for EWMA (if ewma=True).
    :param mask_limits: (float, float)
        Tuple (low, high); values between low and high are masked.
    :param cluster: bool
        If True, reorder variables by hierarchical clustering.
    :param figsize: (float, float)
        Figure size (width, height).
    :returns: Matplotlib Axes with the heatmap.
    """
    cols = df.columns.tolist()
    n = len(cols)
    mat = np.zeros((n, n))
    # Compute correlation matrix
    for i, ci in enumerate(cols):
        for j, cj in enumerate(cols):
            if ewma:
                # EWMA correlation via cov/var
                ser_i = df[ci]
                ser_j = df[cj]
                cov = ser_i.ewm(com=com).cov(ser_j).iloc[-1]
                var_i = ser_i.ewm(com=com).var().iloc[-1]
                var_j = ser_j.ewm(com=com).var().iloc[-1]
                val = cov / np.sqrt(var_i * var_j)
            else:
                val = df[ci].corr(df[cj], method=method)
            mat[i, j] = val if val is not None else np.nan
    # Mask values between limits
    mask = None
    if mask_limits is not None:
        low, high = mask_limits
        mask = (mat >= low) & (mat <= high)
    # Clustering reorder
    order = np.arange(n)
    if cluster:
        # distance = 1 - abs(corr)
        dist = 1 - np.abs(mat)
        link = linkage(dist, method="average")
        order = leaves_list(link)
        mat = mat[order][:, order]
        cols = [cols[i] for i in order]
        if mask is not None:
            mask = mask[order][:, order]
    # Plot heatmap
    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.get_cmap("RdBu_r")
    im = ax.imshow(mat, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(cols, rotation=90, fontsize=8)
    ax.set_yticklabels(cols, fontsize=8)
    # Grey out masked cells
    if mask is not None:
        # overlay grey rectangles
        for i in range(n):
            for j in range(n):
                if mask[i, j]:
                    ax.add_patch(
                        Rectangle(
                            (j-0.5, i-0.5), 1, 1, color='lightgrey', zorder=2)
                    )
    # Annotate correlation values
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{mat[i, j]:.2f}",
                    ha="center", va="center", fontsize=7,
                    color="black", zorder=3)
    fig.colorbar(
        im, ax=ax, fraction=0.046, pad=0.04, label="Correlation")
    ax.set_title("Correlation Heatmap", fontsize=12)
    fig.tight_layout()
    return ax


def lead_lag_analysis(
    x: pd.Series,
    y: pd.Series,
    max_lag: int = 20,
    method: Literal["pearson", "kendall", "spearman"] = "pearson",
    use_fft: Optional[bool] = None,
    granger: bool = False,
    verbose: bool = True,
    plot: bool = True,
    ax: Optional[Axes] = None
) -> pd.DataFrame:
    """
    Compute cross-correlation between x and y over a range of lags.
    This function can optionally use FFT for speed, perform Granger
    causality tests, and plot the lag-correlation.
    :param x: pd.Series
        First time series.
    :param y: pd.Series
        Second time series.
    :param max_lag: int
        Maximum lag (positive means x leads y).
    :param method: str
        Correlation method (pearson, kendall, spearman).
    :param use_fft: bool
        Use FFT if True; defaults to True for large n.
    :param granger: bool
        If True, perform Granger causality test.
    :param verbose: bool
        Print best lag and Granger results.
    :param plot: bool
        Plot bar chart of correlations.
    :param ax: Axes
        Axes to plot into. Creates new if None.
    :returns: DataFrame with 'corr' and 'pvalue', index as lag.
    :raises ValueError: If series differ in length or contain NaNs.
    """
    if len(x) != len(y):
        raise ValueError("Series must have the same length.")
    if x.isna().any() or y.isna().any():
        raise ValueError("Drop/forward-fill NaNs before analysis.")
    n = len(x)
    if use_fft is None:
        use_fft = n > 20_000
    x_np = x.to_numpy()
    y_np = y.to_numpy()
    x_vals = x_np - x_np.mean()
    y_vals = y_np - y_np.mean()
    # FFT options
    if use_fft:
        ccf = signal.fftconvolve(x_vals, y_vals[::-1], mode="full")
        half = n - 1
        ccf = ccf[half - max_lag: half + max_lag + 1]
        denom = np.sqrt(np.dot(x_vals, x_vals) * np.dot(y_vals, y_vals))
        corr_vals = ccf / denom
    else:
        corr_vals = [
            x.corr(y.shift(-lag), method=method)
            for lag in range(-max_lag, max_lag + 1)
        ]
        corr_vals = np.array(corr_vals, dtype=float)
    # Lags
    lags = np.arange(-max_lag, max_lag + 1)
    z = 0.5 * np.log((1 + corr_vals) / (1 - corr_vals))
    se = 1 / np.sqrt(n - np.abs(lags) - 3)
    pvals = 2 * (1 - norm.cdf(np.abs(z) / se))
    # Arranging values
    df = pd.DataFrame({"corr": corr_vals, "pvalue": pvals}, index=lags)
    best = int(df["corr"].abs().idxmax())
    df.attrs["best_lag"] = best
    # Granger test
    if granger:
        # Test whether x Granger-causes y
        data_xy = pd.DataFrame({"y": y, "x": x}).dropna()
        data_yx = pd.DataFrame({"x": x, "y": y}).dropna()
        res_xy = grangercausalitytests(data_xy, maxlag=max_lag, verbose=False)
        res_yx = grangercausalitytests(data_yx, maxlag=max_lag, verbose=False)
        p_xy = res_xy[max_lag][0]["ssr_ftest"][1]
        p_yx = res_yx[max_lag][0]["ssr_ftest"][1]
        df.attrs["granger"] = {
            "x_causes_y": p_xy,
            "y_causes_x": p_yx
        }
    if verbose:
        direction = (
            "x leads y" if best > 0
            else "y leads x" if best < 0
            else "simultaneous"
        )
        print(
            f"Peak correlation {df['corr'].loc[best]:.3f} "
            f"at lag {best} ({direction})"
        )
        if granger:
            print("Granger p-values:", df.attrs["granger"])
    # Function to plot the chart
    def _plot(
        df: pd.DataFrame,
        highlight: int,
        x_name: Optional[str],
        y_name: Optional[str],
        ax: Optional[Axes]
    ) -> Axes:
        """
        Plot lag-correlation bar chart with highlighted peak.
        :param df: pd.DataFrame
            DataFrame from lead_lag_analysis.
        :param highlight: int
            Lag to highlight.
        :param x_name: str
            Name of x series.
        :param y_name: str
            Name of y series.
        :param ax: Axes
            Axes to plot into.
        :returns: The Axes instance.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4))
        # Bar chart
        ax.bar(df.index, df["corr"], width=0.8, alpha=0.7)
        ax.axhline(0, linewidth=0.8, linestyle="--")
        ax.bar(
            highlight, 
            df["corr"].loc[highlight],
            width=0.8, 
            alpha=0.9,
            label=f"Peak lag: {highlight}"
        )
        ax.set_xlabel("Lag (x positive \u2192 x leads y)")
        ax.set_ylabel("Correlation")
        ax.set_title(
            f"Lead–Lag Cross-Correlation"
            f" (x={x_name}, y={y_name})"
        )
        ax.legend(loc="upper right", frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.margins(x=0.02)
        return ax
    # Plotting if necessary
    if plot:
        x_name = x.name if isinstance(x.name, str) else None
        y_name = y.name if isinstance(y.name, str) else None
        ax = _plot(df, best, x_name, y_name, ax)
    return df


def plot_acf_pacf(
    series: pd.Series,
    lags: int = 40,
    figsize: Tuple[int, int] = (12, 5),
    title: Optional[str] = None,
    acf_kwargs: Optional[dict] = None,
    pacf_kwargs: Optional[dict] = None,
) -> Tuple[Axes, Axes]:
    """
    Plot ACF and PACF of a time series in a two-panel chart.
    :param series: pd.Series
        Time series to analyze.
    :param lags: int      
        Number of lags to include.
    :param figsize: (float, float)   
        Figure size (width, height in inches).
    :param title: str     
        Overall figure title. Defaults to None.
    :param acf_kwargs: 
        Additional kwargs passed to statsmodels.plot_acf.
    :param pacf_kwargs:
        Additional kwargs passed to statsmodels.plot_pacf.
    :returns: Tuple of (ax_acf, ax_pacf) matplotlib Axes.
    """
    acf_kwargs = acf_kwargs or {}
    pacf_kwargs = pacf_kwargs or {}
    # Set a clean aesthetic
    sns.set_style("darkgrid")
    # plt.style.use("whitegrid")
    fig, (ax_acf, ax_pacf) = plt.subplots(
        1, 2, figsize=figsize, constrained_layout=True
    )
    if title:
        fig.suptitle(title, fontsize=14, weight="bold", y=1.02)
    # Plot ACF
    plot_acf(
        series.dropna(),
        lags=lags,
        ax=ax_acf,
        zero=False,
        **acf_kwargs
    )
    ax_acf.set_title("Autocorrelation (ACF)", fontsize=12, weight="semibold")
    ax_acf.set_xlabel("Lag")
    ax_acf.set_ylabel("Correlation")
    # Plot PACF
    plot_pacf(
        series.dropna(),
        lags=lags,
        ax=ax_pacf,
        method="ywm",
        zero=False,
        **pacf_kwargs
    )
    ax_pacf.set_title(
        "Partial Autocorrelation (PACF)", 
        fontsize=12,
        weight="semibold"
    )
    ax_pacf.set_xlabel("Lag")
    ax_pacf.set_ylabel("Partial Corr")
    return ax_acf, ax_pacf


# ===================================================================== #
# Dynamic Time Warping
# ===================================================================== #

def dtw_analysis(
    s1: pd.Series,
    s2: pd.Series,
    *,
    normalize: bool = True,
    derivative: bool = False,
    window: int | None = None,
    metric: str = "euclidean",
    show: bool = True,
) -> dict[str, float | int | list | np.ndarray]:
    """
    Dynamic-Time-Warp two 1-d pandas Series and auto-report.
    :param s1, s2: pd.Series
        Anything numeric (price, returns …).  Missing values are dropped.
    :param normalize: bool, default True
        Z-score each series before DTW.
    :param derivative: bool, default False
        Use first diffs (good for return curves).
    :param window: int or None
        Sakoe-Chiba radius |i-j| ≤ window. None = unbounded (slow!).
    :param metric: {'euclidean', 'absolute', 'square'}
        Point cost definition.
    :param show: bool, default True
        If True, print table + show charts immediately.
    :return dict:
        distance      : float
        median_slope  : float
        path_length   : int
        path          : list[(i, j)]
        cost_matrix   : 2-d np.ndarray  (n × m)
    """
    # Input hygene
    a = s1.dropna().to_numpy(dtype=float)
    b = s2.dropna().to_numpy(dtype=float)
    # Using differences
    if derivative:
        a = np.diff(a, prepend=a[0])
        b = np.diff(b, prepend=b[0])
    # Normalization
    if normalize:
        a = (a - a.mean()) / a.std(ddof=0)
        b = (b - b.mean()) / b.std(ddof=0)
    # Key dimensions
    n, m = len(a), len(b)
    w = window if window is not None else max(n, m)
    inf = math.inf
    # Dynamically adjusted matrix
    D = np.full((n + 1, m + 1), inf)
    D[0, 0] = 0.0
    # Cost function
    def cost(x: float, y: float) -> float:
        if metric == "absolute":
            return abs(x - y)
        if metric == "square":
            return (x - y) ** 2
        return abs(x - y)  # euclidean
    # DP
    for i in range(1, n + 1):
        j_lo = max(1, i - w)
        j_hi = min(m, i + w)
        for j in range(j_lo, j_hi + 1):
            d = cost(a[i - 1], b[j - 1])
            D[i, j] = d + min(
                D[i - 1, j],        # insertion
                D[i, j - 1],        # deletion
                D[i - 1, j - 1],    # match
            )
    distance = float(D[n, m])
    # Back Track
    path: list[tuple[int, int]] = []
    i, j = n, m
    while (i, j) != (0, 0):
        path.append((i - 1, j - 1))
        moves = [D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]]
        move = int(np.argmin(moves))
        if move == 0:
            i, j = i - 1, j - 1
        elif move == 1:
            i -= 1
        else:
            j -= 1
    path.reverse()
    dx = np.diff([p[0] for p in path])
    dy = np.diff([p[1] for p in path])
    slopes = np.divide(dy, dx, out=np.zeros_like(dy, dtype=float), where=dx != 0)
    median_slope = float(np.median(slopes[np.isfinite(slopes)]))
    # Result
    result = {
        "distance": distance,
        "median_slope": median_slope,
        "path_length": len(path),
        "path": path,
        "cost_matrix": D[1:, 1:],  # strip padding row/col
    }
    # Auto Report
    if show:
        summary = {
            "DTW Distance": round(distance, 2),
            "Path Length": len(path),
            "DTW Distance over Path Length": round(distance / len(path), 2),
            "Path Length over Max Length": round(len(path) / max(n,m), 2),
            "Median Slope": round(median_slope, 2),
        }
        pprint(summary)
        # Heat Map
        fig1, ax1 = plt.subplots()
        im = ax1.imshow(result["cost_matrix"], origin="lower", aspect="auto")
        ax1.plot(
            [p[0] for p in path],
            [p[1] for p in path],
            label="Optimal path",
            color="white",
            lw=1.5,
        )
        ax1.set(
            title="Cumulative-cost matrix (DTW)",
            xlabel=s1.name or "Series 1 index",
            ylabel=s2.name or "Series 2 index",
        )
        fig1.colorbar(im, ax=ax1, label="Cumulative cost")
        ax1.legend(loc="upper left")
        fig1.tight_layout()
        # Aligned series chart
        fig2, ax2 = plt.subplots()
        t_a = np.arange(n)
        t_b = np.linspace(0, n - 1, m)
        # Warped B
        aligned_b = np.full(n, np.nan)
        counts = np.zeros(n)
        for i_, j_ in path:
            aligned_b[i_] = (
                aligned_b[i_] if not np.isnan(aligned_b[i_]) else 0.0
            ) + b[j_]
            counts[i_] += 1
        aligned_b /= counts
        ax2.plot(t_a, a, label=s1.name or "Series 1")
        ax2.plot(t_a, aligned_b, label=f"{s2.name or 'Series 2'} (warped)")
        # Raw B
        ax2.plot(
            t_b,
            b,
            "--",
            alpha=0.4,
            label=f"{s2.name or 'Series 2'} (raw)",
        )
        ax2.set(
            title="Series alignment after DTW",
            xlabel="Index on Series A’s clock",
            ylabel="Normalised level",
        )
        ax2.legend()
        fig2.tight_layout()
        plt.show()
    return result


# ===================================================================== #
# Moments
# ===================================================================== #

def plot_rolling_moments(
    series: pd.Series,
    periods: Sequence[int] = (20,),
    ewm: bool = False,
    figsize: Tuple[int, int] = (12, 10)
) -> None:
    """
    Plot rolling moments (mean, variance, skewness, kurtosis) in a 2×2 grid.
    :param series: pd.Series
        Time series to analyze.
    :param periods: Sequence[int]
        1–3 window sizes (rolling) or COM values (if ewm is True).
    :param ewm: bool
        If True, use exponentially weighted moments.
    :param colors: Sequence[str], optional
        Colors for each window. Defaults to matplotlib cycle.
    :param figsize: Tuple[int, int]
        Figure size as (width, height).
    :returns: Tuple[Figure, Axes]
        Figure and flattened array of Axes.
    :raises ValueError:
        If number of periods is not between 1 and 3 or colors mismatch.
    """
    # Validate inputs
    if not 1 <= len(periods) <= 3:
        raise ValueError('Choose between 1 and 3 window sizes')
    # Setting colors
    default_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors = default_cycle[:len(periods)]
    # Prepare data: compute each moment per window
    metrics = ['Mean', 'Variance', 'Skewness', 'Kurtosis']
    moments: Dict[int, List[pd.Series]] = {}
    # Computing moments
    for p in periods:
        if ewm:
            m0 = series.ewm(com=p).mean()
            m1 = series.ewm(com=p).var()
            E2 = series.pow(2).ewm(com=p).mean()
            E3 = series.pow(3).ewm(com=p).mean()
            E4 = series.pow(4).ewm(com=p).mean()
            cm3 = E3 - 3 * m0 * E2 + 2 * m0.pow(3)
            m2 = cm3.div(m1.pow(1.5))
            cm4 = (E4 - 4 * m0 * E3 + 6 * m0.pow(2) * E2
                   - 3 * m0.pow(4))
            m3 = cm4.div(m1.pow(2))
        else:
            m0 = series.rolling(p).mean()
            m1 = series.rolling(p).var()
            m2 = series.rolling(p).skew()
            m3 = series.rolling(p).kurt()
        moments[p] = [m0, m1, m2, m3]
    # Build 2×2 subplots
    fig, axs = plt.subplots(2, 2, figsize=figsize, sharex=True)
    axs = axs.flatten()
    for idx, ax in enumerate(axs):
        ax.set_title(metrics[idx], fontsize=13, weight='bold')
        ax.set_ylabel(metrics[idx])
        # plot each window with its color
        for p, color in zip(periods, colors):
            y = moments[p][idx]
            ax.plot(
                y.index,
                y.values,
                color=color,
                linewidth=1,
                label=f"com={p}" if ewm else f"win={p}"
            )
            ax.fill_between(y.index, y.values, alpha=0.1, color=color)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        if idx == 0:
            legend_title = 'COM' if ewm else 'Window'
            ax.legend(title=legend_title, fontsize=9, loc='upper left')
    # shared x-label and layout
    for ax in axs[2:]:
        ax.set_xlabel('Index')
    fig.suptitle('Rolling Moments', fontsize=16, weight='bold')
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))