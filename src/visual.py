"""Visualization tools"""

from typing import Literal, Optional, Tuple
from matplotlib.figure import Figure
from scipy import signal
from scipy.stats import gaussian_kde, norm, skew, kurtosis
from scipy.cluster.hierarchy import linkage, leaves_list
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.nonparametric.smoothers_lowess import lowess
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ===================================================================== #
# Histogram
# ===================================================================== #

def plot_distribution(series, bins=30, figsize=(10, 6)):
    """
    Plot a detailed distribution histogram with KDE and normal comparison.

    Marks mean, median, ±1σ, ±2σ, 5th and 95th percentiles.
    Annotates skewness and kurtosis.
    """
    data = np.asarray(series)
    mu = data.mean()
    med = np.median(data)
    sigma = data.std(ddof=1)
    # Percentiles
    p5, p95 = np.percentile(data, [5, 95])
    # Histogram and density
    fig, ax = plt.subplots(figsize=figsize)
    counts, bins, patches = ax.hist(
        data, bins=bins, density=True, color='lightgray', edgecolor='black'
    )
    # KDE
    kde = gaussian_kde(data)
    x_vals = np.linspace(bins[0], bins[-1], 1000)
    ax.plot(x_vals, kde(x_vals), linestyle='-', linewidth=2, label='KDE')
    # Normal distribution
    ax.plot(
        x_vals, norm.pdf(x_vals, mu, sigma),
        linestyle='--', linewidth=2, label='Normal PDF'
    )
    # Vertical lines and annotations
    vlines = {
        'Mean': mu, 'Median': med,
        '+1σ': mu + sigma, '-1σ': mu - sigma,
        '+2σ': mu + 2*sigma, '-2σ': mu - 2*sigma,
        '5th pct': p5, '95th pct': p95
    }
    style_map = {
        'Mean': ('black', '-'),
        'Median': ('black', '--'),
        '+1σ': ('red', ':'),
        '-1σ': ('red', ':'),
        '+2σ': ('magenta', '-.'),
        '-2σ': ('magenta', '-.'),
        '5th pct': ('cyan', '--'),
        '95th pct': ('cyan', '--')
    }
    for label, xpos in vlines.items():
        color, ls = style_map[label]
        ax.axvline(
            xpos, color=color, linestyle=ls,
            linewidth=1.5, label=label
        )
    # Text box with skewness and kurtosis
    stats_text = (
        f"Median: {med:.3f}\n"
        f"Mean: {mu:.3f}\n"
        f"Std Dev: {sigma:.3f}\n"
        f"Skewness: {skew(data):.3f}\n"
        f"Kurtosis: {kurtosis(data, fisher=False):.3f}"
    )
    ax.text(
        0.97, 0.95, stats_text, transform=ax.transAxes,
        verticalalignment='top', 
        horizontalalignment='right',
        bbox=dict(
            boxstyle='round,pad=0.3', 
            facecolor='white',
            edgecolor='gray'
        )
    )
    # Legend and labels
    ax.legend(loc='upper left', fontsize='small')
    ax.set_title('Distribution')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    plt.tight_layout()
    plt.show()


# ===================================================================== #
# Scatter
# ===================================================================== #

def scatter_plot(
    x: pd.Series,
    y: pd.Series,
    kind: Literal['scatter', 'hex', 'density'] = 'hex',
    bins: int = 50,
    cmap: str = 'viridis',
    fit_line: Literal[
        'none', 'ols', 'through_origin', 'loess', 'all'
    ] = 'ols',
    loess_frac: float = 0.3,
    ax: Optional[Axes] = None
) -> Axes:
    """
    Scatter two series with density visualization and fit lines.
    :param x: pd.Series
        Series for x-axis.
    :param y: pd.Series
        Series for y-axis.
    :param kind: 'scatter', 'hex', or 'density' via KDE coloring.
    :param bins: Number of bins or grid size.
    :param cmap: Colormap for density/hex.
    :param fit_line: str
        Which fit to draw ('none','ols','through_origin','loess','all').
    :param loess_frac: Fraction for LOESS smoothing.
    :param ax: Axes to plot into.
    :returns: Matplotlib Axes.
    """
    x_vals = x.to_numpy()
    y_vals = y.to_numpy()
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    if kind == 'hex':
        hb = ax.hexbin(
            x_vals, y_vals, gridsize=bins, cmap=cmap, mincnt=1
        )
        cbar = ax.figure.colorbar(hb, ax=ax)
        cbar.set_label('Counts')
    elif kind == 'density':
        data = np.vstack([x_vals, y_vals])
        dens = gaussian_kde(data)(data)
        idx = dens.argsort()
        sc = ax.scatter(
            x_vals[idx], y_vals[idx], c=dens[idx],
            cmap=cmap, s=10
        )
        plt.colorbar(sc, ax=ax, label='Density')
    else:
        ax.scatter(x_vals, y_vals, alpha=0.6, s=10)
    # Fit lines
    xlim = ax.get_xlim()
    def plot_line(slope, intercept, label) -> None:
        xs = np.array(xlim)
        ys = slope * xs + intercept
        ax.plot(xs, ys, label=label, lw=1.5)
    if fit_line in ('ols', 'all'):
        m, b = np.polyfit(x_vals, y_vals, 1)
        plot_line(m, b, 'OLS')
    if fit_line in ('through_origin', 'all'):
        m0 = np.dot(x_vals, y_vals) / np.dot(x_vals, x_vals)
        plot_line(m0, 0, 'Through Origin')
    if fit_line in ('loess', 'all'):
        lo = lowess(y_vals, x_vals, frac=loess_frac)
        ax.plot(lo[:, 0], lo[:, 1], label='LOESS', lw=1.5)
    if fit_line != 'none':
        ax.legend(loc='best', fontsize=8)
    xlabel = str(x.name) if x.name is not None else 'x'
    ylabel = str(y.name) if y.name is not None else 'y'
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Scatter of '{ylabel}' against '{xlabel}'")
    plt.tight_layout()
    return ax