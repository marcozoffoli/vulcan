"""
Machine Learning tools
"""

import optuna
import numbers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any, Sequence
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.base import BaseEstimator
from sklearn.model_selection import TimeSeriesSplit
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRFRegressor
from treeinterpreter import treeinterpreter


# ===================================================================== #
# Boosting
# ===================================================================== #

def train_xgb_rf(
    df: pd.DataFrame,
    target: str,
    test_share: float = 0.3,
    n_trials: int = 50,
    n_splits: int = 4,
    random_state: int | None = None,
) -> Tuple[XGBRFRegressor, pd.DataFrame]:
    """
    Fit + tune an XGBoost Random-Forest model on a time-series DataFrame.
    :param df: pandas.DataFrame
        Datetime-indexed frame with predictors and *target* column.
    :param target: str
        Name of the target column to forecast.
    :param test_share: float
        Percentage of total rows (most-recen) held out for final test.
    :param n_trials: int
        Optuna trials for hyper-parameter optimisation.
    :param n_splits: int
        Walk-forward folds used for validation.
    :param random_state: int | None
        Reproducibility seed.
    :return: (xgboost.XGBRFRegressor, pandas.DataFrame)
        Tuple(best_model, performance_df)
        *best_model* contains the scikit-learn estimator object
        *performance_df* lists MAE/RMSE for each validation fold + oos test
    """
    # Basic Checks
    assert df.index.is_monotonic_increasing, "df must be sorted by time"
    assert target in df.columns, f"{target!r} not found in df"
    assert 0 < test_share < 1, "test_size float must be in (0,1)"
    # Getting actual test size
    test_size = int(len(df) * test_share)
    # Assigning ppredictors and target
    y = df[target]
    X = df.drop(columns=[target])
    # Train test split
    X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
    y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]
    # CV Splitter
    tscv = TimeSeriesSplit(
        n_splits=n_splits,
        test_size=max(1, len(X_train) // (n_splits + 1)),
    )
    # Optuna objective
    def objective(trial: optuna.Trial) -> float:
        """Minimise average MAE across CV folds."""
        params: Dict[str, Any] = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.2, log=True
            ),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bynode": trial.suggest_float(
                "colsample_bynode", 0.5, 1.0
            ),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_float(
                "min_child_weight", 1.0, 20.0
            ),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
            "gamma": trial.suggest_float("gamma", 0.0, 2.0),
            "random_state": random_state,
            "n_jobs": -1,
        }
        # Getting errors for each fold
        fold_mae: list[float] = []
        for tr_idx, val_idx in tscv.split(X_train):
            model = XGBRFRegressor(**params)
            model.fit(
                X_train.iloc[tr_idx],
                y_train.iloc[tr_idx],
                eval_set=[(X_train.iloc[val_idx], y_train.iloc[val_idx])],
                verbose=False,
            )
            preds = model.predict(X_train.iloc[val_idx])
            fold_mae.append(mean_absolute_error(y_train.iloc[val_idx], preds))
        return float(np.mean(fold_mae))
    # Hyper-parameter search
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_params = study.best_trial.params | {
        "random_state": random_state,
        "n_jobs": -1,
    }
    # Fit final model on full train
    best_model = XGBRFRegressor(**best_params)
    best_model.fit(X_train, y_train)
    # Gather fold + test metrics
    metrics: list[dict[str, Any]] = []
    for i, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
        model = XGBRFRegressor(**best_params)
        model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        val_pred = model.predict(X_train.iloc[val_idx])
        metrics.append(
            {
                "fold": i,
                "mae": mean_absolute_error(y_train.iloc[val_idx], val_pred),
                "rmse": np.sqrt(
                    mean_squared_error(y_train.iloc[val_idx], val_pred)
                ),
                "rsquared": model.score(
                    X_train.iloc[val_idx], y_train.iloc[val_idx]
                )
            }
        )
    test_pred = best_model.predict(X_test)
    metrics.append(
        {
            "fold": "test",
            "mae": mean_absolute_error(y_test, test_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, test_pred)),
            "rsquared": best_model.score(X_test, y_test)
        }
    )
    return best_model, pd.DataFrame(metrics)


# ===================================================================== #
# Feature Imprtance
# ===================================================================== #

def plot_feature_importance(
    est: BaseEstimator,
    *,
    feature_names: Sequence[str] | None = None,
    X: pd.DataFrame | None = None,
    y: pd.Series | None = None,
    top: int | None = 20,
    n_repeats: int = 30,
    random_state: int | None = None,
    shapley: bool = False,
    ax: Axes | None = None,
) -> None:
    """
    Plot feature importance for a fitted estimator.
    :param est: BaseEstimator
        Fitted scikit-learn estimator.
    :param feature_names: Sequence[str] or None
        Names of input columns. If None, uses ``est.feature_names_in_``.
    :param X: pd.DataFrame or None
        Validation features for permutation or SHAP.
    :param y: pd.Series or None
        Validation targets for permutation importance.
    :param top: int or None
        How many top features to show. None → all.
    :param n_repeats: int
        Repeats for permutation importance.
    :param random_state: int or None
        Seed for permutation importance.
    :param shapley: bool
        If True, compute SHAP value importances (requires X_val).
    :param ax: plt.Axes or None
        Matplotlib Axes to plot on. If None, new fig is made.
    :raises ValueError: 
        If importance can't be inferred or required data not provided.
    """
    # Feature‐name fallback
    if feature_names is None and hasattr(est, "feature_names_in_"):
        feature_names = list(getattr(est, "feature_names_in_"))
    # Function to infer imprtance
    def _infer_importance(
        est: BaseEstimator,
        *,
        X: pd.DataFrame | None,
        y: pd.Series | None,
        n_repeats: int,
        random_state: int | None,
    ) -> np.ndarray:
        """Return 1D array of importances or raise ValueError."""
        if hasattr(est, "feature_importances_"):
            imp = getattr(est, "feature_importances_")
        elif hasattr(est, "coef_"):
            coef = getattr(est, "coef_")
            if coef.ndim > 1:
                # average across multiple outputs
                imp = np.mean(np.abs(coef), axis=0)
            else:
                imp = np.abs(coef).ravel()
        elif X is not None and y is not None:
            res = permutation_importance(
                est,
                X,
                y,
                n_repeats=n_repeats,
                random_state=random_state,
            )
            imp = getattr(res, "importances_mean")
        else:
            raise ValueError(
                "Estimator has no native importance; provide X_val and y_val "
                "for permutation_importance."
            )
        return np.asarray(imp, dtype=float)
    # Compute importances
    if shapley:
        if X is None:
            raise ValueError("X_val is required when shapley=True")
        _, _, contribs = treeinterpreter.predict(est, X.to_numpy())
        contribs = np.asarray(contribs, dtype=float)
        imp = np.mean(np.abs(contribs), axis=0)
    else:
        imp = _infer_importance(
            est=est,
            X=X,
            y=y,
            n_repeats=n_repeats,
            random_state=random_state,
        )
    # Final name handling & sanity checks
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(len(imp))]
    if len(imp) != len(feature_names):
        raise ValueError(
            "Length mismatch: %d importances vs %d feature names"
            % (len(imp), len(feature_names))
        )
    # Build and select top
    s = pd.Series(imp, index=pd.Index(feature_names))
    if isinstance(top, numbers.Integral):
        top = min(top, len(s))
        # pick top-K highest
        idx = np.argsort(-s.to_numpy())[:top]
        s = s.iloc[idx]
    # for barh we want ascending
    s = s.sort_values()
    # Plot
    if ax is None:
        _, ax = plt.subplots(figsize=(8, max(4, len(s) * 0.3)))
    s.plot.barh(ax=ax)
    ax.set_title(
        "SHAP Feature Importance" if shapley else "Feature Importance"
    )
    ax.set_xlabel("Mean |importance|")
    plt.tight_layout()
    plt.show()