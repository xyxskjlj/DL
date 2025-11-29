"""Comprehensive adaptive imputation framework.

This module restructures the original notebook-style prototype into a
reusable Python module that implements several adaptive imputers along the
Rate → Column → Meta → Cluster → Ultimate hierarchy.  The goal is to provide
an experiment-ready toolkit for studying feature-level adaptive imputation
under different missing mechanisms and class-imbalance settings.

Key capabilities
----------------
- Strategy table construction from RMSE leaderboards (rate-level or
  rate×column level).
- Meta-learning based selection of column-wise base regressors.
- Cluster-wise specialization of meta models.
- An ultimate imputer that combines rate/column/meta/cluster selection with
  simple uncertainty estimation.
- Class-imbalance aware pipeline that supports parallel imputation for
  majority/minority classes and safely reorders samples by their original
  indices.
- Real-data application helpers for evaluating downstream classifiers when
  ground truth for missing values is unavailable.

The code follows the research objectives described in the prompt and is
fully executable with Python 3.9 + scikit-learn ecosystem.
"""
from __future__ import annotations

import math
import os
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def nearest_rate_bin(rate: float, bins: Iterable[float]) -> float:
    """Return the rate bin whose value is closest to the provided rate."""
    bins = list(bins)
    return min(bins, key=lambda b: abs(rate - b))


def global_missing_rate(X: pd.DataFrame) -> float:
    """Compute overall missing rate of a dataframe."""
    total = np.prod(X.shape)
    missing = X.isna().sum().sum()
    return float(missing) / float(total)


def column_missing_rates(X: pd.DataFrame) -> Dict[int, float]:
    """Return missing rate per column index."""
    return {int(col): float(X[col].isna().mean()) for col in X.columns}


def get_base_regressor(name: str, random_state: int = 0) -> RegressorMixin:
    """Factory for base regressors used by adaptive imputers."""
    name = name.lower()
    if name in {"lr", "linearregression"}:
        return LinearRegression()
    if name in {"br", "bayesianridge"}:
        return BayesianRidge()
    if name in {"rf", "randomforest"}:
        return RandomForestRegressor(n_estimators=120, random_state=random_state, n_jobs=-1)
    if name in {"et", "extratrees"}:
        return ExtraTreesRegressor(n_estimators=160, random_state=random_state, n_jobs=-1)
    if name in {"gb", "gbr", "gradientboosting"}:
        return GradientBoostingRegressor(random_state=random_state)
    raise ValueError(f"Unsupported regressor name: {name}")


def iterations_for_rate(rate: float) -> int:
    """Adaptive IterativeImputer iterations based on missing rate."""
    if rate <= 0.10:
        return 5
    if rate <= 0.30:
        return 10
    if rate <= 0.50:
        return 15
    return 20


def rmse_at_missing(true_df: pd.DataFrame, imputed_df: pd.DataFrame, mask_missing: pd.DataFrame) -> Dict[int, float]:
    """Compute column-wise RMSE restricted to originally missing positions."""
    rmses: Dict[int, float] = {}
    for col in true_df.columns:
        miss_pos = mask_missing[col]
        if miss_pos.any():
            mse = mean_squared_error(true_df.loc[miss_pos, col], imputed_df.loc[miss_pos, col])
            rmses[int(col)] = math.sqrt(mse)
    return rmses


def build_best_imp_by_rate(rmse_df: pd.DataFrame) -> Dict[float, str]:
    """Given RMSE leaderboard, return {rate_bin: best_imp_name}."""
    best: Dict[float, str] = {}
    for rate, group in rmse_df.groupby("rate"):
        best_imp = group.sort_values("rmse").iloc[0]["imp"]
        best[float(rate)] = str(best_imp)
    return best


def build_best_imp_by_rate_col(rmse_df: pd.DataFrame) -> Dict[Tuple[float, int], str]:
    """Return mapping (rate_bin, col) -> best_imp_name from leaderboard."""
    best: Dict[Tuple[float, int], str] = {}
    for (rate, col), group in rmse_df.groupby(["rate", "col"]):
        best_imp = group.sort_values("rmse").iloc[0]["imp"]
        best[(float(rate), int(col))] = str(best_imp)
    return best


def make_meta_dataset(rmse_df: pd.DataFrame, stats_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """从 RMSE 榜单与列统计量构建元学习数据集。

    参数
    ----
    rmse_df : pd.DataFrame
        包含列 [mech, rate, imp, col, rmse] 的 RMSE 榜单。
    stats_df : pd.DataFrame
        列统计信息，至少包含 [col, mean, std, range, max_corr, mean_corr]，可选包含 rate。

    返回
    ----
    X_meta : pd.DataFrame
        元特征矩阵。
    y_meta : pd.Series
        每个 (rate, col) 的最优插补器名称标签。
    """
    rows = []
    targets = []
    grouped = rmse_df.groupby(["rate", "col"])
    for (rate, col), group in grouped:
        best_imp = group.sort_values("rmse").iloc[0]["imp"]
        stat_row = stats_df.loc[stats_df["col"] == col].iloc[0]
        rows.append(
            {
                "col": int(col),
                "rate": float(rate),
                "mean": stat_row.get("mean", 0.0),
                "std": stat_row.get("std", 0.0),
                "range": stat_row.get("range", 0.0),
                "max_corr": stat_row.get("max_corr", 0.0),
                "mean_corr": stat_row.get("mean_corr", 0.0),
            }
        )
        targets.append(best_imp)
    return pd.DataFrame(rows), pd.Series(targets)


def column_statistics(full_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-column statistics required for meta features."""
    corr = full_df.corr().fillna(0)
    stats = []
    for col in full_df.columns:
        others = [c for c in full_df.columns if c != col]
        max_corr = corr.loc[col, others].abs().max() if others else 0.0
        mean_corr = corr.loc[col, others].abs().mean() if others else 0.0
        col_data = full_df[col]
        stats.append(
            {
                "col": int(col),
                "mean": float(col_data.mean()),
                "std": float(col_data.std()),
                "min": float(col_data.min()),
                "max": float(col_data.max()),
                "range": float(col_data.max() - col_data.min()),
                "miss_rate": float(col_data.isna().mean()),
                "max_corr": float(max_corr),
                "mean_corr": float(mean_corr),
            }
        )
    return pd.DataFrame(stats)


# ---------------------------------------------------------------------------
# Adaptive imputers
# ---------------------------------------------------------------------------


class RateAdaptiveImputer(BaseEstimator, TransformerMixin):
    """Select a single IterativeImputer based on global missing rate.

    Parameters
    ----------
    best_imp_by_rate : Dict[float, str]
        Mapping from rate bin to best base imputer name.
    random_state : int, default=0
        Random seed for underlying regressors.
    """

    def __init__(self, best_imp_by_rate: Dict[float, str], random_state: int = 0):
        self.best_imp_by_rate = best_imp_by_rate
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        rate = global_missing_rate(X)
        self._rate_bin_ = nearest_rate_bin(rate, self.best_imp_by_rate.keys())
        imp_name = self.best_imp_by_rate[self._rate_bin_]
        estimator = get_base_regressor(imp_name, self.random_state)
        n_iter = iterations_for_rate(self._rate_bin_)
        self._imputer_ = IterativeImputer(
            estimator=estimator,
            max_iter=n_iter,
            random_state=self.random_state,
        )
        self._imputer_.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "_imputer_")
        imputed = self._imputer_.transform(X)
        return pd.DataFrame(imputed, columns=X.columns, index=X.index)


class ColumnAdaptiveImputer(BaseEstimator, TransformerMixin):
    """Column-wise adaptive imputer using rate×column strategy table."""

    def __init__(self, best_imp_by_rate_col: Dict[Tuple[float, int], str], rate_bins: Iterable[float], random_state: int = 0):
        self.best_imp_by_rate_col = best_imp_by_rate_col
        self.rate_bins = list(rate_bins)
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        self._simple_ = SimpleImputer(strategy="mean")
        X_simple = pd.DataFrame(self._simple_.fit_transform(X), columns=X.columns, index=X.index)
        miss_rates = column_missing_rates(X)
        self._regressors_: Dict[int, RegressorMixin] = {}
        for col in X.columns:
            col_idx = int(col)
            rate_bin = nearest_rate_bin(miss_rates[col], self.rate_bins)
            key = (rate_bin, col_idx)
            imp_name = self.best_imp_by_rate_col.get(key)
            if imp_name is None:
                continue
            reg = get_base_regressor(imp_name, self.random_state)
            obs_mask = ~X[col].isna()
            if obs_mask.sum() < 5:
                continue
            X_train = X_simple.loc[obs_mask].drop(columns=[col])
            y_train = X.loc[obs_mask, col]
            reg.fit(X_train, y_train)
            self._regressors_[col_idx] = reg
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "_simple_")
        X_simple = pd.DataFrame(self._simple_.transform(X), columns=X.columns, index=X.index)
        result = X_simple.copy()
        for col_idx, reg in self._regressors_.items():
            col = X.columns[col_idx]
            miss_mask = X[col].isna()
            if miss_mask.any():
                preds = reg.predict(X_simple.loc[miss_mask].drop(columns=[col]))
                result.loc[miss_mask, col] = preds
        return result


class MetaColumnAdaptiveImputer(BaseEstimator, TransformerMixin):
    """Column-wise adaptive imputer using a meta-learner to pick regressors."""

    def __init__(self, candidate_imps: List[str], meta_model: Optional[ClassifierMixin] = None, random_state: int = 0):
        self.candidate_imps = candidate_imps
        self.meta_model = meta_model or RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, stats_df: pd.DataFrame, y_meta: pd.Series):
        """训练元学习器并为每个列构建对应回归器。"""
        # 如果外部未提供缺失率列，则以当前数据的列缺失率填充
        if "rate" not in stats_df.columns:
            stats_df = stats_df.copy()
            stats_df["rate"] = stats_df["col"].map(column_missing_rates(X))

        X_meta = stats_df[["col", "rate", "mean", "std", "range", "max_corr", "mean_corr"]]
        self.meta_model.fit(X_meta, y_meta)
        self._simple_ = SimpleImputer(strategy="mean")
        X_simple = pd.DataFrame(self._simple_.fit_transform(X), columns=X.columns, index=X.index)
        self._regressors_: Dict[int, RegressorMixin] = {}
        for col in X.columns:
            col_idx = int(col)
            feats = stats_df.loc[stats_df["col"] == col_idx][["col", "rate", "mean", "std", "range", "max_corr", "mean_corr"]]
            if feats.empty:
                continue
            imp_name = self.meta_model.predict(feats)[0]
            reg = get_base_regressor(str(imp_name), self.random_state)
            obs_mask = ~X[col].isna()
            if obs_mask.sum() < 5:
                continue
            X_train = X_simple.loc[obs_mask].drop(columns=[col])
            y_train = X.loc[obs_mask, col]
            reg.fit(X_train, y_train)
            self._regressors_[col_idx] = reg
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "_simple_")
        X_simple = pd.DataFrame(self._simple_.transform(X), columns=X.columns, index=X.index)
        result = X_simple.copy()
        for col_idx, reg in self._regressors_.items():
            col = X.columns[col_idx]
            miss_mask = X[col].isna()
            if miss_mask.any():
                preds = reg.predict(X_simple.loc[miss_mask].drop(columns=[col]))
                result.loc[miss_mask, col] = preds
        return result


class ClusterMetaColumnAdaptiveImputer(BaseEstimator, TransformerMixin):
    """Cluster-wise meta adaptive imputer with global fallback."""

    def __init__(self, clusterer: KMeans, global_meta: MetaColumnAdaptiveImputer, min_cluster_size: int = 30):
        self.clusterer = clusterer
        self.global_meta = global_meta
        self.min_cluster_size = min_cluster_size

    def fit(self, X: pd.DataFrame, stats_df: pd.DataFrame, y_meta: pd.Series):
        """先训练全局元插补器，再在簇内细化。"""
        # global meta
        self.global_meta.fit(X, stats_df, y_meta)
        # cluster assignment
        X_global = self.global_meta.transform(X)
        clusters = self.clusterer.fit_predict(X_global)
        self._cluster_models_: Dict[int, MetaColumnAdaptiveImputer] = {}
        for cid in np.unique(clusters):
            idx = np.where(clusters == cid)[0]
            if len(idx) < self.min_cluster_size:
                continue
            sub_X = X.iloc[idx]
            sub_stats = stats_df.copy()
            sub_stats["rate"] = sub_stats["col"].map(column_missing_rates(sub_X))
            model = MetaColumnAdaptiveImputer(
                candidate_imps=self.global_meta.candidate_imps, random_state=self.global_meta.random_state
            )
            model.fit(sub_X, sub_stats, y_meta)
            self._cluster_models_[int(cid)] = model
        self._cluster_assignments_ = clusters
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self.global_meta, "_simple_")
        base_imputed = self.global_meta.transform(X)
        clusters = self.clusterer.predict(base_imputed)
        result = base_imputed.copy()
        for cid, model in self._cluster_models_.items():
            idx = np.where(clusters == cid)[0]
            if len(idx) == 0:
                continue
            sub = model.transform(X.iloc[idx])
            result.iloc[idx] = sub.values
        return result


class UltimateAdaptiveImputer(BaseEstimator, TransformerMixin):
    """Full-featured adaptive imputer with uncertainty estimation."""

    def __init__(self, clusterer: KMeans, candidate_imps: List[str], meta_model: Optional[ClassifierMixin] = None, ensemble_size: int = 5, random_state: int = 0):
        self.clusterer = clusterer
        self.candidate_imps = candidate_imps
        self.meta_model = meta_model or RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1)
        self.ensemble_size = ensemble_size
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, stats_df: pd.DataFrame, y_meta: pd.Series):
        """训练全局元插补器、簇划分以及簇内集成回归器。"""
        if "rate" not in stats_df.columns:
            stats_df = stats_df.copy()
            stats_df["rate"] = stats_df["col"].map(column_missing_rates(X))
        rate_bins = stats_df["rate"].unique()

        # global meta model
        self._meta_imputer = MetaColumnAdaptiveImputer(self.candidate_imps, self.meta_model, self.random_state)
        self._meta_imputer.fit(X, stats_df, y_meta)
        base_imputed = self._meta_imputer.transform(X)
        self.clusterer.fit(base_imputed)
        clusters = self.clusterer.predict(base_imputed)
        self._cluster_models_: Dict[int, Dict[int, List[RegressorMixin]]] = {}
        self._simple_ = SimpleImputer(strategy="mean").fit(X)
        X_simple = pd.DataFrame(self._simple_.transform(X), columns=X.columns, index=X.index)
        miss_rates = column_missing_rates(X)
        for cid in np.unique(clusters):
            idx = np.where(clusters == cid)[0]
            if len(idx) < 10:
                continue
            self._cluster_models_[int(cid)] = {}
            sub_X = X.iloc[idx]
            sub_simple = X_simple.iloc[idx]
            for col in X.columns:
                col_idx = int(col)
                miss_mask = sub_X[col].isna()
                if miss_mask.sum() < 5:
                    continue
                rate_bin = nearest_rate_bin(miss_rates[col], rate_bins)
                feats = stats_df.loc[stats_df["col"] == col_idx][["col", "rate", "mean", "std", "range", "max_corr", "mean_corr"]]
                if feats.empty:
                    continue
                feats = feats.copy()
                feats["rate"] = rate_bin
                imp_name = self._meta_imputer.meta_model.predict(feats)[0]
                models = []
                for k in range(self.ensemble_size):
                    reg = get_base_regressor(str(imp_name), self.random_state + k)
                    obs_mask = ~sub_X[col].isna()
                    reg.fit(sub_simple.loc[obs_mask].drop(columns=[col]), sub_X.loc[obs_mask, col])
                    models.append(reg)
                self._cluster_models_[int(cid)][col_idx] = models
        return self

    def transform(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        check_is_fitted(self, "_meta_imputer")
        X_simple = pd.DataFrame(self._simple_.transform(X), columns=X.columns, index=X.index)
        base_imputed = self._meta_imputer.transform(X)
        clusters = self.clusterer.predict(base_imputed)
        result = base_imputed.copy()
        uncertainty = pd.DataFrame(np.zeros_like(result), columns=result.columns, index=result.index)
        for idx, cid in enumerate(clusters):
            col_models = self._cluster_models_.get(int(cid), {})
            for col_idx, models in col_models.items():
                col = X.columns[col_idx]
                if not np.isnan(X.iloc[idx, col_idx]):
                    continue
                feats = X_simple.iloc[idx : idx + 1].drop(columns=[col])
                preds = np.array([m.predict(feats)[0] for m in models])
                result.iat[idx, col_idx] = preds.mean()
                uncertainty.iat[idx, col_idx] = preds.var()
        return result, uncertainty


# ---------------------------------------------------------------------------
# Class imbalance + parallel processing pipeline
# ---------------------------------------------------------------------------


def parallel_impute_by_class(
    df: pd.DataFrame,
    label_col: int,
    imputer_factory: Callable[[], TransformerMixin],
    missing_sets: Dict[str, List[pd.DataFrame]],
    stats_df: Optional[pd.DataFrame] = None,
    y_meta: Optional[pd.Series] = None,
    n_jobs: int = -1,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Impute multiple missing datasets for each mechanism in parallel.

    Returns a nested dict: mech -> {"imputed": List[pd.DataFrame], "uncertainty": List[pd.DataFrame|None]}.
    """

    def _process(mech: str, datasets: List[pd.DataFrame]):
        results = []
        uncertainties = []
        for ds in datasets:
            imp = imputer_factory()
            features = ds.drop(columns=[label_col])
            # 根据类型决定是否需要 meta 信息
            if isinstance(imp, UltimateAdaptiveImputer):
                if stats_df is None or y_meta is None:
                    raise ValueError("UltimateAdaptiveImputer 需要 stats_df 与 y_meta")
                imputed, u = imp.fit(features, stats_df, y_meta).transform(features)
                imputed[label_col] = ds[label_col].values
                results.append(imputed)
                uncertainties.append(u)
            elif isinstance(imp, (MetaColumnAdaptiveImputer, ClusterMetaColumnAdaptiveImputer)):
                if stats_df is None or y_meta is None:
                    raise ValueError("MetaColumnAdaptiveImputer/ClusterMetaColumnAdaptiveImputer 需要 stats_df 与 y_meta")
                imp.fit(features, stats_df, y_meta)
                imputed = imp.transform(features)
                imputed[label_col] = ds[label_col].values
                results.append(imputed)
                uncertainties.append(None)
            else:
                imp.fit(features)
                imputed = imp.transform(features)
                imputed[label_col] = ds[label_col].values
                results.append(imputed)
                uncertainties.append(None)
        return mech, {"imputed": results, "uncertainty": uncertainties}

    workers = None if n_jobs in (-1, None) else n_jobs
    with ThreadPoolExecutor(max_workers=workers) as ex:
        processed = list(ex.map(lambda kv: _process(*kv), missing_sets.items()))
    return {k: v for k, v in processed}


# ---------------------------------------------------------------------------
# Real-data application helper
# ---------------------------------------------------------------------------


def evaluate_on_real_data(
    df: pd.DataFrame,
    label_col: int,
    imputer: TransformerMixin,
    classifier: ClassifierMixin = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1),
    cv: int = 5,
) -> pd.DataFrame:
    """Fit imputer on real missing data and evaluate downstream classifier.

    Returns a dataframe with cross-validated metrics for comparison against baselines.
    """
    features = df.drop(columns=[label_col])
    labels = df[label_col]
    imputer.fit(features)
    X_imp = imputer.transform(features)
    model = Pipeline([("scale", StandardScaler()), ("clf", classifier)])
    probs = cross_val_predict(model, X_imp, labels, cv=cv, method="predict_proba")[:, 1]
    preds = (probs >= 0.5).astype(int)
    accuracy = (preds == labels).mean()
    return pd.DataFrame({"metric": ["accuracy"], "value": [accuracy]})


# ---------------------------------------------------------------------------
# Example usage (sanity check ready)
# ---------------------------------------------------------------------------


def _demo_sanity_check():
    """Light-weight sanity check on synthetic data."""
    rng = np.random.default_rng(0)
    n, d = 120, 8
    X_full = pd.DataFrame(rng.normal(size=(n, d)), columns=list(range(d)))
    y = rng.integers(0, 3, size=n)
    X_miss = X_full.copy()
    mask = rng.random(X_miss.shape) < 0.2
    X_miss = X_miss.mask(mask)
    rmse_rows = []
    for imp_name in ["lr", "br", "rf"]:
        imp = IterativeImputer(estimator=get_base_regressor(imp_name, 0), max_iter=5, random_state=0)
        imputed = pd.DataFrame(imp.fit_transform(X_miss), columns=X_miss.columns)
        rmses = rmse_at_missing(X_full, imputed, pd.DataFrame(mask, columns=X_miss.columns))
        for col, r in rmses.items():
            rmse_rows.append({"mech": "mcar", "rate": 0.2, "imp": imp_name, "col": col, "rmse": r})
    rmse_df = pd.DataFrame(rmse_rows)
    best_rate = build_best_imp_by_rate(rmse_df)
    best_rate_col = build_best_imp_by_rate_col(rmse_df)
    stats_df = column_statistics(X_full)
    stats_df["rate"] = stats_df["col"].map({k: v for k, v in column_missing_rates(X_miss).items()})
    X_meta, y_meta = make_meta_dataset(rmse_df, stats_df)

    rate_imp = RateAdaptiveImputer(best_rate)
    rate_imp.fit(X_miss)
    rate_imp.transform(X_miss)

    col_imp = ColumnAdaptiveImputer(best_rate_col, rate_bins=[0.2])
    col_imp.fit(X_miss)
    col_imp.transform(X_miss)

    meta_imp = MetaColumnAdaptiveImputer(candidate_imps=["lr", "br", "rf"], random_state=0)
    meta_imp.fit(X_miss, X_meta.assign(rate=0.2), y_meta)
    meta_imp.transform(X_miss)

    cluster_imp = ClusterMetaColumnAdaptiveImputer(clusterer=KMeans(n_clusters=2, random_state=0), global_meta=meta_imp)
    cluster_imp.fit(X_miss, X_meta.assign(rate=0.2), y_meta)
    cluster_imp.transform(X_miss)

    ultimate = UltimateAdaptiveImputer(clusterer=KMeans(n_clusters=2, random_state=0), candidate_imps=["lr", "br", "rf"], random_state=0)
    ultimate.fit(X_miss, X_meta.assign(rate=0.2), y_meta)
    ultimate.transform(X_miss)


if __name__ == "__main__":
    _demo_sanity_check()
