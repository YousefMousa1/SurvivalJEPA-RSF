import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

AGE_BINS = [0, 39, 54, 69, 84, 101]
SIZE_BINS = [0, 20, 50, 181]


def impute_missing(df):
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            median = df[col].median()
            df[col] = df[col].fillna(median)
        else:
            mode = df[col].mode(dropna=True)
            if len(mode) == 0:
                df[col] = df[col].fillna("unknown")
            else:
                df[col] = df[col].fillna(mode.iloc[0])
    return df


def impute_missing_mi(df, m=5, seed=42, max_iter=10):
    """
    Multiple Imputation via IterativeImputer (sample_posterior=True).
    Returns a list of imputed DataFrames.
    """
    rng = np.random.default_rng(seed)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = [c for c in df.columns if c not in numeric_cols]

    filled = df.copy()
    for col in non_numeric_cols:
        mode = filled[col].mode(dropna=True)
        filled[col] = filled[col].fillna(mode.iloc[0] if len(mode) else "unknown")

    imputations = []
    if numeric_cols:
        imputer = IterativeImputer(
            random_state=seed,
            max_iter=max_iter,
            sample_posterior=True,
        )
        for i in range(m):
            imputer.random_state = int(rng.integers(0, 1_000_000))
            numeric_imputed = imputer.fit_transform(filled[numeric_cols])
            imputed_df = filled.copy()
            imputed_df[numeric_cols] = numeric_imputed
            imputations.append(imputed_df)
    else:
        imputations = [filled.copy() for _ in range(m)]

    return imputations


def discretize_age(series):
    labels = ["0-39", "39-54", "54-69", "69-84", "84-100"]
    return pd.cut(series, bins=AGE_BINS, labels=labels, right=False, include_lowest=True)


def discretize_size(series):
    labels = ["0-20", "20-50", "50-180"]
    return pd.cut(series, bins=SIZE_BINS, labels=labels, right=False, include_lowest=True)


def discretize_lymph_nodes(series):
    def _bin(value):
        if pd.isna(value):
            return np.nan
        if value == 0:
            return "0"
        if value == 1:
            return "1"
        if 2 <= value <= 3:
            return "2-3"
        if 4 <= value <= 5:
            return "4-5"
        if 6 <= value <= 9:
            return "6-9"
        return "10+"

    return series.apply(_bin)


def equal_width_discretize(series, bins=3, labels=None):
    if labels is None:
        labels = ["low", "medium", "high"]
    return pd.cut(series, bins=bins, labels=labels, include_lowest=True)


def prepare_clinical(df):
    df = df.copy()
    if "age_at_diagnosis" in df.columns:
        df["age_at_diagnosis"] = discretize_age(df["age_at_diagnosis"])
    if "size" in df.columns:
        df["size"] = discretize_size(df["size"])
    if "lymph_nodes_positive" in df.columns:
        df["lymph_nodes_positive"] = discretize_lymph_nodes(df["lymph_nodes_positive"])
    return df


def discretize_genes(df, gene_columns):
    df = df.copy()
    for col in gene_columns:
        df[col] = equal_width_discretize(df[col], bins=3)
    return df


def one_hot_encode(df):
    return pd.get_dummies(df, drop_first=False)
