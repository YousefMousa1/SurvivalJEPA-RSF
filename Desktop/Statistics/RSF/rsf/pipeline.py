import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from .metrics import concordance_index_by_area, survival_area
from .preprocess import (
    discretize_genes,
    impute_missing,
    impute_missing_mi,
    one_hot_encode,
    prepare_clinical,
)
from .relief import relief_rank
from .rsf import RandomSurvivalForest

CLINICAL_COLUMNS = [
    "age_at_diagnosis",
    "size",
    "lymph_nodes_positive",
    "grade",
    "histological",
    "ER_IHC_status",
    "ER_Expr",
    "PR_Expr",
    "HER2_SNP6_state",
    "HER2_Expr",
    "treatment",
    "inf_men_status",
    "group",
    "stage",
    "lymph_nodes_removed",
    "NPI",
    "cellularity",
    "Pam50_subtype",
    "int_clust_memb",
    "site",
    "Genefu",
]


def _infer_gene_columns(df, clinical_columns):
    exclude = set(clinical_columns + ["day", "status"])
    return [col for col in df.columns if col not in exclude]


def _build_time_event(df, horizon_years):
    days = df["day"].astype(float).to_numpy()
    status = df["status"].astype(str).str.lower().to_numpy()
    time_years = days / 365.25
    horizon = float(horizon_years)
    truncated_time = np.minimum(time_years, horizon)
    event = (status == "dead") & (time_years <= horizon)
    return truncated_time, event.astype(int)


def _target_at_horizon(times, events, horizon_years):
    # Binary target for Relief: dead by horizon -> 1, else 0.
    return events


def _prepare_feature_sets(df, dataset_type):
    clinical_cols = [c for c in CLINICAL_COLUMNS if c in df.columns]
    gene_cols = _infer_gene_columns(df, clinical_cols)

    clinical_df = df[clinical_cols].copy()
    gene_df = df[gene_cols].copy()

    clinical_df = prepare_clinical(clinical_df)

    if dataset_type == "Clinical_Only":
        clinical_df = clinical_df.drop(columns=["Pam50_subtype"], errors="ignore")
        return clinical_df, [], clinical_df.columns.tolist()

    if dataset_type == "Clinical_PAM":
        return clinical_df, [], clinical_df.columns.tolist()

    if dataset_type == "Clinical_Gene":
        clinical_df = clinical_df.drop(columns=["Pam50_subtype"], errors="ignore")
        gene_df = discretize_genes(gene_df, gene_cols)
        combined = pd.concat([clinical_df, gene_df], axis=1)
        return combined, gene_cols, combined.columns.tolist()

    raise ValueError(f"Unknown dataset_type: {dataset_type}")


def _select_features_method1(X_df, y, feature_count, rng):
    X_array = X_df.to_numpy()
    weights = relief_rank(X_array, y, rng=rng)
    top_indices = np.argsort(weights)[::-1][:feature_count]
    return X_df.columns[top_indices].tolist()


def _select_features_method2(X_df, y, gene_cols, clinical_cols, feature_count, rng):
    gene_df = X_df[gene_cols]
    gene_array = gene_df.to_numpy()
    weights = relief_rank(gene_array, y, rng=rng)
    gene_count = max(feature_count - len(clinical_cols), 0)
    top_gene_indices = np.argsort(weights)[::-1][:gene_count]
    selected_genes = gene_df.columns[top_gene_indices].tolist()
    return clinical_cols + selected_genes


def _one_hot_train_test(train_df, test_df):
    combined = pd.concat([train_df, test_df], axis=0)
    combined_enc = one_hot_encode(combined)
    train_enc = combined_enc.iloc[: len(train_df)]
    test_enc = combined_enc.iloc[len(train_df) :]
    return train_enc.to_numpy(), test_enc.to_numpy()


def run_cross_validation(
    df,
    dataset_type,
    method,
    feature_counts,
    horizon_years,
    n_splits=5,
    n_trees=200,
    min_unique_deaths=1,
    mtry=None,
    max_depth=None,
    seed=42,
    mi_imputations=0,
):
    rng = np.random.default_rng(seed)
    times, events = _build_time_event(df, horizon_years)

    X_df, gene_cols, all_cols = _prepare_feature_sets(df, dataset_type)
    if mi_imputations and mi_imputations > 1:
        imputed_datasets = impute_missing_mi(X_df, m=mi_imputations, seed=seed)
    else:
        imputed_datasets = [impute_missing(X_df)]

    clinical_cols = [c for c in all_cols if c not in gene_cols]

    results = {count: [] for count in feature_counts}

    for X_imputed in imputed_datasets:
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for train_idx, test_idx in kfold.split(X_imputed):
            X_train = X_imputed.iloc[train_idx]
            X_test = X_imputed.iloc[test_idx]
            times_train = times[train_idx]
            events_train = events[train_idx]
            times_test = times[test_idx]
            events_test = events[test_idx]

            y_train = _target_at_horizon(times_train, events_train, horizon_years)

            for count in feature_counts:
                if method == 1:
                    selected = _select_features_method1(X_train, y_train, count, rng)
                elif method == 2:
                    if dataset_type != "Clinical_Gene":
                        raise ValueError("Method 2 is only valid for Clinical_Gene data.")
                    selected = _select_features_method2(
                        X_train,
                        y_train,
                        gene_cols,
                        clinical_cols,
                        count,
                        rng,
                    )
                else:
                    raise ValueError("Method must be 1 or 2.")

                X_train_sel = X_train[selected]
                X_test_sel = X_test[selected]

                X_train_mat, X_test_mat = _one_hot_train_test(X_train_sel, X_test_sel)

                eval_times = np.linspace(0, horizon_years, num=100)

                model = RandomSurvivalForest(
                    n_trees=n_trees,
                    min_unique_deaths=min_unique_deaths,
                    mtry=mtry,
                    max_depth=max_depth,
                    rng=rng,
                )
                model.fit(X_train_mat, times_train, events_train)
                survival = model.predict_survival(X_test_mat, eval_times)

                areas = np.array([survival_area(eval_times, s) for s in survival])
                c_index = concordance_index_by_area(areas, times_test, events_test)
                results[count].append(c_index)

    summary = {}
    for count, scores in results.items():
        summary[count] = float(np.nanmean(scores))
    return summary
