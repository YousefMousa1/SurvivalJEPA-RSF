RSF Implementation (Neapolitan & Jiang, 2015)

This project implements the Random Survival Forest (RSF) pipeline described in
"Study of Integrated Heterogeneous Data Reveals Prognostic Power of Gene Expression for Breast Cancer Survival".
It includes the preprocessing choices, ReliefF feature selection (Method 1 and Method 2),
5/10/15-year horizon evaluation, and the concordance computation described in the paper.

What matches the paper
- RSF: bootstrap sampling, random feature selection at each split, log-rank split criterion,
  leaf nodes constrained by minimum unique deaths, Nelson-Aalen CHF, ensemble-average CHF.
- Gene expression discretization: equal-width bins into low/medium/high (M=3).
- Clinical discretization: age_at_diagnosis, size, lymph_nodes_positive with the paper's bins.
- ReliefF: discrete-feature Relief algorithm, nearest hit/miss updates.
- Evaluation: 5-fold cross validation and concordance based on survival-curve area.

Notes about imputation
- The paper uses the Multiple Imputation with Diagnostics (MI) package. This code can run
  multiple imputations via an IterativeImputer with posterior sampling.
  Use --mi-imputations N to enable.

Dataset requirements
- CSV with at least these columns:
  - day: number of days since initial consultation
  - status: "dead" or "alive"
- Clinical columns (if present) should use the paper's names:
  age_at_diagnosis, size, lymph_nodes_positive, grade, histological, ER_IHC_status,
  ER_Expr, PR_Expr, HER2_SNP6_state, HER2_Expr, treatment, inf_men_status, group, stage,
  lymph_nodes_removed, NPI, cellularity, Pam50_subtype, int_clust_memb, site, Genefu.
- Gene expression columns are inferred as all remaining columns not in the clinical list
  and not in {day, status}.

Running the pipeline
- Clinical + Gene data with Method 2 (paper default):
  python run_rsf.py --data data.csv --dataset Clinical_Gene --method 2

- Clinical only:
  python run_rsf.py --data data.csv --dataset Clinical_Only --method 1

- Clinical + PAM50:
  python run_rsf.py --data data.csv --dataset Clinical_PAM --method 1

- With multiple imputations (MI):
  python run_rsf.py --data data.csv --dataset Clinical_Gene --method 2 --mi-imputations 5

Running with T-JEPA embeddings
- If you have exported embeddings from the T-JEPA repo (see that README),
  run RSF on the embedding CSV like this:
  python run_rsf.py --data /path/to/metabric_tjepa_embeddings_64_cox5y.csv \
    --dataset Clinical_Gene --method 1 --feature-counts 64 --n-trees 50 --years 5 10 15

Key parameters
- --feature-counts: default 30 50 100 150
  - Method 1: ReliefF selects from combined clinical + genes.
  - Method 2: ReliefF selects from genes only, then adds all clinical features.
- --n-trees: number of bootstrap trees (default 200)
- --mtry: number of candidate covariates per split (default sqrt(p))
- --min-unique-deaths: stop splitting when <= D unique deaths at a node

Outputs
- JSON mapping each horizon year to the mean concordance index for each feature-count.

Files
- rsf/rsf.py: RSF and survival tree implementation
- rsf/relief.py: Relief feature ranking
- rsf/preprocess.py: discretization and imputation
- rsf/pipeline.py: cross-validation pipeline
- rsf/metrics.py: concordance calculation
- run_rsf.py: CLI wrapper
