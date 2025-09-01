#%% MCRAE FILTERED DENSITY
from IPython.display import display
import pandas as pd
import unicodedata
import numpy as np
from itertools import combinations
from openpyxl import load_workbook
from typing import Callable, Iterable, Optional, Tuple
import nltk
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from tqdm import tqdm
from scipy.sparse import csr_matrix


# ensure resources (safe to call multiple times)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

lemmatizer = WordNetLemmatizer()

def default_noun_normalizer(s: str) -> str:
    """Lowercase + WordNet noun lemma (handles plurals/irregulars)."""
    return lemmatizer.lemmatize(str(s).strip().lower(), pos="n")

def filter_by_lemma(
    df: pd.DataFrame,
    col: str,
    concepts_to_keep: Iterable[str],
    normalizer: Optional[Callable[[str], str]] = None,
    matched_list_col: Optional[str] = "matched_keep_originals",
    matched_canonical_col: Optional[str] = "matched_keep_canonical",
) -> pd.DataFrame:
    """
    Filter df rows where df[col] matches concepts_to_keep in a case/plural-insensitive way,
    without altering originals. Optionally add annotation columns if names are provided.
    """
    if normalizer is None:
        normalizer = default_noun_normalizer

    # build lemma -> originals mapping
    lemma_to_originals = defaultdict(list)
    for orig in concepts_to_keep:
        lem = normalizer(orig)
        if orig not in lemma_to_originals[lem]:
            lemma_to_originals[lem].append(orig)

    keep_lemmas = set(lemma_to_originals.keys())

    # normalize the target column
    norm_series = df[col].map(normalizer)

    # filter rows
    mask = norm_series.isin(keep_lemmas)
    out = df.loc[mask].copy()

    # optional extra columns
    if matched_list_col is not None:
        out[matched_list_col] = norm_series[mask].map(
            lambda lem: ", ".join(lemma_to_originals[lem])
        )
    if matched_canonical_col is not None:
        out[matched_canonical_col] = norm_series[mask].map(
            lambda lem: lemma_to_originals[lem][0]
        )

    return out

#%% INPUTS
### MCRAE
input_file_mcrae = '../others/McRae-norm_filtered.xlsx'  # Path to your input Excel file
mcrae_filtered = pd.read_excel(input_file_mcrae)
mcrae_filtered_restricted =  mcrae_filtered[
    mcrae_filtered.groupby("Feature")["Concept"].transform("nunique") >= 3
]
mcrae_filtered_restricted_notax = mcrae_filtered_restricted[mcrae_filtered_restricted['WB_Maj'] != 'c']
mcrae_filtered_restricted_notax.drop(columns=["Num_Corred_Pairs_No_Tax", "%_Corred_Pairs_No_Tax"], inplace=True)

mcrae_filtered_restricted_notax_matrix = mcrae_filtered_restricted_notax.pivot_table(
    index="Concept",
    columns="Feature",
    values="Prod_Freq",
    fill_value=0
)

concepts_to_keep = mcrae_filtered_restricted_notax["Concept"].unique()

### MRNGO
input_file_mrngo = '../processed/MRNGO_manual_vector_lemmas_summary.xlsx'  # Path to your input Excel file

mrngo = pd.read_excel(input_file_mrngo, sheet_name='aggregated_data')
mrngo_unique = mrngo.drop_duplicates(subset=['ID', 'concept', 'vector_lemma_C'])
mrngo_unique['ID_Count'] = (mrngo_unique.groupby(['vector_lemma_C', 'concept'])['ID'].transform('nunique'))
mrngo_unique_filtered_ID = mrngo_unique[mrngo_unique['ID_Count'] >= 2]
mrngo_unique_filtered = mrngo_unique_filtered_ID[mrngo_unique_filtered_ID["concept_EN"].isin(concepts_to_keep)]
mrngo_unique_filtered['Concept_Count'] = mrngo_unique_filtered['vector_lemma_C'].map((mrngo_unique_filtered.groupby('vector_lemma_C')['concept'].nunique()))
mrngo_unique_filtered_restricted = mrngo_unique_filtered[mrngo_unique_filtered['Concept_Count'] >= 3]
mrngo_unique_filtered_restricted_notax = mrngo_unique_filtered_restricted[mrngo_unique_filtered_restricted['WB_Maj'] != 'c']

#%% BOOTSTRAP

# -------------------------------------------------------------------
# Minimal version of your density function (works on any long df)
# -------------------------------------------------------------------
def add_feature_corr_metrics(
    df_long: pd.DataFrame,
    concept_col: str = "Concept",
    feature_col: str = "Feature",
    value_col: str = "Prod_Freq",
    min_r2: float = 0.065,
    matrix_wide: pd.DataFrame | None = None,
    percent_normalization: str = "dataset",  # "dataset" or "concept"
) -> pd.DataFrame:
    # build or use matrix
    if matrix_wide is None:
        matrix_wide = (
            df_long
            .pivot_table(index=concept_col, columns=feature_col, values=value_col, aggfunc="sum", fill_value=0)
        )
    numeric_matrix = matrix_wide.select_dtypes(include=[np.number]).fillna(0)

    # correlations
    corr = numeric_matrix.corr()
    r2 = corr.pow(2)

    feats = corr.columns.tolist()
    recs = []
    for i in range(len(feats)):
        for j in range(i+1, len(feats)):
            r2_ij = r2.iat[i, j]
            recs.append({"Feature1": feats[i], "Feature2": feats[j], "r2": r2_ij if r2_ij >= min_r2 else 0.0})
    high_corr = pd.DataFrame.from_records(recs, columns=["Feature1","Feature2","r2"])

    # denominator for % (dataset-level)
    all_feats = pd.Index(pd.unique(pd.concat([high_corr["Feature1"], high_corr["Feature2"]])))
    total_pairs_dataset = int(len(all_feats) * (len(all_feats) - 1) / 2)

    # feature set per concept (present if >0)
    concept_to_feats = (
        df_long.loc[df_long[value_col] > 0, [concept_col, feature_col]]
        .drop_duplicates()
        .groupby(concept_col)[feature_col]
        .apply(set)
    )

    rows = []
    for concept, feats_set in concept_to_feats.items():
        if not feats_set:
            rows.append((concept, 0, 0.0, 0.0))
            continue
        mask = high_corr["Feature1"].isin(feats_set) & high_corr["Feature2"].isin(feats_set)
        nz = high_corr.loc[mask & (high_corr["r2"] > 0.0)]

        num_corr_pairs = int(nz.shape[0])
        density_pct = float((nz["r2"] * 100.0).sum())

        if percent_normalization == "dataset":
            pct = 0.0 if total_pairs_dataset == 0 else num_corr_pairs / total_pairs_dataset
        else:
            m = len(feats_set)
            den = m * (m - 1) / 2
            pct = 0.0 if den == 0 else num_corr_pairs / den

        rows.append((concept, num_corr_pairs, pct, density_pct))

    metrics = pd.DataFrame(rows, columns=[concept_col, "Num_Corred_Pairs_No_Tax", "%_Corred_Pairs_No_Tax", "Density"])
    out = df_long.merge(metrics, on=concept_col, how="left")
    out["Num_Corred_Pairs_No_Tax"] = out["Num_Corred_Pairs_No_Tax"].fillna(0).astype(int)
    out["%_Corred_Pairs_No_Tax"] = out["%_Corred_Pairs_No_Tax"].fillna(0.0)
    out["Density"] = out["Density"].fillna(0.0)
    return out

#% Helper: observed density from your counts
def compute_observed_density(df_counts: pd.DataFrame,
                             concept_col="Concept", feature_col="Feature", freq_col="Prod_Freq",
                             min_r2=0.065) -> float:
    matrix = df_counts.pivot_table(index=concept_col, columns=feature_col, values=freq_col, fill_value=0)
    enriched = add_feature_corr_metrics(df_counts, concept_col, feature_col, freq_col,
                                        min_r2=min_r2, matrix_wide=matrix)
    return enriched.groupby(concept_col)["Density"].mean().mean()  # mean across concepts

#% Bootstrap from aggregated counts via binomial downsampling
def simulate_resample_density_from_counts(df_counts: pd.DataFrame,
                                          N_full: int | dict | pd.Series = 30,
                                          n_subsample: int = 15,
                                          k: int = 10_000,
                                          concept_col: str = "Concept",
                                          feature_col: str = "Feature",
                                          freq_col: str = "Prod_Freq",
                                          min_r2: float = 0.065,
                                          percent_normalization: str = "dataset",
                                          random_state: int | None = 42) -> np.ndarray:
    """
    df_counts: long df with aggregated counts per (Concept, Feature).
    N_full: either a single int applied to all concepts (e.g., 30),
            or a mapping/Series {concept: N_concept} if N varies.
    """
    rng = np.random.default_rng(random_state)

    # Prepare per-concept N if needed
    if isinstance(N_full, (int, np.integer)):
        N_map = None
        N_const = int(N_full)
    else:
        # dict/Series mapping
        N_map = pd.Series(N_full)
        N_const = None

    concepts = df_counts[concept_col].unique()
    densities = np.empty(k, dtype=float)

    # Pre-split by concept to speed up
    groups = {c: g[[feature_col, freq_col]].reset_index(drop=True) for c, g in df_counts.groupby(concept_col)}

    for it in range(k):
        print(it)
        # build one resampled counts table
        parts = []
        for c in concepts:
            g = groups[c]
            Nc = int(N_map.loc[c]) if N_map is not None else N_const
            # avoid div-by-zero
            if Nc <= 0:
                # no participants => all zeros
                freqs_ds = np.zeros(len(g), dtype=int)
            else:
                p = np.clip(g[freq_col].to_numpy(dtype=float) / Nc, 0.0, 1.0)
                freqs_ds = rng.binomial(n=n_subsample, p=p)
            tmp = pd.DataFrame({concept_col: c, feature_col: g[feature_col].values, freq_col: freqs_ds})
            parts.append(tmp)

        boot_counts = pd.concat(parts, ignore_index=True)
        boot_matrix = boot_counts.pivot_table(index=concept_col, columns=feature_col, values=freq_col, fill_value=0)

        boot_enriched = add_feature_corr_metrics(boot_counts, concept_col, feature_col, freq_col,
                                                 min_r2=min_r2, matrix_wide=boot_matrix,
                                                 percent_normalization=percent_normalization)
        # aggregate density across concepts for this iteration
        densities[it] = boot_enriched.groupby(concept_col)["Density"].mean().mean()

    return densities

#%% ------- EXAMPLE USAGE -------
# Expected input: df_counts with columns ["Concept","Feature","Prod_Freq"]
# If you also have per-concept N (max participants) varying, prepare a mapping like:
# N_by_concept = df_counts.groupby("Concept")["MaxParticipants"].max()

# Parameters
concept_col = "Concept"
feature_col = "Feature"
freq_col    = "Prod_Freq"
min_r2      = 0.065
N_full      = 30      # or pass a dict/Series {concept: N} if it varies
n_subsample = 15
k_boot      = 1000

# 1) Observed density from the full counts (e.g., N=30)
observed_density = compute_observed_density(
    df_counts=mrngo_unique_filtered_notax_restricted,            # <-- your counts df
    concept_col="concept",
    feature_col="vector_lemma_C",
    freq_col="ID_Count",
    min_r2=min_r2
)
#%%

# 2) Bootstrap distribution under N=15 by binomial downsampling
boot_densities = simulate_resample_density_from_counts(
    df_counts=mcrae_filtered_notax,            # <-- same counts df
    N_full=N_full,                             # or N_by_concept mapping
    n_subsample=n_subsample,
    k=k_boot,
    concept_col=concept_col,
    feature_col=feature_col,
    freq_col=freq_col,
    min_r2=min_r2,
    percent_normalization="dataset",           # or "concept"
    random_state=42
)

# 3) Inference: CI and p-values comparing observed vs bootstrap
lower, upper = np.percentile(boot_densities, [2.5, 97.5])
mu = boot_densities.mean()

# two-sided empirical p-value around bootstrap mean
p_two_sided = (np.sum(np.abs(boot_densities - mu) >= np.abs(observed_density - mu)) + 1) / (len(boot_densities) + 1)

# one-sided p-values
p_high = (np.sum(boot_densities >= observed_density) + 1) / (len(boot_densities) + 1)
p_low  = (np.sum(boot_densities <= observed_density) + 1) / (len(boot_densities) + 1)

print(f"Observed density (full N={N_full}): {observed_density:.6f}")
print(f"Bootstrap (N={n_subsample}) mean:   {mu:.6f}")
print(f"95% bootstrap CI: [{lower:.6f}, {upper:.6f}]")
print(f"p (two-sided): {p_two_sided:.6f} | p_high: {p_high:.6f} | p_low: {p_low:.6f}")

