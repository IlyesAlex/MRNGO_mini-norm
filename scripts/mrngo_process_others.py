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
input_file_mcrae_full = "../others/McRae-norm.xlsx"
mcrae_full = pd.read_excel(input_file_mcrae_full)
mcrae_full_notax = mcrae_full#mcrae_full[mcrae_full['WB_Maj'] != 'c']
mcrae_full_notax.drop(columns=["Num_Corred_Pairs_No_Tax", "%_Corred_Pairs_No_Tax"], inplace=True)
mcrae_full_notax_restrict =  mcrae_full_notax[
    mcrae_full_notax.groupby("Feature")["Concept"].transform("nunique") >= 3
]

input_file_mcrae = '../others/McRae-norm_filtered.xlsx'  # Path to your input Excel file
mcrae_filtered = pd.read_excel(input_file_mcrae)
mcrae_filtered_notax = mcrae_filtered[mcrae_filtered['WB_Maj'] != 'c']
mcrae_filtered_notax.drop(columns=["Num_Corred_Pairs_No_Tax", "%_Corred_Pairs_No_Tax"], inplace=True)

mcrae_filtered_notax_matrix = mcrae_filtered_notax.pivot_table(
    index="Concept",
    columns="Feature",
    values="Prod_Freq",
    fill_value=0
)

concepts_to_keep = mcrae_filtered_notax["Concept"].unique()

### MRNGO
input_file_mrngo = '../processed/MRNGO_manual_vector_lemmas_summary.xlsx'  # Path to your input Excel file

mrngo = pd.read_excel(input_file_mrngo, sheet_name='aggregated_data')
mrngo_unique = mrngo.drop_duplicates(subset=['ID', 'concept', 'vector_lemma_C'])
mrngo_unique['ID_Count'] = (mrngo_unique.groupby(['vector_lemma_C', 'concept'])['ID'].transform('nunique'))
mrngo_unique_filtered_ID = mrngo_unique[mrngo_unique['ID_Count'] >= 2]
mrngo_unique_filtered_ID['Concept_Count'] = mrngo_unique_filtered_ID['vector_lemma_C'].map((mrngo_unique.groupby('vector_lemma_C')['concept'].nunique()))
mrngo_unique_filtered = mrngo_unique_filtered_ID[mrngo_unique_filtered_ID["concept_EN"].isin(concepts_to_keep)]
mrngo_unique_filtered_notax = mrngo_unique_filtered[mrngo_unique_filtered['WB_Maj'] != 'c']
mrngo_unique_filtered_notax_restricted = mrngo_unique_filtered_notax[mrngo_unique_filtered_notax['Concept_Count'] >= 3] 

### VINSON
input_file_vinson = "../others/Vinson-norm.xlsx"
vinson_matrix = pd.read_excel(input_file_vinson, index_col=0)
vinson = (
    vinson_matrix.stack()
          .reset_index(name="frequency")
          .rename(columns={"level_0":"Feature", "level_1":"Concept"})
          .query("frequency != 0")
)
vinson_filtered = filter_by_lemma(vinson, "Concept", concepts_to_keep)

#%%

def add_feature_corr_metrics(
    df_long: pd.DataFrame,
    concept_col: str = "Concept",
    feature_col: str = "Feature",
    value_col: Optional[str] = "Prod_Freq",      # ← set to None to use row counts per (Concept, Feature)
    min_r2: float = 0.065,
    matrix_wide: pd.DataFrame | None = None,     # optional precomputed Concept×Feature matrix
    percent_normalization: str = "dataset",      # "dataset" or "concept"
) -> pd.DataFrame:
    """
    Compute per-concept correlation-based metrics from a concept×feature dataset
    and merge them back to the long df.

    Modes
    -----
    1) Value mode (default): if `value_col` is provided and exists in df_long, use it.
    2) Count mode: if `value_col` is None or missing, count rows per (Concept, Feature).

    Added columns
    -------------
    - 'Num_Corred_Pairs_No_Tax'
    - '%_Corred_Pairs_No_Tax'
    - 'Density'

    Parameters
    ----------
    df_long : pd.DataFrame
        Long-form table with at least [concept_col, feature_col].
    concept_col, feature_col, value_col : str or None
        Column names for concept, feature, and numeric value (optional).
    min_r2 : float
        Minimum shared variance (r^2) threshold to count a pair as “correlated”.
    matrix_wide : pd.DataFrame | None
        Optional Concept×Feature matrix (index=concepts, columns=features). If None, it is built.
    percent_normalization : {"dataset","concept"}
        - "dataset": % = (# correlated pairs within concept) / (all possible pairs in dataset)
        - "concept": % = (# correlated pairs within concept) / (m*(m-1)/2) for that concept

    Returns
    -------
    pd.DataFrame
        The original df_long with 3 per-concept columns added.
    """
    # ---- 0) Validate presence of key columns ----
    required_base = {concept_col, feature_col}
    missing_base = required_base - set(df_long.columns)
    if missing_base:
        raise ValueError(f"`df_long` is missing required columns: {missing_base}")

    # ---- 1) Build a working long df that always has a numeric value column ----
    if value_col is None or value_col not in df_long.columns:
        # COUNT MODE: count rows per (Concept, Feature)
        work = (
            df_long
            .groupby([concept_col, feature_col], dropna=False)
            .size()
            .reset_index(name="__value__")
        )
        used_value_col = "__value__"
    else:
        # VALUE MODE: use the provided numeric column (coerce if needed)
        work = df_long[[concept_col, feature_col, value_col]].copy()
        if not np.issubdtype(work[value_col].dtype, np.number):
            work[value_col] = pd.to_numeric(work[value_col], errors="coerce").fillna(0)
        used_value_col = value_col

    # ---- 2) Build / accept the matrix ----
    if matrix_wide is None:
        matrix_wide = (
            work
            .pivot_table(index=concept_col, columns=feature_col, values=used_value_col,
                         aggfunc="sum", fill_value=0)
        )

    numeric_matrix = matrix_wide.select_dtypes(include=[np.number]).fillna(0)

    # ---- 3) Feature–feature correlations & r^2 ----
    corr = numeric_matrix.corr()
    r2 = corr.pow(2)

    feats = corr.columns.tolist()
    recs = []
    for i in range(len(feats)):
        for j in range(i+1, len(feats)):
            r2_ij = r2.iat[i, j]
            recs.append({
                "Feature1": feats[i],
                "Feature2": feats[j],
                "r2": r2_ij if r2_ij >= min_r2 else 0.0
            })
            if r2_ij >= min_r2:
                print(feats[i], feats[j], r2_ij)
    high_corr = pd.DataFrame.from_records(recs, columns=["Feature1","Feature2","r2"])

    # for "dataset" normalization
    all_feats = pd.Index(pd.unique(pd.concat([high_corr["Feature1"], high_corr["Feature2"]])))
    total_pairs_dataset = int(len(all_feats) * (len(all_feats) - 1) / 2)

    # ---- 4) Feature set per concept (value > 0 considered present) ----
    concept_to_feats = (
        work.loc[work[used_value_col] > 0, [concept_col, feature_col]]
        .drop_duplicates()
        .groupby(concept_col)[feature_col]
        .apply(set)
    )

    if percent_normalization not in {"dataset", "concept"}:
        raise ValueError("percent_normalization must be 'dataset' or 'concept'.")

    rows = []
    for concept, feats_set in concept_to_feats.items():
        if not feats_set:
            rows.append((concept, 0, 0.0, 0.0))
            continue

        mask = high_corr["Feature1"].isin(feats_set) & high_corr["Feature2"].isin(feats_set)
        subpairs = high_corr.loc[mask]
        nz = subpairs.loc[subpairs["r2"] > 0.0]
        print(concept, feats_set, nz)

        num_corr_pairs = int(nz.shape[0])
        density_pct = float((nz["r2"] * 100.0).sum())

        if percent_normalization == "dataset":
            pct = 0.0 if total_pairs_dataset == 0 else num_corr_pairs / total_pairs_dataset
        else:  # "concept"
            m = len(feats_set)
            den = m * (m - 1) / 2
            pct = 0.0 if den == 0 else num_corr_pairs / den

        rows.append((concept, num_corr_pairs, pct, density_pct))

    metrics = pd.DataFrame(
        rows,
        columns=[concept_col, "Num_Corred_Pairs_No_Tax", "%_Corred_Pairs_No_Tax", "Density"]
    )

    # ---- 5) Merge metrics back to the ORIGINAL df_long ----
    out = df_long.merge(metrics, on=concept_col, how="left")
    out["Num_Corred_Pairs_No_Tax"] = out["Num_Corred_Pairs_No_Tax"].fillna(0).astype(int)
    out["%_Corred_Pairs_No_Tax"] = out["%_Corred_Pairs_No_Tax"].fillna(0.0)
    out["Density"] = out["Density"].fillna(0.0)

    return out, concept_to_feats, numeric_matrix

#%% MATRICES

mrngo_enhanced = add_feature_corr_metrics(mrngo_unique_filtered_notax_restricted, "concept", "vector_lemma_C")
vinson_enhanced = add_feature_corr_metrics(vinson_filtered, "Concept", "Feature", "frequency")
mcrae_enhanced, mcrae_conceptsfeat, mcrae_matrix = add_feature_corr_metrics(mcrae_filtered_notax, "Concept", "Feature", "Prod_Freq")
mcrae_enhanced_full, mcrae_conceptsfeat_full, mcrae_matrix_full = add_feature_corr_metrics(mcrae_full_notax_restrict, "Concept", "Feature", "Prod_Freq")

