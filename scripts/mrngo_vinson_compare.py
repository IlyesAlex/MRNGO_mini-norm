#%% VINSON FILTERED DENSITY
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
import networkx as nx
from pathlib import Path
import re
import textwrap


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
input_file_vinson = '../others/Vinson-norm_filtered.xlsx'  # Path to your input Excel file
vinson_filtered = pd.read_excel(input_file_vinson)
concepts_to_keep = vinson_filtered["Concept"].unique()

### MRNGO
input_file_mrngo = '../processed/MRNGO_manual_vector_lemmas_summary.xlsx'  # Path to your input Excel file

mrngo = pd.read_excel(input_file_mrngo, sheet_name='aggregated_data')
mrngo_selected = mrngo[["ID", "concept_EN", "vector_lemma_C"]].rename(
    columns={
        "concept_EN": "Concept",
        "vector_lemma_C": "Feature"
    }
)
mrngo_unique = mrngo_selected.drop_duplicates(subset=['ID', 'Concept', 'Feature'])
mrngo_unique['Frequency'] = (mrngo_unique.groupby(['Feature', 'Concept'])['ID'].transform('nunique'))
mrngo_unique_simple = mrngo_unique.drop(columns=["ID"]).drop_duplicates().reset_index(drop=True)
mrngo_unique_filtered = filter_by_lemma(mrngo_unique_simple, "Concept", concepts_to_keep)
mrngo_unique_filtered = mrngo_unique_filtered[mrngo_unique_filtered["Feature"].map(lambda x: "_IND" not in x)]
mrngo_unique_filtered = mrngo_unique_filtered[["Concept", "Feature", "Frequency"]].reset_index(drop=True)

#%% BOOTSTRAP

# ============================
# Core metrics
# ============================
def compute_metrics(df, r2_thresh=0.10):
    """
    Expects columns: Concept, Feature, Frequency

    Returns:
      enriched_rows: long df with columns
        Concept, Feature, Frequency,
        Num_Features, Num_Features_Nonunique,
        CPF, Distinctiveness, Mean_Distinctiveness,
        Density, Density_Weighted, Mean_Distance
      concept_metrics: per-concept summary of the same concept-level metrics
      r2_ff: feature×feature r^2 matrix (based on raw counts, filtered by Frequency>=2 and CPF>=3)
    """
    # --- Clean & collapse duplicates
    df = df[["Concept","Feature","Frequency"]].copy()
    df["Frequency"] = pd.to_numeric(df["Frequency"], errors="coerce").fillna(0)
    agg_cf = df.groupby(["Concept","Feature"], as_index=False)["Frequency"].sum()

    # --- CPF & Distinctiveness (presence = Frequency >= 1)
    present_cf = agg_cf.loc[agg_cf["Frequency"] >= 1, ["Concept","Feature"]]
    cpf = (present_cf.drop_duplicates()
                    .groupby("Feature")["Concept"].nunique()
                    .rename("CPF")
                    .reset_index())
    df_row = agg_cf.merge(cpf, on="Feature", how="left")
    df_row["Distinctiveness"] = 1.0 / df_row["CPF"].replace({0: np.nan})

    # --- Concept-level feature counts (each feature counted once)
    num_features = (agg_cf.loc[agg_cf["Frequency"] >= 1]
                         .groupby("Concept")["Feature"].nunique()
                         .rename("Num_Features"))
    num_features_nonunique = (agg_cf.loc[agg_cf["Frequency"] > 1]
                                   .groupby("Concept")["Feature"].nunique()
                                   .rename("Num_Features_Nonunique"))
    by_c = (pd.DataFrame(num_features)
              .join(num_features_nonunique, how="left")
              .reset_index())

    # --- Mean Distinctiveness per concept (only present features)
    mean_distinct = (df_row.loc[df_row["Frequency"] >= 1]
                       .groupby("Concept")["Distinctiveness"]
                       .mean()
                       .rename("Mean_Distinctiveness")
                       .reset_index())

    # ============================
    # Correlation / Density (RAW counts)
    # Filter to rows with Frequency >= 2 AND features with CPF >= 3
    # ============================
    corr_df = agg_cf.merge(cpf, on="Feature", how="left")
    corr_df = corr_df[(corr_df["Frequency"] >= 2) & (corr_df["CPF"] >= 3)]

    # Pivot and compute feature×feature r^2
    raw_corr = corr_df.pivot(index="Concept", columns="Feature", values="Frequency").fillna(0)
    if raw_corr.shape[1] >= 2:
        r2_ff = (raw_corr.corr(method="pearson") ** 2).fillna(0.0)
        np.fill_diagonal(r2_ff.values, 0.0)
    else:
        r2_ff = pd.DataFrame(index=raw_corr.columns, columns=raw_corr.columns, dtype=float).fillna(0.0)

    # Per-concept Density & Density_Weighted: sum/mean r^2 for within-concept feature pairs (above threshold)
    dens_rows = []
    all_concepts = agg_cf["Concept"].unique()
    for concept in all_concepts:
        feats = corr_df.loc[corr_df["Concept"] == concept, "Feature"].unique().tolist()
        if len(feats) < 2 or r2_ff.shape[0] == 0:
            dens_rows.append({"Concept": concept, "Density": np.nan, "Density_Weighted": np.nan})
            continue
        # r^2 submatrix for this concept's eligible features
        sub = r2_ff.loc[r2_ff.index.intersection(feats), r2_ff.columns.intersection(feats)]
        if sub.shape[0] < 2:
            dens_rows.append({"Concept": concept, "Density": np.nan, "Density_Weighted": np.nan})
            continue
        tri = sub.where(np.triu(np.ones(sub.shape, dtype=bool), k=1))
        vals = tri.stack().values
        sel = vals[vals > r2_thresh]
        density = float(sel.sum()) if sel.size else 0.0
        density_weighted = float(sel.mean()) if sel.size else np.nan
        dens_rows.append({"Concept": concept, "Density": density, "Density_Weighted": density_weighted})
    dens = pd.DataFrame(dens_rows)

    # --- Mean shortest-path distance (edge if share ≥1 feature; uses presence >=1, not corr filters)
    presence = (agg_cf.assign(Present=agg_cf["Frequency"] >= 1)
                      .pivot(index="Concept", columns="Feature", values="Present")
                      .fillna(0).astype(int))
    shared = presence.values @ presence.values.T
    np.fill_diagonal(shared, 0)
    concepts = list(presence.index)

    G = nx.Graph()
    G.add_nodes_from(concepts)
    for i in range(len(concepts)):
        for j in range(i+1, len(concepts)):
            if shared[i, j] >= 1:
                G.add_edge(concepts[i], concepts[j])

    mean_dist = {}
    for c in concepts:
        lengths = nx.single_source_shortest_path_length(G, c)
        vals = [d for n, d in lengths.items() if n != c]
        mean_dist[c] = float(np.mean(vals)) if len(vals) else np.nan
    md = pd.DataFrame({"Concept": concepts, "Mean_Distance": [mean_dist[c] for c in concepts]})

    # --- Merge concept-level metrics and broadcast to rows
    concept_metrics = (by_c.merge(mean_distinct, on="Concept", how="left")
                         .merge(dens, on="Concept", how="left")
                         .merge(md, on="Concept", how="left"))

    enriched = df_row.merge(concept_metrics, on="Concept", how="left")

    cols = ["Concept","Feature","Frequency",
            "Num_Features","Num_Features_Nonunique",
            "CPF","Distinctiveness","Mean_Distinctiveness",
            "Density","Density_Weighted","Mean_Distance"]
    return enriched[cols], concept_metrics, r2_ff

# ============================
# Bootstrap (20 -> 15 via Binomial on aggregated counts)
# ============================
def bootstrap_vinson(vinson_df, k=1000, n_from=20, n_to=15, seed=42, r2_thresh=0.10):
    """
    Down-sample 20-participant aggregated counts to 15 via Binomial(n_to, p=freq/n_from),
    then compute metrics for each replicate. Returns a long per-concept metrics table with 'iter'.
    """
    rng = np.random.default_rng(seed)
    out = []
    for it in range(k):
        sim = vinson_df[["Concept","Feature","Frequency"]].copy()
        p = np.clip(sim["Frequency"].astype(float) / float(n_from), 0.0, 1.0)
        sim["Frequency"] = rng.binomial(n=n_to, p=p)
        _, cm, _ = compute_metrics(sim, r2_thresh=r2_thresh)
        cm = cm.copy()
        cm["iter"] = it
        out.append(cm)
    return pd.concat(out, ignore_index=True)

# ---------- UPDATED compare: robust to mismatches; optional concept intersection ----------
def compare_baseline_to_bootstrap(
    baseline_cm,
    boot_cm,
    metrics=("Num_Features","Num_Features_Nonunique",
             "Mean_Distinctiveness","Density","Density_Weighted","Mean_Distance"),
    keep_unmatched=False  # if False: only compare concepts present in BOTH baseline & boot
):
    # Normalize concept strings to reduce accidental mismatches
    base = baseline_cm.copy()
    boot = boot_cm.copy()
    base["Concept"] = base["Concept"].astype(str).str.strip()
    boot["Concept"] = boot["Concept"].astype(str).str.strip()

    base_idx = base.set_index("Concept")

    # Concept set handling
    if keep_unmatched:
        concepts_to_check = base_idx.index.unique()
    else:
        concepts_to_check = base_idx.index.intersection(boot["Concept"].unique())

    # Only keep metrics that actually exist in both dataframes
    metrics = tuple(m for m in metrics if (m in base.columns) and (m in boot.columns))
    if not metrics:
        return pd.DataFrame(columns=["Concept","metric","baseline","pvalue","mean","std","q025","q975"])

    rows = []
    for concept in concepts_to_check:
        if concept not in base_idx.index:
            continue
        brow = base_idx.loc[concept]

        sub = boot.loc[boot["Concept"] == concept]

        for m in metrics:
            arr = pd.to_numeric(sub[m], errors="coerce").dropna().values
            bval = pd.to_numeric(pd.Series([brow[m]]), errors="coerce").iloc[0]

            if arr.size == 0 or pd.isna(bval):
                rows.append({"Concept": concept, "metric": m, "baseline": bval,
                             "pvalue": np.nan, "mean": np.nan, "std": np.nan,
                             "q025": np.nan, "q975": np.nan})
                continue

            mean = float(np.mean(arr))
            std  = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
            q025, q975 = np.quantile(arr, [0.025, 0.975])
            # Empirical two-sided p-value (discrete-friendly)
            pval = float(2 * min((arr <= bval).mean(), (arr >= bval).mean()))
            pval = min(pval, 1.0)

            rows.append({"Concept": concept, "metric": m, "baseline": float(bval),
                         "pvalue": pval, "mean": mean, "std": std, "q025": q025, "q975": q975})
    return pd.DataFrame(rows)


# ---------- NEW: dataset-level (global) comparison ----------
def compare_dataset_to_bootstrap(
    baseline_cm,
    boot_cm,
    agg_map=None  # dict metric -> 'sum' | 'mean' | callable
):
    """
    Compare the WHOLE DATASET by aggregating concept metrics, then computing empirical p-values
    from bootstrap replicates. Default aggregations:
      - sum: Num_Features, Num_Features_Nonunique
      - mean: Mean_Distinctiveness, Density, Density_Weighted, Mean_Distance
    Returns: rows with metric | baseline | pvalue | mean | std | q025 | q975
    """
    if agg_map is None:
        agg_map = {
            "Num_Features": "mean",
            "Num_Features_Nonunique": "mean",
            "Mean_Distinctiveness": "mean",
            "Density": "mean",
            "Density_Weighted": "mean",
            "Mean_Distance": "mean",
        }

    # Clean up concept labels
    base = baseline_cm.copy()
    boot = boot_cm.copy()
    base["Concept"] = base["Concept"].astype(str).str.strip()
    boot["Concept"] = boot["Concept"].astype(str).str.strip()

    rows = []
    # Bootstrap iterations present?
    if "iter" not in boot.columns:
        raise ValueError("boot_cm must contain an 'iter' column for dataset-level comparison.")

    for metric, how in agg_map.items():
        if metric not in base.columns or metric not in boot.columns:
            # skip missing
            continue

        # Baseline aggregate
        if how == "sum":
            bval = pd.to_numeric(base[metric], errors="coerce").fillna(0).sum()
        elif how == "mean":
            bval = pd.to_numeric(base[metric], errors="coerce").dropna().mean()
        elif callable(how):
            bval = how(pd.to_numeric(base[metric], errors="coerce"))
        else:
            raise ValueError(f"Unknown aggregation for {metric}: {how}")

        # Bootstrap distribution over iterations
        grp = boot.groupby("iter")[metric]
        if how == "sum":
            arr = grp.apply(lambda s: pd.to_numeric(s, errors="coerce").fillna(0).sum()).values
        elif how == "mean":
            arr = grp.apply(lambda s: pd.to_numeric(s, errors="coerce").dropna().mean()).values
        elif callable(how):
            arr = grp.apply(lambda s: how(pd.to_numeric(s, errors="coerce"))).values

        arr = pd.to_numeric(pd.Series(arr), errors="coerce").dropna().values
        if arr.size == 0 or pd.isna(bval):
            rows.append({"metric": metric, "baseline": bval,
                         "pvalue": np.nan, "mean": np.nan, "std": np.nan, "q025": np.nan, "q975": np.nan})
            continue

        mean = float(np.mean(arr))
        std  = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
        q025, q975 = np.quantile(arr, [0.025, 0.975])
        pval = float(2 * min((arr <= bval).mean(), (arr >= bval).mean()))
        pval = min(pval, 1.0)

        rows.append({"metric": metric, "baseline": float(bval),
                     "pvalue": pval, "mean": mean, "std": std, "q025": q025, "q975": q975})
    return pd.DataFrame(rows)

def plot_dataset_bootstrap(
    baseline_cm,
    boot_cm,
    output_dir="bootstrap_outputs/dataset_level",
    agg_map=None,     # dict: metric -> 'sum' | 'mean' | callable
    bins=40,
    dpi=150
):
    """
    Dataset-level bootstrap plots:
      - Histogram color: #9b7ddc
      - Baseline line:   #E45756, with ±1 SE band
      - 95% bootstrap interval shaded
      - Y label: "Number of simulated datasets"
      - X label: telltale names
      - ONLY the APA-style p-value is displayed, centered; no title or extra text.
    """
    import matplotlib.pyplot as plt
    from pathlib import Path
    import re

    if agg_map is None:
        agg_map = {
            "Num_Features": "mean",
            "Num_Features_Nonunique": "mean",
            "Mean_Distinctiveness": "mean",
            "Density": "mean",
            "Density_Weighted": "mean",
            "Mean_Distance": "mean",
        }

    # Friendly X-labels
    label_map = {
        "Num_Features": "Mean number of features per concept",
        "Num_Features_Nonunique": "Mean number of non-unique features per concept",
        "Mean_Distinctiveness": "Mean distinctiveness (1/[concepts per features]) per concept",
        "Density": "Sum of feature-pair r² within concepts ([concepts per features] ≥ 3, Frequency ≥ 2, r² > 0.10)",
        "Density_Weighted": "Mean feature-pair r² within concepts ([concepts per features] ≥ 3, Frequency ≥ 2, r² > 0.10)",
        "Mean_Distance": "Mean graph distance between concepts",
    }

    # APA p-value formatter
    def format_apa_p(p):
        if pd.isna(p):
            return "p = n.s."
        if p < 0.001:
            return "p < 0.001"
        if p < 0.01:
            return "p < 0.01"
        return f"p = {p:.2f}"

    # Baseline SE for aggregated metrics (from concept-level variation)
    def baseline_se(series, how):
        vals = pd.to_numeric(series, errors="coerce").dropna().values
        n = len(vals)
        if n == 0:
            return np.nan
        if n == 1:
            return 0.0
        sd = float(np.std(vals, ddof=1))
        if how == "mean":
            return sd / np.sqrt(n)
        if how == "sum":
            return sd * np.sqrt(n)
        return np.nan

    # Clean labels
    base = baseline_cm.copy()
    boot = boot_cm.copy()
    base["Concept"] = base["Concept"].astype(str).str.strip()
    boot["Concept"] = boot["Concept"].astype(str).str.strip()

    if "iter" not in boot.columns:
        raise ValueError("boot_cm must contain an 'iter' column for dataset-level plotting.")

    def _safe(name):
        return re.sub(r"[^a-zA-Z0-9_.-]+","_", str(name)).strip("_")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Get dataset-level stats (for p-values)
    comp = compare_dataset_to_bootstrap(base, boot, agg_map=agg_map)

    for metric, how in agg_map.items():
        if metric not in base.columns or metric not in boot.columns:
            continue

        # Baseline aggregate
        if how == "sum":
            bval = pd.to_numeric(base[metric], errors="coerce").fillna(0).sum()
        elif how == "mean":
            bval = pd.to_numeric(base[metric], errors="coerce").dropna().mean()
        elif callable(how):
            bval = how(pd.to_numeric(base[metric], errors="coerce"))
        else:
            continue

        bse = baseline_se(base[metric], how)

        # Bootstrap aggregated distribution
        grp = boot.groupby("iter")[metric]
        if how == "sum":
            arr = grp.apply(lambda s: pd.to_numeric(s, errors="coerce").fillna(0).sum()).values
        elif how == "mean":
            arr = grp.apply(lambda s: pd.to_numeric(s, errors="coerce").dropna().mean()).values
        elif callable(how):
            arr = grp.apply(lambda s: how(pd.to_numeric(s, errors="coerce"))).values
        else:
            continue

        arr = pd.to_numeric(pd.Series(arr), errors="coerce").dropna().values
        if arr.size == 0 or pd.isna(bval):
            continue

        q025, q975 = np.quantile(arr, [0.025, 0.975])
        row = comp.loc[comp["metric"] == metric]
        apa = format_apa_p(row["pvalue"].iloc[0]) if not row.empty else "p = n.s."

        # Plot
        fig, ax = plt.subplots(figsize=(6,4))
        ax.hist(arr, bins=bins, alpha=0.9, color="#9b7ddc")  # histogram in #9b7ddc
        # 95% bootstrap CI band
        ax.axvspan(q025, q975, alpha=0.12)
        # Baseline ±1 SE band (if available)
        if not pd.isna(bse) and bse > 0:
            ax.axvspan(bval - bse, bval + bse, color="#E45756", alpha=0.18)
        # Baseline line in aesthetic red
        ax.axvline(bval, linestyle="--", linewidth=2, color="#E45756")

        # Labels
        ax.set_ylabel("Number of simulated datasets")
        #ax.xaxis.label.set_wrap(True)
        ax.set_xlabel(textwrap.fill(label_map.get(metric, metric), width=72))

        # ONLY the p-value, centered
        ax.text(0.5, 0.94, apa, ha="center", va="top", transform=ax.transAxes, fontsize=11)

        fig.tight_layout()
        fig.savefig(Path(output_dir) / f"{_safe(metric)}.png", dpi=dpi)
        plt.close(fig)

    return comp
# ============================
# Example usage (uncomment to run)
# ============================
# baseline_enriched, baseline_concept_metrics, r2_ff = compute_metrics(mrngo_unique_filtered, r2_thresh=0.10)
# boot_cm = bootstrap_vinson(vinson_filtered, k=1000, n_from=20, n_to=15, seed=42, r2_thresh=0.10)
# comp = compare_baseline_to_bootstrap(baseline_concept_metrics, boot_cm)
# plot_bootstrap_vs_baseline(boot_cm, baseline_concept_metrics, output_dir="bootstrap_outputs")

            
#%%
#baseline_enriched, baseline_concept_metrics, r2_ff = compute_metrics(mrngo_unique_filtered, r2_thresh=0.10)
#boot_cm = bootstrap_vinson(vinson_filtered, k=1000, n_from=20, n_to=15, seed=42, r2_thresh=0.10)
comp_dataset = plot_dataset_bootstrap(baseline_concept_metrics, boot_cm, output_dir="../figures/compare")
