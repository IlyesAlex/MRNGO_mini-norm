#%% FEATURE ELEMENT FREQUENCIES
import pandas as pd
from pathlib import Path


#%% PATHS
try:
    repo_dir = Path(__file__).resolve().parents[1]
except NameError:
    # Spyder fallback if this cell is executed without __file__.
    repo_dir = Path.cwd()

others_dir = repo_dir / "others"
kremer_file = others_dir / "Kremer-norm.xlsx"
mcrae_file = others_dir / "McRae-norm.xlsx"


#%% SETTINGS
pd.set_option("display.max_columns", 80)
pd.set_option("display.width", 180)

# Broad English preposition list. The key cases for these norms are things
# like used_for, used_by, used_on, made_of, found_in, lives_in, etc.
prepositions = {
    "aboard", "about", "above", "across", "after", "against", "along",
    "amid", "among", "around", "as", "at", "before", "behind", "below",
    "beneath", "beside", "between", "beyond", "by", "despite", "down",
    "during", "except", "for", "from", "in", "inside", "into", "like",
    "near", "of", "off", "on", "onto", "out", "outside", "over", "past",
    "through", "throughout", "to", "toward", "towards", "under",
    "underneath", "until", "up", "upon", "with", "within", "without",
}


#%% HELPERS
def load_feature_sheet(path, sheet_name, source_name):
    """Load one sheet and keep source metadata for combined inspection."""
    df = pd.read_excel(path, sheet_name=sheet_name)
    df["source"] = source_name
    df["sheet"] = sheet_name
    return df


def add_feature_parts(df, feature_col="Feature"):
    """
    Split Feature values on "_" and expose first/second elements.

    Leaves original Feature intact and creates an inspectable long-ish table:
    feature_first, feature_second, n_elements, is_multi_element,
    second_is_preposition.
    """
    out = df.copy()
    feature_clean = out[feature_col].astype("string").str.strip()
    parts = feature_clean.str.split("_")

    out["feature_clean"] = feature_clean
    out["feature_first"] = parts.str[0]
    out["feature_second"] = parts.str[1]
    out["n_elements"] = parts.str.len()
    out["is_multi_element"] = out["n_elements"].ge(2)
    out["second_is_preposition"] = (
        out["is_multi_element"]
        & out["feature_second"].str.lower().isin(prepositions)
    )
    out["feature_element"] = out["feature_first"].mask(
        out["second_is_preposition"],
        out["feature_first"] + "_" + out["feature_second"],
    )

    if "Prod_Freq" in out.columns:
        out["prod_freq_value"] = pd.to_numeric(out["Prod_Freq"], errors="coerce")
    elif "Prod.Frequency" in out.columns:
        out["prod_freq_value"] = pd.to_numeric(out["Prod.Frequency"], errors="coerce")
    else:
        out["prod_freq_value"] = pd.NA

    return out


def make_feature_element_frequency(df):
    """
    One frequency table for extracted Feature elements.

    If the second element is a preposition, the element is first_second
    (for example used_for, used_by, made_of). Otherwise, only the first
    element is used (for example has, is, made).

    Frequencies aggregate over raw Feature rows. n_raw_features tells how many
    distinct original Feature strings contributed to each extracted element.
    """
    freq = (
        df.groupby("feature_element", dropna=False)
        .agg(
            n_rows=("feature_clean", "size"),
            n_raw_features=("feature_clean", "nunique"),
            raw_features=("feature_clean", lambda x: "; ".join(sorted(x.dropna().unique()))),
            sources=("source", lambda x: "; ".join(sorted(x.dropna().unique()))),
            has_second_preposition=("second_is_preposition", "any"),
        )
        .reset_index()
        .sort_values(["n_rows", "feature_element"], ascending=[False, True])
        .reset_index(drop=True)
    )

    freq["source_coverage"] = freq["sources"].map(
        {
            "Kremer": "Kremer only",
            "McRae": "McRae only",
            "Kremer; McRae": "both",
        }
    )

    if df["prod_freq_value"].notna().any():
        weighted = (
            df.groupby("feature_element", dropna=False)["prod_freq_value"]
            .sum()
            .reset_index(name="prod_freq_sum")
        )
        freq = freq.merge(weighted, on="feature_element", how="left")

    return freq


#%% LOAD DATA
kremer_overall = load_feature_sheet(kremer_file, "overall", "Kremer")
kremer_responses_all = load_feature_sheet(kremer_file, "responses_all", "Kremer")
mcrae_raw = load_feature_sheet(mcrae_file, "raw", "McRae")


#%% ADD FEATURE ELEMENTS
kremer_overall_parts = add_feature_parts(kremer_overall)
kremer_responses_all_parts = add_feature_parts(kremer_responses_all)
mcrae_raw_parts = add_feature_parts(mcrae_raw)

all_feature_parts = pd.concat(
    [
        kremer_overall_parts,
        kremer_responses_all_parts,
        mcrae_raw_parts,
    ],
    ignore_index=True,
    sort=False,
)


#%% MAIN FREQUENCY TABLE
feature_element_frequency = make_feature_element_frequency(all_feature_parts)

del kremer_overall
del kremer_responses_all
del mcrae_raw
del kremer_overall_parts
del kremer_responses_all_parts
del mcrae_raw_parts
del all_feature_parts


#%% QUICK CONSOLE VIEWS
# This is intentionally not written to disk. Inspect this DataFrame in
# Spyder's Variable Explorer, or evaluate its name in the console.
feature_element_frequency
