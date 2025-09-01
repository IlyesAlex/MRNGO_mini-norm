import pandas as pd
import unicodedata
import numpy as np
from itertools import combinations
from openpyxl import load_workbook

#%%
# Set the file paths; update input_file if needed
input_file = '../processed/MRNGO_manual_vector_lemmas_summary.xlsx'  # Path to your input Excel file
output_file = '../processed/MRNGO_manual_vector_lemmas_summary.xlsx'  # Path for the output Excel file

output_vector = '../processed/MRNGO_mini-norm_vectorspace.xlsx'
output_csv = "../processed/MRNGO_mini-norm_wordanalytics.csv"


def strip_accents(s: str) -> str:
    # decompose accents, then drop non-ASCII characters
    return (
        unicodedata.normalize('NFKD', s)
                   .encode('ascii', 'ignore')
                   .decode('ascii')
    )

input_file_mcrae = '../others/McRae-norm_filtered.xlsx'  # Path to your input Excel file
mcrae_filtered = pd.read_excel(input_file_mcrae)
mcrae_filtered_notax = mcrae_filtered[mcrae_filtered['WB_Maj'] != 'c']

mcrae_filtered_notax_matrix = mcrae_filtered_notax.pivot_table(
    index="Concept",
    columns="Feature",
    values="Prod_Freq",
    fill_value=0
)

concepts_to_keep = mcrae_filtered_notax["Concept"].unique()
#%% Read the raw data from the 'nyers' sheet
df = pd.read_excel(input_file, sheet_name='aggregated_data')
filtered_matrix = pd.read_excel("../processed/MRNGO_mini-norm_vectorspace.xlsx", sheet_name="mcrae_notax_vector_matrix")

df_unique = df.drop_duplicates(subset=['ID', 'concept', 'vector_lemma_C'])
df_unique['ID_Count'] = (df_unique.groupby(['vector_lemma_C', 'concept'])['ID'].transform('nunique'))
df_unique_filtered_ID = df_unique[df_unique['ID_Count'] >= 2]
df_unique_filtered_ID['Concept_Count'] = df_unique_filtered_ID['vector_lemma_C'].map((df_unique.groupby('vector_lemma_C')['concept'].nunique()))
df_unique_filtered = df_unique_filtered_ID[df_unique_filtered_ID['Concept_Count'] >= 3]
df_unique_filtered_notax = df_unique_filtered[df_unique_filtered['WB_Maj'] != 'c']
df_unique_filtered_notax_restricted = df_unique_filtered_notax[df_unique_filtered_notax["concept_EN"].isin(concepts_to_keep)]

#%% 1) Aggregate via category
df_cat = (
    df_unique
    .groupby(['category', 'concept', 'vector_lemma_C'])
    .size()
    .reset_index(name='frequency')
)

# 3) Aggregate via ID
df_id = (
    df_unique
    .groupby(['ID', 'category', 'vector_lemma_C'])
    .size()
    .reset_index(name='frequency')
)

# 3) Aggregate via vector
df_freq = (
    df_unique
    .groupby(['vector_lemma_C'])
    .size()
    .reset_index(name='frequency')
)

with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    df.to_excel(writer, sheet_name="aggregated_data", index=False)
    df_cat.to_excel(writer, sheet_name='category_based_summary', index=False)
    df_id.to_excel(writer, sheet_name='ID_based_summary', index=False)
    df_freq.to_excel(writer, sheet_name="categorize_features", index=False)

df_minimal = (
    df[['ID', 'concept', 'vector_lemma_C']]
    .rename(columns={
        'concept': 'Concept',
        'vector_lemma_C': 'Property'
    })
)

# apply to your two text-columns:
df_minimal['Concept']  = df_minimal['Concept'].apply(strip_accents)
df_minimal['Property'] = df_minimal['Property'].apply(strip_accents)

# now export normally
df_minimal.to_csv(output_csv, index=False, encoding='utf-8')

print(f'Kész! A feldolgozott adatot a "{output_file}" és a "{output_csv}" fájlok tartalmazzák.')


#%% Vector space

def build_matrix(df_src):
    counts = (
        df_src.groupby(['concept', 'vector_lemma_C'], as_index=False)
              .size()
              .rename(columns={'size': 'frequency'})
    )
    return (
        counts.pivot(index='concept', columns='vector_lemma_C', values='frequency')
              .fillna(0).astype(int).reset_index()
    )

# 1) All features (from df_unique)
full_matrix = build_matrix(df_unique)

# 2) Filtered by ID-support (from df_unique_filtered_ID)
filtered_matrix = build_matrix(df_unique_filtered_ID)

# 3) McRae (from df_unique_filtered)
filtered_matrix_mcrae = build_matrix(df_unique_filtered)

# 4) McRae No-Tax (WB_Maj != "c" within df_unique_filtered)
filtered_matrix_mcrae_notax = build_matrix(
    df_unique_filtered[df_unique_filtered['WB_Maj'] != 'c']
)

filtered_matrix_notax_restricted = build_matrix(df_unique_filtered_notax_restricted)

#%%
# --- Write sheets --------------------------------------------------------------
with pd.ExcelWriter(output_vector, engine='openpyxl') as writer:
    full_matrix.to_excel(writer, sheet_name='concept_vector_matrix', index=False)
    filtered_matrix.to_excel(writer, sheet_name='filtered_vector_matrix', index=False)
    filtered_matrix_mcrae.to_excel(writer, sheet_name='mcrae_vector_matrix', index=False)
    filtered_matrix_mcrae_notax.to_excel(writer, sheet_name='mcrae_notax_vector_matrix', index=False)
    filtered_matrix_notax_restricted.to_excel(writer, sheet_name='mcrae_notax_filtered_vector_matrix', index=False)

print("✅ Done. 4 sheets written:")
print("   • concept_vector_matrix")
print("   • filtered_vector_matrix")
print("   • mcrae_vector_matrix")
print("   • mcrae_notax_vector_matrix")

#%% 3) MCRAE indices
# Correlation
# 1) set 'concept' as the index
df_matrix = filtered_matrix_notax_restricted.set_index('concept')

# 2) drop any non-numeric columns (in case one slipped in)
numeric_matrix = df_matrix.select_dtypes(include=[int, float])

# 3) compute Pearson r and r²
corr = numeric_matrix.corr()
r2   = corr.pow(2)

# 4) collect all feature-pairs with shared variance ≥ 10%
min_shared_variance = 0.065
records = []
feats = corr.columns.tolist()
for i in range(len(feats)):
    for j in range(i+1, len(feats)):
        if r2.iat[i, j] >= min_shared_variance:
            records.append({
                'Feature1': feats[i],
                'Feature2': feats[j],
                'r':        corr.iat[i, j],
                'r2':       r2.iat[i, j]
            })
        else:
            records.append({
                'Feature1': feats[i],
                'Feature2': feats[j],
                'r':        corr.iat[i, j],
                'r2':       0.0
            })

high_corr = pd.DataFrame(records).sort_values('r2', ascending=False)

def enrich_all_features(df_raw: pd.DataFrame,
                        high_corr: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich a long‐form concept×feature DataFrame with:
      • frequency per (Concept,Feature)
      • Rank_PF, Sum_PF, CPF, Disting, Distinct, CV
      • string‐length metrics
      • Num_Feats, Num_Disting_Feats, Disting_Feats_%
      • Num_Corred_Pairs & Density, using an external high_corr table

    Parameters
    ----------
    df_raw : pd.DataFrame
        Must contain at least ['concept','vector_lemma_C'] (one row per response).
    high_corr : pd.DataFrame
        Must contain ['Feature1','Feature2', …] rows listing all feature‐pairs
        with r2 above your chosen threshold.

    Returns
    -------
    pd.DataFrame
        One row per (Concept,Feature), with all the added columns.
    """
    # — Step 0: count freq if needed —
    if 'frequency' not in df_raw.columns:
        df = (
            df_raw
            .groupby(['concept','vector_lemma_C'], dropna=False)
            .size()
            .reset_index(name='frequency')
        )
    else:
        df = df_raw.rename(columns={'vector_lemma_C':'vector_lemma_C'}).copy()

    # rename for clarity
    df = df.rename(columns={'concept':'Concept',
                            'vector_lemma_C':'Feature'})

    # 1) Rank_PF within each concept
    df['Rank_PF'] = (
        df
        .groupby('Concept')['frequency']
        .rank(method='dense', ascending=False)
        .astype(int)
    )

    # 2) Sum_PF = total freq per feature across all concepts
    sum_pf = df.groupby('Feature')['frequency'].sum()
    df['Sum_PF'] = df['Feature'].map(sum_pf)

    # 3) CPF = number of distinct concepts per feature
    cpf = df.groupby('Feature')['Concept'].nunique()
    df['CPF'] = df['Feature'].map(cpf)

    # 4) Disting / 5) Distinct / 6) CV
    df['Disting'] = np.where(df['CPF'] <= 2, 'D', 'ND')
    df['Distinct'] = 1.0 / df['CPF']
    df['CV'] = df['frequency'] / df['Sum_PF']

    # 7–10) Length metrics
    df['Feat_Length_Including_Spaces'] = df['Feature'].str.len()
    df['Length_Letters'] = (
        df['Feature']
        .str.replace(r'[^A-Za-z]', '', regex=True)
        .str.len()
    )
    vowels = 'aeiouAEIOU'
    df['Length_Syllables'] = df['Feature'].str.count(f'[{vowels}]+')
    df['Length_Phonemes']  = df['Length_Letters']

    # 11) Num_Feats / 12) Num_Disting_Feats / 13) Disting_Feats_%
    nf = df.groupby('Concept')['Feature'].count()
    df['Num_Feats'] = df['Concept'].map(nf)
    nd = df[df['Disting']=='D'].groupby('Concept')['Feature'].count()
    df['Num_Disting_Feats'] = df['Concept'].map(nd).fillna(0).astype(int)
    df['Disting_Feats_%'] = df['Num_Disting_Feats'] / df['Num_Feats']

    # 14) use high_corr to compute per-concept Num_Corred_Pairs & Density
    records = []

    # Ensure expected columns and keep only what's needed
    hc = high_corr.rename(columns={'Feature1': 'Feature1', 'Feature2': 'Feature2', 'r2': 'r2'})[
        ['Feature1', 'Feature2', 'r2']
    ].copy()

    # Compute total possible feature pairs in the DATASET (based on all features present in hc)
    all_feats = pd.Index(pd.unique(pd.concat([hc['Feature1'], hc['Feature2']])))
    total_pairs_dataset = int(len(all_feats) * (len(all_feats) - 1) / 2)

    for concept, sub in df.groupby('Concept'):
        feats = set(sub['Feature'])
        m = len(feats)

        # all pairs in hc that lie entirely within this concept's feature set
        mask_in_concept = hc['Feature1'].isin(feats) & hc['Feature2'].isin(feats)
        subpairs = hc[mask_in_concept]

        # non-zero shared variance pairs
        nz = subpairs[subpairs['r2'] > 0.0]

        # Num_Corred_Pairs_No_Tax: count of nz pairs
        num_pairs_no_tax = int(nz.shape[0])

        # Density (redefined): sum of r2 * 100 over non-zero pairs
        density_shared_var_pct = float((nz['r2'] * 100.0).sum())

        # %_Corred_Pairs_No_Tax: relative to ALL possible pairs in the dataset
        pct_corred_pairs_no_tax = 0.0 if total_pairs_dataset == 0 else num_pairs_no_tax / total_pairs_dataset

        records.append(
            (concept, num_pairs_no_tax, pct_corred_pairs_no_tax, density_shared_var_pct)
        )

    corr_df = pd.DataFrame.from_records(
        records,
        columns=['Concept', 'Num_Corred_Pairs_No_Tax', '%_Corred_Pairs_No_Tax', 'Density']
    ).set_index('Concept')

    # final merge
    df_final = df.merge(corr_df, left_on='Concept', right_index=True)

    return df_final

# ————————————— example usage —————————————

df_out = enrich_all_features(df_unique_filtered_notax_restricted, high_corr)

#%%
with pd.ExcelWriter(
    "../processed/MRNGO_manual_vector_lemmas_summary.xlsx",
    engine="openpyxl",
    mode="a",                    # <-- append mode
    if_sheet_exists="replace"    # replace the sheet if it already exists
) as writer:
    df_out.to_excel(writer, sheet_name="mcrae_metrics", index=False)

    

