import pandas as pd
import unicodedata
import numpy as np
from itertools import combinations
from openpyxl import load_workbook


# Set the file paths; update input_file if needed
input_file = '../processed/MRNGO_manual_vector_lemmas_summary_ORIG.xlsx'  # Path to your input Excel file
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


#%% Read the raw data from the 'nyers' sheet
df = pd.read_excel(input_file, sheet_name='nyers')
filtered_matrix = pd.read_excel("../processed/MRNGO_mini-norm_vectorspace.xlsx", sheet_name="mcrae_vector_matrix")

#%% 1) Aggregate via category
df_cat = (
    df
    .groupby(['category', 'concept', 'vector_lemma_C'])
    .size()
    .reset_index(name='frequency')
)

# 3) Aggregate via ID
df_id = (
    df
    .groupby(['ID', 'category', 'vector_lemma_C'])
    .size()
    .reset_index(name='frequency')
)

# 3) Aggregate via vector
df_freq = (
    df
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

# 2) count each (concept, lemma) pair
df_counts = (
    df
    .groupby(['concept', 'vector_lemma_C'])
    .size()
    .reset_index(name='frequency')
)

# 3) pivot to get the full concept×lemma matrix
full_matrix = (
    df_counts
    .pivot(index='concept',
           columns='vector_lemma_C',
           values='frequency')
    .fillna(0)
    .astype(int)
    .reset_index()
)

# 4) find which lemmas occur only once in total
lemma_totals = df_counts.groupby('vector_lemma_C')['frequency'].sum()
keep_lemmas = lemma_totals[lemma_totals > 1].index

# 5) build a filtered matrix (drop any lemma-col with total freq == 1)
filtered_matrix = full_matrix.loc[
    :, ['concept'] + [c for c in full_matrix.columns if c in keep_lemmas]
]

# count in how many unique concepts each lemma appears
lemma_concept_counts = df_counts.groupby('vector_lemma_C')['concept'] \
                                 .nunique()

# keep only those lemmas that appear in 3 or more concepts
keep_lemmas_mcrae = lemma_concept_counts[lemma_concept_counts >= 3].index

# 5) build a filtered matrix (drop any lemma-col with total freq == 1)
filtered_matrix_mcrae = full_matrix.loc[
    :, ['concept'] + [c for c in full_matrix.columns if c in keep_lemmas_mcrae]
]

# 6) write both sheets into one Excel file
with pd.ExcelWriter(output_vector, engine='openpyxl') as writer:
    full_matrix.to_excel(writer,
                         sheet_name='concept_vector_matrix',
                         index=False)
    filtered_matrix.to_excel(writer,
                              sheet_name='filtered_vector_matrix',
                              index=False)
    filtered_matrix_mcrae.to_excel(writer,
                              sheet_name='mcrae_vector_matrix',
                              index=False)

print("✅ Done. 2 sheets written to processed_data.xlsx:")
print("   • concept_vector_matrix  (all features)")
print("   • filtered_features      (drops lemmas with total freq=1)")

#%% 3) MCRAE indices
# Correlation
# 1) set 'concept' as the index
df_matrix = filtered_matrix.set_index('concept')

# 2) drop any non-numeric columns (in case one slipped in)
numeric_matrix = df_matrix.select_dtypes(include=[int, float])

# 3) compute Pearson r and r²
corr = numeric_matrix.corr()
r2   = corr.pow(2)

# 4) collect all feature-pairs with shared variance ≥ 10%
min_shared_variance = 0.1
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
    # make sure high_corr uses same column names
    hc = high_corr.rename(columns={'Feature1':'Feature1','Feature2':'Feature2'})
    for concept, sub in df.groupby('Concept'):
        feats = set(sub['Feature'])
        m = len(feats)
        # count how many high_corr pairs fall inside this concept's feature set
        mask1 = hc['Feature1'].isin(feats)
        mask2 = hc['Feature2'].isin(feats)
        num_pairs = int((hc[mask1 & mask2].shape[0]))
        density = 0.0 if m < 2 else num_pairs / (m*(m-1)/2)
        records.append((concept, num_pairs, density))

    corr_df = pd.DataFrame.from_records(
        records,
        columns=['Concept','Num_Corred_Pairs','Density']
    ).set_index('Concept')

    # final merge
    df_final = df.merge(corr_df,
                        left_on='Concept',
                        right_index=True)

    return df_final

# ————————————— example usage —————————————

df_out = enrich_all_features(df, high_corr)


with pd.ExcelWriter(
    "../processed/MRNGO_manual_vector_lemmas_summary.xlsx",
    engine="openpyxl",
    mode="a",                    # <-- append mode
    if_sheet_exists="replace"    # replace the sheet if it already exists
) as writer:
    df_out.to_excel(writer, sheet_name="mcrae_metrics", index=False)

    

