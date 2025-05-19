import pandas as pd
import unicodedata
import numpy as np

keep_concepts    = [
    "apple",
    "table", "desk",
    "car",
    "bean", "beans",
    "pencil",
    "shoe", "shoes",
    "lemon",
    "strawberry",
    "forest",
    "mushroom",
    "child", "children", "kid",
    "helicopter",
    "princess",
    "playground",
    "duck",
    "sword",
    "potato",
    "dog",
    "hand", "hands",
    "book",
    "leg", "legs",
    "horse",
    "pants", "trousers",
    "living room", "livingroom", "living-room",
    "sweater", "jumper", "pullover",
    "broom",
    "eye", "eyes",
    "carpet", "rug",
    "firefighter",
    "train"
]  # your EN concepts to keep


def compute_richness_scores(long_df: pd.DataFrame,
                               T: float = 280) -> pd.DataFrame:
    """
    Given a DataFrame of response-vectors (rows = items, cols = values 0/1/2),
    compute for each row:
      S_obs = number of non-zero cells
      Q1    = number of 1’s
      Q2    = number of 2’s
      S_hat = adjusted score per Richness

    Parameters
    ----------
    df : pd.DataFrame
        Rows are cases (e.g. concepts), columns are features (values in {0,1,2}).
    T : float, default 280
        Total number of participants

    Returns
    -------
    pd.DataFrame
        Indexed like df, with columns [S_obs, Q1, Q2, S_hat].
    """

    # 2) Pivot back to wide (missing combos → 0)
    wide = long_df.pivot_table(
        index='Concept',
        columns='Feature',
        values='Frequency',
        fill_value=0
    )

    # 3) Precompute constant
    A = (T - 1) / T

    # 4) Compute S_obs, Q1, Q2
    S_obs = (wide != 0).sum(axis=1)
    Q1    = (wide == 1).sum(axis=1)
    Q2    = (wide == 2).sum(axis=1)

    # 5) Compute S_hat with vectorized np.where
    S_hat = np.where(
        Q2 >  0,
        S_obs + A * (Q1**2) / (2 * Q2),
        S_obs + A * (Q1 * (Q1 - 1) / 2)
    )

    # 6) Assemble result
    scores = pd.DataFrame({
        'S_obs': S_obs,
        'Q1':    Q1,
        'Q2':    Q2,
        'S_hat': S_hat
    }, index=wide.index)

    return scores

def long_format_vectors(target_df):
    # 2) Name your index
    target_df.index.name = "Concept"
    
    # 3) Melt (stack) into long form
    long = (
        target_df
        .stack()                       # -> MultiIndex: (Concept, Feature)
        .reset_index(name="Frequency") # back to columns, values in ‘Frequency’
    )
    
    # 4) Rename the auto‐generated column names
    long = long.rename(columns={"level_1": "Feature"})
    
    # 5) Keep only non‐zero frequencies
    long = long.loc[long["Frequency"] > 0]
    
    # 6) (Opció) Ha duplikátum lehet, össze is gyúrhatod:
    long = (
        long
        .groupby(["Concept", "Feature"], as_index=False)
        ["Frequency"]
        .sum()
    )
    return long


#%% KREMER
# Set the file paths; update input_file if needed
input_file_kremer = '../others/Kremer-norm.xlsx'  # Path to your input Excel file
output_file_kremer = '../others/Kremer-norm_filtered.xlsx'  # Path for the output Excel file

kremer_sheet_names = ["overall", "responses_all"]

# 2) load both sheets
freq_df   = pd.read_excel(input_file_kremer, sheet_name=kremer_sheet_names[0])
phrase_df = pd.read_excel(input_file_kremer, sheet_name=kremer_sheet_names[1])

# 3) for each nationality, filter & dump
for nat in freq_df['Nationality'].unique():
    # subset by nationality + concept filter
    f1 = freq_df[(freq_df['Nationality'] == nat) &
                 (freq_df['Concept(EN)'].isin(keep_concepts))]
    f2 = phrase_df[(phrase_df['Nationality'] == nat) &
                   (phrase_df['Concept(EN)'].isin(keep_concepts))]

    # 4a) write the filtered Excel
    out_xlsx = f'../others/Kremer_norm_filtered-{nat}.xlsx'
    with pd.ExcelWriter(out_xlsx, engine='openpyxl') as writer:
        f1.to_excel(writer, sheet_name='frequency', index=False)
        f2.to_excel(writer, sheet_name='phrases',   index=False)
    print(f'→ Wrote {out_xlsx}')

    # 4b) minimal CSV from the phrase‐table
    csv_df = (
        f2
        .rename(columns={
            'SubjectCode': 'ID',
            'Concept(EN)' : 'Concept',
            'Feature'     : 'Property'
        })
        [['ID','Concept','Property']]
    )
    out_csv = f'../others/Kremer_norm_filtered-{nat}.csv'
    csv_df.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f'→ Wrote {out_csv}')


#%% VINSON
vinson_norm_vectors = pd.read_excel("../others/Vinson-norm.xlsx", index_col=0).transpose()

vinson_long = long_format_vectors(vinson_norm_vectors)
vinson_long = vinson_long[vinson_long['Concept'].map(lambda x: x.lower()).isin(keep_concepts)]
vinson_scores = compute_richness_scores(vinson_long, T=280)

with pd.ExcelWriter("../others/Vinson-norm_filtered.xlsx", engine='openpyxl') as writer:
    vinson_long.to_excel(writer, sheet_name='features',  index=False)
    vinson_scores.to_excel(writer, sheet_name='richness',  index=True)


#%% MCRAE
mcrae_norm_longformat = pd.read_excel("../others/McRae-norm.xlsx")

mcrae_norm_filtered = mcrae_norm_longformat[mcrae_norm_longformat['Concept'].map(lambda x: x.lower()).isin(keep_concepts)]

with pd.ExcelWriter("../others/McRae-norm_filtered.xlsx", engine='openpyxl') as writer:
    mcrae_norm_filtered.to_excel(writer, sheet_name='features',  index=False)
