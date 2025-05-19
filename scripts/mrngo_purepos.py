import requests
import pandas as pd
from io import StringIO

def emtsv_annotate(text, tools=('tok','morph','pos', 'ner', 'spell'), host='localhost', port=5000):
    """
    Sends `text` to your local emtsv Docker container
    and returns a DataFrame with columns like FORM, LEMMA, UPOS, XPOS, FEATS.
    Note: the REST‐API endpoint uses slashes between tools, not commas :contentReference[oaicite:0]{index=0}.
    """
    endpoint = "/".join(tools)
    url      = f"http://{host}:{port}/{endpoint}"
    r        = requests.post(url, data={'text': text})
    r.raise_for_status()
    return pd.read_csv(
        StringIO(r.text),
        sep="\t",
        comment="#",
        quoting=3,          # QUOTE_NONE
        keep_default_na=False
    )


#%% ─── 1) Load your annotated DataFrame ─────────────────────────────────────────
df = pd.read_excel("../lemmatized/MRNGO_annotated_lemmas.xlsx")
df   = df.iloc[:, :-3]
df = df.drop_duplicates().reset_index(drop=True)


#%% --- 3) For each row, annotate its "response" text and build a long‐format DF ---
records = []
for _, row in df.iterrows():
    text = row["response_correct"]  # adjust column name if needed
    ann  = emtsv_annotate(text, tools=('tok','morph','pos', 'conv-morph2', 'dep', 'udpipe-parse', 'spell'))
    print(ann)
    # ann columns: ID, FORM, LEMMA, UPOS, XPOS, FEATS, etc.
    for _, tok in ann.iterrows():
        rec = row.to_dict()
        rec.update({
            "token": tok["form"],
            "lemma_id": tok["id"],
            "lemma": tok["lemma"],
            "dependency": tok["head.1"],
            "relation": tok["deprel.1"],
            "upos":  tok["xpostag"],
            "hunspell": tok["hunspell_anas"]
        })
        records.append(rec)

long_df = pd.DataFrame(records)

#%% EXTRACT
# 1) extract *all* bracketed tokens (dropping the optional leading slash)
upos_lists = long_df["upos"].str.findall(r'\[/?([^]]+)\]')

# 2) find how many “slots” you need (the maximum list‐length)
max_parts = upos_lists.map(len).max()

# 3) turn those lists into a small DataFrame with one column per part
upos_cols = pd.DataFrame(
    upos_lists.tolist(),
    index=long_df.index,
    columns=[f"upos_part{i+1}" for i in range(max_parts)]
)

# 4) join it back to your original df
long_df = pd.concat([long_df, upos_cols], axis=1)

# 5) now you have e.g. upos_part1, upos_part2, upos_part3, … for every row
print(long_df[["upos"] + upos_cols.columns.tolist()].head())

#%% DEPENDENCY
# 1. Load your annotated file

# 2. Make sure these are the column names you use:
#    - sentence identifier: 'doc_id'
#    - token id:            'id'
#    - head index:          'dependency'
#    - semantic label:      'concept'
#    Adjust the names below if they differ.
sent_col   = 'doc_id'
id_col     = 'lemma_id'
head_col   = 'dependency'
label_col  = 'lemma'

# 3. Build a per-sentence map from token ID → concept
maps = {
    sent: grp.set_index(id_col)[label_col].to_dict()
    for sent, grp in long_df.groupby(sent_col)
}

# 4. Replace each dependency index with the corresponding concept (or ROOT)
def resolve_dep(row):
    dep_idx = row[head_col]
    if dep_idx == 0:
        return 'ROOT'
    return maps[row[sent_col]].get(dep_idx, 'ROOT')

# 5. Compute the new column as a Series
dep_tokens = long_df.apply(resolve_dep, axis=1)

# 6. Insert it right after the numeric `dependency` column
dep_idx = long_df.columns.get_loc(head_col)
long_df.insert(dep_idx + 1, 'dependency_token', dep_tokens)


#%% 4 EXPORT
long_df.to_excel("../lemmatized/MRNGO_annotated_lemmas_emtsv.xlsx", index=False)

#%% PROBA

x = emtsv_annotate("kisebb, nagyobb kutyák is vannak", tools=('tok','morph','pos', 'conv-morph2', 'dep', 'udpipe-parse'))