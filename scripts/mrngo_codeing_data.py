import pandas as pd

#%% ─── 1) Load your annotated DataFrame ─────────────────────────────────────────
# adjust path as needed:
df = pd.read_excel("../lemmatized/MRNGO_annotated_lemmas.xlsx")
df["lemma"] = df["lemma"].fillna("").astype(str)


#%% ─── 2) Identify lemmas that ever occur as adjectives ────────────────────────
adj_lemmas = set(df.loc[df["upos"] == "ADJ", "lemma"])

#%% ─── 3) Define prefix logic ─────────────────────────────────────────────────
def compute_prefix(upos, lemma):
    # VERB/AUX (except "tud" and copulae) → "TUD_"
    if upos in ("VERB", "AUX") and lemma != "tud" and not (upos == "AUX" and lemma in ("van", "lesz", "volt")):
        return "TUD_"
    # ADJ or NOUN that also appears as ADJ → "ILYEN_"
    if upos == "ADJ" or (upos == "NOUN" and lemma in adj_lemmas):
        return "ILYEN_"
    # other NOUNs → "VAN_"
    if upos == "NOUN":
        return "VAN_"
    # everything else → no prefix
    return ""

#%% ─── 4) Build the unified lemma column ────────────────────────────────────────
df["prefix"]        = df.apply(lambda r: compute_prefix(r["upos"], r["lemma"]), axis=1)
df["lemma_unified"] = df["prefix"] + df["lemma"].str.upper()

cleaned_df = (
    df[df["prefix"] != ""]
      .drop(columns=["prefix"])  # drop helper if you like
      .reset_index(drop=True)
)

#%% ─── 5) Inspect or export ────────────────────────────────────────────────────
print(cleaned_df.head(10))
cleaned_df.to_excel("../lemmatized/MRNGO_vector_lemmas_v1.xlsx", index=False)
