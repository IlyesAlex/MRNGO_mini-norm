import pandas as pd

#%% 1) Load your long‐format EMTSV‐annotated table
df = pd.read_excel('../lemmatized/MRNGO_annotated_lemmas_emtsv.xlsx')

#%% 2) Merge in head‐token info so we can test its upos_part1 & lemma
dep_info = (
    df[['doc_id','upos_part1','lemma']]
      .rename(columns={
         'upos_part1':'dep_upos_part1',
         'lemma':'dep_lemma'
      })
)
df = df.merge(
    dep_info,
    how='left',
    left_on=['doc_id','dependency_token'],
    right_on=['doc_id','dep_lemma']
)
df = df.drop_duplicates()

#%% 3) Define the heuristic function for upos_part1 == 'Adj'
def compute_ADJ_vector(row, current):
    """Apply ADJ‐specific heuristics, return new vector_lemma or keep current."""
    if row['upos_part1'] != 'Adj':
        return current

    # omit manner‐advs modifying adjectives
    if row['upos_part3'] == '_Manner/Adv' and row['relation'] == 'amod:att':
        return current

    # omit if head is a non‐copula verb
    if row['dep_upos_part1'] == 'V' and row['dep_lemma'] not in ('van','tud'):
        return current

    # omit uninformative adjectives
    if row['lemma'] in ('található','színű','alakú','kis'):
        return current

    # recode typo "jármú" → FAJTA_JÁRMŰ
    if row['lemma'] == 'jármú':
        return 'FAJTA_' + 'jármű'.upper()

    # if head is the concept itself or ROOT → ILYEN_
    if row['dependency_token'] == row['concept'] or row['dependency_token'] == 'ROOT':
        return 'ILYEN_' + row['lemma'].upper()

    return current

def compute_NOUN_vector(row, current):
    """Apply NOUN‐specific heuristics, return new vector_lemma or keep current."""
    if row['upos_part1'] != 'N':
        return current

    # omit if it's the concept itself
    if row['lemma'] == row['concept']:
        return current

    # handle plural promotion
    upos2 = row['upos_part3'] if row['upos_part2'] == 'Pl' else row['upos_part2']

    # possessive‐3rd on van/ROOT → RÉSZE_
    if pd.notnull(row['upos_part2']) and 'Poss.3' in row['upos_part2'] and row.get('dependency_token') in ('van','ROOT'):
        return 'RÉSZE_' + row['lemma'].upper()

    # Poss.3Sg in a conjunction → RÉSZE_
    if row['upos_part2'] == 'Poss.3Sg' and row['relation'] == 'conj':
        return 'RÉSZE_' + row['lemma'].upper()

    # omit certain locative/ablative cases
    if upos2 in ('Abl','Ade','Dat','Del','Ess','Subl','Ter'):
        return current

    # accusative object of a verb → TOKEN_LEMMA
    if (upos2 == 'Acc'
        and row['dependency_token'] != 'ROOT'
        and row['relation'] == 'obj'
        and row['dep_upos_part1'] == 'V'):
        return f"{row['dependency_token'].upper()}_{row['lemma'].upper()}"

    # allative → LEMMA_TOKEN
    if upos2 == 'All':
        return f"{row['lemma'].upper()}_{row['dependency_token'].upper()}"

    # elative → ANYAG_LEMMA
    if upos2 == 'Ela':
        return 'MIBŐL_' + row['lemma'].upper()

    # illative or inessive → BENNE_LEMMA
    if upos2 in ('Ill','Ine'):
        return 'HOL_' + row['lemma'].upper()

    # nominative → FAJTA_LEMMA
    if upos2 == 'Nom':
        if row['dep_upos_part1'] == 'V':
            cluster = df[
                (df['doc_id'] == row['doc_id']) &
                (df['category'] == row['category']) &
                (df['concept'] == row['concept']) &
                (df["ID"] == row["ID"])
            ]
            has_poss_or_supe = (
                (cluster['upos_part1'] == 'N') &
                cluster['upos_part2'].str.contains('Poss|Supe', na=False)
            ).any()
            if has_poss_or_supe:
                return 'RÉSZE_' + row['lemma'].upper()
        
        # Otherwise, default to FAJTA_
        return 'FAJTA_' + row['lemma'].upper()

    # superessive → RAJTA_LEMMA
    if upos2 == 'Supe':
        return 'HOL_' + row['lemma'].upper()
    
    # dative → LEMMA_RÉSZE
    if upos2 == 'Supe':
        return row['lemma'].upper() + '_RÉSZE' 

    return current

def compute_VERB_vector(row, current):
    if row['upos_part1'] != 'V':
        return current
    # causative verbs
    if row['upos_part2'] == '_Caus/V':
        return 'LEHET_' + row['lemma'].upper() + '_MŰV'
    # infinitives
    if row['upos_part2'] == 'Inf':
        if row['dependency_token'] == 'ROOT':
            return 'TUD_' + row['lemma'].upper()
        if row['dependency_token'] == 'van':
            return 'LEHET_' + row['lemma'].upper()
        if row['dep_upos_part1'] == 'V' or row['dependency_token'] == 'kell':
            return f"{row['dependency_token'].upper()}_{row['lemma'].upper()}"
        
    if row['upos_part2'] == 'Prs.NDef.3Sg':
        return row['lemma'].upper() + "_RÉSZE"
        
    return current

df['vector_lemma'] = None

# 5) Apply noun rules first, then adjective rules
df['vector_lemma'] = df.apply(lambda r: compute_NOUN_vector(r, r['vector_lemma']), axis=1)
df['vector_lemma'] = df.apply(lambda r: compute_ADJ_vector(r, r['vector_lemma']), axis=1)
df['vector_lemma'] = df.apply(lambda r: compute_VERB_vector(r, r['vector_lemma']), axis=1)


# 6) Inspect results
print(
    df.loc[df['vector_lemma'].notnull(),
           ['doc_id','token','lemma','upos_part1','upos_part2',
            'upos_part3','relation','dependency_token','vector_lemma']]
    .head(10)
)

#%% EXPORT

# 7) Sort by ascending doc_id and export
df_sorted = df.sort_values(by='response_correct', ascending=True).reset_index(drop=True)

# Export to Excel or CSV as needed:
df_sorted.to_excel('../lemmatized/MRNGO_vector_lemmas_v3.xlsx', index=False)
