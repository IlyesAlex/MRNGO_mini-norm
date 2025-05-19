#%% PAKAGES
import json
import pandas as pd
import datetime
import re
from dateutil import parser


#%% --- CONFIGURATION ---
FILTERED_JSON      = '../raw/mrngo-data-filtered.json'
PARTICIPANTS_XLSX  = '../MRNGO_participants.xlsx'
OUTPUT_XLSX        = '../preprocessed/MRNGO_preprocessed.xlsx'

#%% column names in the participants file
ID_COLUMN          = 'Document ID'   # your ID column
BIRTHDATE_COLUMN   = 'birthdate'        # must be in YYYY-MM-DD or pandas‐parseable format
GENDER_COLUMN      = 'gender'
SCHOOL_COLUMN      = 'school'
GRADE_COLUMN       = 'grade'

#%% --- load participants ---
df_part = pd.read_excel(PARTICIPANTS_XLSX, dtype={ID_COLUMN: str})
df_part.set_index(ID_COLUMN, inplace=True)

#%% --- load filtered JSON ---
with open(FILTERED_JSON, 'r', encoding='utf-8') as f:
    data = json.load(f)

# determine today's date (in Europe/Budapest your system time should reflect this)
today = datetime.datetime.now().date()
rows = []

def parse_birthdate(raw):
    """Try parsing with dateutil; if it fails, clean up common separators and try again."""
    try:
        return parser.parse(raw, dayfirst=False, yearfirst=True).date()
    except (parser.ParserError, TypeError, ValueError):
        # replace all non-digit characters with a dash
        cleaned = re.sub(r'[^0-9]', '-', str(raw))
        # collapse multiple dashes
        cleaned = re.sub(r'-+', '-', cleaned).strip('-')
        return parser.parse(cleaned, dayfirst=False, yearfirst=True).date()

#%% --- iterate through the JSON structure ---
for category_name, category in data['database']['categories'].items():
    for concept_name, concept in category['concepts'].items():
        for pid, answer_obj in concept.get('answers', {}).items():
            # answer_obj is a dict, e.g.
            # { "answer_content": "...", "answer_username": "...", ... }
            if pid not in df_part.index:
                continue

            # participant metadata
            part = df_part.loc[pid]
            try:
                bd = parse_birthdate(part[BIRTHDATE_COLUMN])
            except Exception as e:
                print(f"Skipping {pid}: bad birthdate {part[BIRTHDATE_COLUMN]} – {e}")
                continue
            age = today.year - bd.year - ((today.month, today.day) < (bd.month, bd.day))

            gender    = part[GENDER_COLUMN]
            education = part[SCHOOL_COLUMN]
            grade_str = str(part[GRADE_COLUMN])
            m = re.match(r'(\d+)', grade_str)
            grade = int(m.group(1)) if (m := re.match(r'(\d+)', str(part[GRADE_COLUMN]).replace('"','').replace('.',''))) else None

            # **Key change**: extract the actual text field
            content = answer_obj.get('answer_content', '')
            for resp in content.split(';'):
                resp = resp.strip()
                if not resp:
                    continue

                rows.append({
                    'ID':        pid,
                    'age':       age,
                    'sex':       gender,
                    'education': education,
                    'grade':     grade,
                    'category':  category_name,
                    'concept':   concept["concept_name"],
                    'type':      'text',
                    'response':  resp
                })
                

#%% --- build DataFrame and export ---
df_out = pd.DataFrame(rows)
df_out.to_excel(OUTPUT_XLSX, index=False)

print(f"Preprocessed data written to {OUTPUT_XLSX} ({len(df_out)} rows)")

#%% METRICS
# Create an age_sex_grade string for each row
df_out['age_sex_grade'] = (
    df_out['age'].astype(str) + "_" +
    df_out['sex'].astype(str) + "_" +
    df_out['grade'].astype(str)
)

# Total unique age_sex_grade groups and total responses
unique_groups = df_out['age_sex_grade'].nunique()
total_responses = len(df_out)

# Count distinct concepts answered by each age_sex_grade group
concept_counts = (
    df_out[['age_sex_grade', 'concept']]
    .drop_duplicates()
    .groupby('age_sex_grade')
    .size()
    .to_dict()
)

print(f"Unique age_sex_grade groups: {unique_groups}")
print(f"Total responses across all concepts: {total_responses}")
print("Number of concepts answered per age_sex_grade:")
print(concept_counts)


#%%
df_out_test = df_out.loc[:,["ID", "concept"]]
df_out_test = df_out_test.drop_duplicates().reset_index(drop=True)
counts = df_out_test["ID"].value_counts()
print(counts)