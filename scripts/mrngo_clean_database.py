#%% PACKAGES
import json
import pandas as pd

#%% --- CONFIGURATION ---
JSON_IN   = '../raw/mrngo-data-all.json'
XLSX      = '../MRNGO_participants.xlsx'
JSON_OUT  = '../raw/mrngo-data-filtered.json'
ID_COLUMN = 'Document ID'   # <<— change this to the exact column name in your XLSX

#%% --- load valid IDs from Excel ---
df = pd.read_excel(XLSX)
valid_ids = set(df[ID_COLUMN].astype(str))

override_invalid = {
    'dEY7O3ymVSVzJ1aJv2FOihRooNs2',
    'e1h0NeMykXR9g70L9pXCNYLZwas1',
    'QaCckxsqxva7kDnDO7SOFey5L253'
}
valid_ids -= override_invalid

#%% --- load the JSON ---
with open(JSON_IN, 'r', encoding='utf-8') as f:
    data = json.load(f)

#%% --- filter answers in each concept ---
for category in data['database']['categories'].values():
    for concept in category['concepts'].values():
        original_answers = concept.get('answers', {})
        # keep only those answers whose key is in valid_ids
        filtered = {aid: ans
                    for aid, ans in original_answers.items()
                    if aid in valid_ids}

        removed = len(original_answers) - len(filtered)
        print(removed, len(original_answers), len(filtered))
        if removed:
            concept['answers'] = filtered
            # adjust the count
            concept['answer_count'] = len(filtered)
            
#%% METRICS
unique_ids = set()
total_responses = 0
concept_counts = {}

for category in data['database']['categories'].values():
    for concept_name, concept in category['concepts'].items():
        answers = concept.get('answers', {})
        total_responses += len(answers)
        for pid in answers.keys():
            unique_ids.add(pid)
            # increment the count of concepts this ID has answered
            concept_counts[pid] = concept_counts.get(pid, 0) + 1

print(f"Unique IDs remaining: {len(unique_ids)}")
print(f"Total responses across all concepts: {total_responses}")
print("Number of concepts answered per ID:")
print(concept_counts)

#%% --- write out filtered JSON ---
with open(JSON_OUT, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Filtered JSON written to {JSON_OUT}")
