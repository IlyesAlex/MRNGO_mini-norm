import os, json, re
from collections import defaultdict
from datetime import datetime

BASE_PATH = "C:/_ALEX_/RESEARCH/MRNGO/A_conceptnorm/data/"

RAW_JSON_PATH = BASE_PATH + "main_data/mrngo_raw.json"
PARTICIPANTS_XLSX_PATH = BASE_PATH + "main_data/mrngo_participants.xlsx"   # change if needed
OUT_PATH = BASE_PATH + "/main_data/mrngo_lite.json"

print("RAW_JSON_PATH exists:", os.path.exists(RAW_JSON_PATH))
print("PARTICIPANTS_XLSX_PATH exists:", os.path.exists(PARTICIPANTS_XLSX_PATH))

#%%
with open(RAW_JSON_PATH, "r", encoding="utf-8") as f:
    raw = json.load(f)

db = raw.get("database", {})
categories = db.get("categories", {}) or {}

num_categories = len(categories)
num_concepts = sum(len((cat or {}).get("concepts", {}) or {}) for cat in categories.values())

print(f"#categories: {num_categories}")
print(f"#concepts:   {num_concepts}")

# Optional: show one category id and count
for k, v in categories.items():
    print("Sample category:", k, "concepts:", len((v or {}).get("concepts", {}) or {}))
    break

#%%
participants_gender = {}  # uid -> 'male'|'female'|None

if os.path.exists(PARTICIPANTS_XLSX_PATH):
    try:
        import pandas as pd  # requires openpyxl for .xlsx
        df = pd.read_excel(PARTICIPANTS_XLSX_PATH)
        
        def strip_quotes_cell(x):
            if isinstance(x, str):
                return re.sub(r'^\s*"(.*)"\s*$', r'\1', x)
            return x
        
        df = df.applymap(strip_quotes_cell)

        # Find columns by name (case-insensitive)
        def norm(s): return str(s).strip().lower()
        cols = {norm(c): c for c in df.columns}

        id_col = cols.get("document id") or cols.get("document_id") or cols.get("uid") or cols.get("user id")
        gender_col = cols.get("gender") or cols.get("sex") or cols.get("nem")

        if id_col and gender_col:
            for _, row in df.iterrows():
                uid = str(row.get(id_col, "")).strip()
                if not uid:
                    continue
                g = str(row.get(gender_col, "")).strip().lower()
                if g in {"male", "m", "boy", "fiú"}:
                    participants_gender[uid] = "male"
                elif g in {"female", "f", "girl", "lány"}:
                    participants_gender[uid] = "female"
                else:
                    participants_gender[uid] = None

        print("Participants loaded:", len(participants_gender))
        # Show a few
        for i, (k, v) in enumerate(participants_gender.items()):
            print("  ", k, "→", v)
            if i >= 2: break

    except Exception as e:
        print("⚠️ Could not read participants XLSX:", e)
else:
    print("No participants XLSX found; male/female counts will be omitted.")

#%%
def safe_date_from_answer(answer_obj):
    """
    Return 'YYYY-MM-DD' derived from time_in_secs (ms) or time_of_save fallback.
    """
    ts = answer_obj.get("time_in_secs")
    if isinstance(ts, (int, float)) and ts > 0:
        try:
            return datetime.utcfromtimestamp(ts / 1000.0).strftime("%Y-%m-%d")
        except Exception:
            pass

    # Fallback: handle strings like "2025. 09. 24." or "2025. 09. 24. 9:22:10"
    tos = answer_obj.get("time_of_save", "")
    if isinstance(tos, str) and tos:
        m = re.search(r"(\d{4})\D+(\d{1,2})\D+(\d{1,2})", tos)
        if m:
            y, mo, d = map(int, m.groups())
            try:
                return datetime(y, mo, d).strftime("%Y-%m-%d")
            except Exception:
                pass
    return None

conceptIndex = {}

for cat_id, cat in categories.items():
    concepts = (cat or {}).get("concepts", {}) or {}
    for concept_id, concept in concepts.items():
        name = concept.get("concept_name")
        subcat = concept.get("subcategory")
        stored_count = concept.get("answer_count")

        # If stored_count is fine, use it; else recompute from answers
        answers = (concept or {}).get("answers", {}) or {}
        if isinstance(stored_count, int):
            answer_count = stored_count
        else:
            # recompute
            answer_count = sum(
                1 for uid, ans in answers.items()
                if uid != "default_answer" and isinstance(ans, dict) and (ans.get("time_in_secs") or ans.get("time_of_save"))
            )

        # Optional male/female split if participants_gender was loaded
        male_count = female_count = 0
        if participants_gender:
            for uid, ans in answers.items():
                if uid == "default_answer" or not isinstance(ans, dict):
                    continue
                g = participants_gender.get(uid)
                if g == "male":
                    male_count += 1
                elif g == "female":
                    female_count += 1

        entry = {
            "name": name,
            "subcategory": subcat,
            "answer_count": int(answer_count),
        }
        if participants_gender:
            entry["male_count"] = male_count
            entry["female_count"] = female_count

        conceptIndex[concept_id] = entry

print("conceptIndex concepts:", len(conceptIndex))
# Show a few samples
for i, (cid, meta) in enumerate(conceptIndex.items()):
    print(cid, "→", meta)
    if i >= 2: break


#%%
answersByUser = defaultdict(dict)               # uid -> {conceptId: time_in_secs or True}
subcatsToday = defaultdict(lambda: defaultdict(dict))  # uid -> day -> {subcat: True}
catsToday = defaultdict(lambda: defaultdict(dict))

for cat_id, cat in categories.items():
    concepts = (cat or {}).get("concepts", {}) or {}
    for concept_id, concept in concepts.items():
        subcat = (concept or {}).get("subcategory")
        answers = (concept or {}).get("answers", {}) or {}

        for uid, ans in answers.items():
            if uid == "default_answer" or not isinstance(ans, dict):
                continue

            ts = ans.get("time_in_secs")
            if isinstance(ts, (int, float)) and ts > 0:
                answersByUser[uid][concept_id] = int(ts)
            elif ans.get("time_of_save"):
                answersByUser[uid][concept_id] = True  # no numeric ts, but mark as answered
            else:
                continue  # skip empty placeholders

            day = safe_date_from_answer(ans)
            if day and subcat:
                subcatsToday[uid][day][str(subcat)] = True

            # NEW: mark the CATEGORY for this day too (blocks repeating category in a day)
            if day and cat_id:
                catsToday[uid][day][str(cat_id)] = True


# Peek at sizes
total_users = len(answersByUser)
total_subcats_users = len(subcatsToday)
total_cats_users = len(catsToday)

print("answersByUser users:", total_users)
print("subcatsToday users:", total_subcats_users)
print("catsToday users:", total_cats_users)  # NEW

# Show a couple examples
for i, (u, m) in enumerate(answersByUser.items()):
    some = list(m.items())[:5]
    print("User:", u, "answered keys (first 5):", [k for k, _ in some])
    if i >= 1: break

for i, (u, days) in enumerate(subcatsToday.items()):
    if i > 1: break
    print("User (subcats):", u)
    for j, (day, subs) in enumerate(days.items()):
        print("  ", day, "→ subcats:", list(subs.keys())[:8])
        if j >= 1: break

# NEW: peek categories per day
for i, (u, days) in enumerate(catsToday.items()):
    if i > 1: break
    print("User (cats):", u)
    for j, (day, cats) in enumerate(days.items()):
        print("  ", day, "→ cats:", list(cats.keys())[:8])
        if j >= 1: break


#%%
def deep_convert(d):
    if isinstance(d, defaultdict):
        return {k: deep_convert(v) for k, v in d.items()}
    if isinstance(d, dict):
        return {k: deep_convert(v) for k, v in d.items()}
    return d

lite = {
    "conceptIndex": conceptIndex,
    "answersByUser": deep_convert(answersByUser),
    "subcatsToday": deep_convert(subcatsToday),
    # NEW:
    "catsToday": deep_convert(catsToday),
}

dump = json.dumps(lite, ensure_ascii=False, separators=(",", ":"))
print("Lite JSON bytes (approx, uncompressed):", len(dump))
print("Top-level keys:", list(lite.keys()))

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(lite, f, ensure_ascii=False, indent=2, sort_keys=False)
print("Wrote:", OUT_PATH, "size(bytes):", os.path.getsize(OUT_PATH))


