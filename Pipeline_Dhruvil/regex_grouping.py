import re, pandas as pd, json

patterns = [
  ("simple", re.compile(r'^[A-Z][a-z]?\d*(?:\.\d+)?([A-Z][a-z]?\d*(?:\.\d+)?)*$')),
  ("variable_x", re.compile(r'[+-]x|[+-]d|_[xyz]|\b[A-Za-z]\s*=\s*x')),
  ("dash_glass", re.compile(r'\d.*(?:–|-)\d')),
  ("%_doped",  re.compile(r'\d+(\.\d+)?\s*(wt|vol|at|mol)?\s*%')),
  ("colon",    re.compile(r'.+?:.+')),
  ("slash",    re.compile(r'.+/.+')),
  ("parenthetical", re.compile(r'\([A-Za-z]{2,}.*?\)')),
#   ("hard_particle", re.compile(r'(MWNT|B4C|TiB2)')),
#   ("organic",  re.compile(r'(poly|peek|oil|carbonate|glyc|arylite|pet)', re.I)),
#   ("alloy_grade", re.compile(r'(EUROFER|F82H|SS3|SM5|Heraeus|DuPond|Ferro)')),
  ("complex_mol", re.compile(r'\[.*\]')),
]

tags_list = [tag for tag, pat in patterns]
tags_list.append("misc")

def classify(s):
    for tag, pat in patterns:
        if pat.search(s):
            return tag
    return "misc"

# Read the JSON file
with open("Dhruvil-Model Results/final_results.json", 'r') as f:
    data = json.load(f)

# Extract unmatched compositions and create DataFrame
unmatched_compositions = data['unmatched_compositions']
df = pd.DataFrame(unmatched_compositions)

# Apply classification
df['cluster'] = df['composition'].apply(classify)

# Save results
df.to_csv("unmatched_compositions_analysis.csv", index=False)

# Print statistics
print("\nAnalysis of Unmatched Compositions:")
print("=" * 50)
print("\nCluster Distribution:")
print(df["cluster"].value_counts())
print("\nTotal unmatched compositions:", len(df))