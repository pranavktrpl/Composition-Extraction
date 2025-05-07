import re, unicodedata, json, joblib, numpy as np, pandas as pd, umap, hdbscan
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from pathlib import Path

# ---------- USER CONFIG ----------
INPUT_JSON = "compositions.json"        # a JSON file with {"compositions": [ ... ]}
VEC_PKL    = "vec.pkl"
UMAP_PKL   = "umap.pkl"
HDBSCAN_PKL= "hdbscan.pkl"
FLAG_PKL   = "flag_cols.pkl"
MIN_DF     = 3
N_NEIGHB   = 25
MIN_DIST   = 0.1
UMAP_DIM   = 10
MIN_CLUST  = 15
# ----------------------------------

SUB_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
FLAG_COLS = ["has_pct","has_wt","has_vol","has_at","colon","slash","many_dashes","paren","bracket"]

def canon(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = s.translate(SUB_MAP).replace("–","-").replace("—","-")
    s = re.sub(r"\s+"," ", s).strip()
    return s

def flags(s: str):
    return [
        "%" in s,
        "wt%" in s.lower(),
        "vol%" in s.lower(),
        "at%" in s.lower() or "mol%" in s.lower(),
        ":" in s,
        "/" in s,
        s.count("-") > 1,
        "(" in s and ")" in s,
        "[" in s
    ]

# ---------- LOAD & CLEAN ----------
with open(INPUT_JSON) as f:
    data = json.load(f)["compositions"]
df = pd.DataFrame({"raw": data})
df["canon"] = df["raw"].apply(canon)
df[FLAG_COLS] = df["canon"].apply(lambda x: pd.Series(flags(x))).astype(float)

# ---------- VECTORIZE ----------
vec = TfidfVectorizer(analyzer="char", ngram_range=(3,5), min_df=MIN_DF)
X_text = vec.fit_transform(df["canon"])
X = hstack([X_text, df[FLAG_COLS].values])

# ---------- DIM REDUCTION ----------
umap_model = umap.UMAP(
    n_neighbors=N_NEIGHB,
    min_dist=MIN_DIST,
    n_components=UMAP_DIM,
    metric="cosine",
    random_state=0
).fit(X)

X_embed = umap_model.transform(X)

# ---------- CLUSTER ----------
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=MIN_CLUST,
    metric="euclidean"
).fit(X_embed)

df["cluster_id"] = clusterer.labels_

print("Clusters discovered:", df["cluster_id"].unique())
print(df.groupby("cluster_id").size())

# ---------- SAVE ARTIFACTS ----------
joblib.dump(vec, VEC_PKL)
joblib.dump(umap_model, UMAP_PKL)
joblib.dump(clusterer, HDBSCAN_PKL)
joblib.dump(FLAG_COLS, FLAG_PKL)

# Save a CSV snapshot for manual bucket naming
df.to_csv("clustered_compositions.csv", index=False)

print("Vectorizer, UMAP reducer, HDBSCAN model, and flag list saved.")
