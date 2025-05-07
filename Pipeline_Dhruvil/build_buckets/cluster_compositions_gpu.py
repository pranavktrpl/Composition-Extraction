import re, unicodedata, json, joblib, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from pathlib import Path
import os
import time

# Import GPU accelerated libraries from RAPIDS
try:
    import cudf
    import cuml
    from cuml.manifold import UMAP
    from cuml.cluster import HDBSCAN
    from cupyx.scipy.sparse import csr_matrix as cu_csr_matrix
    HAS_CUDA = True
    print("Using GPU acceleration with RAPIDS cuML")
except ImportError:
    import umap
    import hdbscan
    HAS_CUDA = False
    print("RAPIDS cuML not found. Using CPU implementation.")
    print("To install GPU acceleration: pip install cudf-cu12 cuml-cu12 cuspatial-cu12 cupy-cuda12x")

# ---------- USER CONFIG ----------
INPUT_JSON = "compositions.json"      # a JSON file with {"compositions": [ ... ]}
OUTPUT_DIR = Path("clusters_gpu")     # Directory for output files
VEC_PKL    = "vec.pkl"
UMAP_PKL   = "umap.pkl"
HDBSCAN_PKL= "hdbscan.pkl"
FLAG_PKL   = "flag_cols.pkl"
MIN_DF     = 3
N_NEIGHB   = 25
MIN_DIST   = 0.1
UMAP_DIM   = 10
MIN_CLUST  = 15
MIN_SAMPLES = 10
# ----------------------------------

FLAG_COLS = ["has_pct","has_wt","has_vol","has_at","has_mol","has_colon","has_slash","many_dashes","paren","bracket"]

def canon(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    subs = str.maketrans("₀₁₂₃₄₅₆₇₈₉⁰¹²³⁴⁵⁶⁷⁸⁹", "01234567890123456789")
    s = s.translate(subs)
    s = s.replace("–", "-").replace("—", "-")   # dash unification
    s = re.sub(r"\s+", " ", s).strip()
    # optional glass chatter
    s = re.sub(r"\bglass\b", "", s, flags=re.I)
    return s

def flags(s: str):
    return [
        "%" in s,
        "wt%" in s.lower(),
        "vol%" in s.lower(),
        "at%" in s.lower(),
        "mol%" in s.lower(),
        ":" in s,
        "/" in s,
        s.count("-") > 1,
        "(" in s and ")" in s,
        "[" in s
    ]

def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # ---------- LOAD & CLEAN ----------
    print(f"Loading composition data from {INPUT_JSON}...")
    try:
        with open(INPUT_JSON) as f:
            data = json.load(f)
            compositions = data["compositions"]
            num_piis = data.get("source_piis", "unknown")
        print(f"Loaded {len(compositions)} compositions from {num_piis} PIIs")
    except FileNotFoundError:
        print(f"Error: {INPUT_JSON} not found. Run process_compositions.py first.")
        return
    except KeyError:
        print(f"Error: {INPUT_JSON} doesn't contain a 'compositions' key.")
        return
    
    df = pd.DataFrame({"raw": compositions})
    df["canon"] = df["raw"].apply(canon)
    df[FLAG_COLS] = df["canon"].apply(lambda x: pd.Series(flags(x))).astype(float)
    
    # ---------- VECTORIZE ----------
    print("Vectorizing data...")
    vec = TfidfVectorizer(
        analyzer="char", 
        ngram_range=(3,5), 
        min_df=MIN_DF,
        sublinear_tf=True,
    )
    X_text = vec.fit_transform(df["canon"])
    X = hstack([X_text, df[FLAG_COLS].values])
    
    # ---------- DIM REDUCTION ----------
    print("Applying UMAP dimensionality reduction...")
    start_time = time.time()
    
    if HAS_CUDA:
        # Convert scipy sparse matrix to GPU sparse matrix
        X_gpu = cu_csr_matrix(X)
        
        # Use GPU-accelerated UMAP
        umap_model = UMAP(
            n_neighbors=N_NEIGHB,
            min_dist=MIN_DIST,
            n_components=UMAP_DIM,
            metric="cosine",
            random_state=0,
            init="spectral",  # Supported init method in cuML
            verbose=True,
        )
        X_umap = umap_model.fit_transform(X_gpu)
    else:
        # Use CPU UMAP
        umap_model = umap.UMAP(
            n_neighbors=N_NEIGHB,
            min_dist=MIN_DIST,
            n_components=UMAP_DIM,
            metric="cosine",
            random_state=0,
        )
        X_umap = umap_model.fit_transform(X)
    
    umap_time = time.time() - start_time
    print(f"UMAP completed in {umap_time:.2f} seconds")
    
    # ---------- CLUSTER ----------
    print("Clustering with HDBSCAN...")
    start_time = time.time()
    
    if HAS_CUDA:
        # Use GPU-accelerated HDBSCAN
        clusterer = HDBSCAN(
            min_cluster_size=MIN_CLUST,
            min_samples=MIN_SAMPLES,
            metric="euclidean",
            verbose=True,
        )
        # cuML HDBSCAN expects X_umap as device array
        labels = clusterer.fit_predict(X_umap)
    else:
        # Use CPU HDBSCAN
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=MIN_CLUST,
            min_samples=MIN_SAMPLES,
            metric="euclidean",
        )
        labels = clusterer.fit_predict(X_umap)
    
    hdbscan_time = time.time() - start_time
    print(f"HDBSCAN completed in {hdbscan_time:.2f} seconds")
    
    # Move results back to CPU if needed
    if HAS_CUDA:
        labels = labels.get() if hasattr(labels, 'get') else labels
    
    df["cluster"] = labels  # -1 = noise
    
    unique_clusters = df["cluster"].unique()
    print(f"Discovered {len(unique_clusters)} clusters")
    cluster_sizes = df.groupby("cluster").size()
    print(cluster_sizes)
    
    # ---------- SAVE ARTIFACTS ----------
    joblib.dump(vec, OUTPUT_DIR / VEC_PKL)
    joblib.dump(umap_model, OUTPUT_DIR / UMAP_PKL)
    joblib.dump(clusterer, OUTPUT_DIR / HDBSCAN_PKL)
    joblib.dump(FLAG_COLS, OUTPUT_DIR / FLAG_PKL)
    
    # Save a CSV snapshot for manual bucket naming
    df.to_csv(OUTPUT_DIR / "clustered_compositions.csv", index=False)
    
    print(f"Vectorizer, UMAP reducer, HDBSCAN model, and flag list saved to {OUTPUT_DIR}")
    
    # Save examples from each cluster
    print("\nExamples from the largest clusters:")
    cluster_examples = {}
    
    for cluster_id in sorted(cluster_sizes.index, key=lambda x: cluster_sizes[x], reverse=True)[:10]:
        if cluster_id >= 0:  # Skip noise cluster (-1)
            examples = df[df["cluster"] == cluster_id]["raw"].values[:5].tolist()
            print(f"Cluster {cluster_id} ({cluster_sizes[cluster_id]} items): {examples[:3]}")
            cluster_examples[str(cluster_id)] = examples
    
    with open(OUTPUT_DIR / "cluster_examples.json", "w") as f:
        json.dump(cluster_examples, f, indent=2)
    
    # Save performance metrics
    if HAS_CUDA:
        with open(OUTPUT_DIR / "performance_metrics.json", "w") as f:
            json.dump({
                "umap_time_seconds": umap_time,
                "hdbscan_time_seconds": hdbscan_time,
                "total_time_seconds": umap_time + hdbscan_time,
                "using_gpu": True,
                "num_compositions": len(compositions),
                "num_piis": num_piis
            }, f, indent=2)
    
    print(f"\nCluster examples saved to {OUTPUT_DIR}/cluster_examples.json")
    print("Clustering complete!")

if __name__ == "__main__":
    main() 