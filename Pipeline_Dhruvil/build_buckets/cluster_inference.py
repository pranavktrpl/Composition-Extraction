import re, unicodedata, json, joblib, numpy as np, pandas as pd
from scipy.sparse import hstack
from pathlib import Path
import os

# ---------- USER CONFIG ----------
INPUT_JSON = "composition-sentences.json"  # JSON file with {"compositions": [ ... ]}
MODEL_DIR = Path("clusters")          # Directory where models are saved
OUTPUT_FILE = "clustered_output.json" # Output file for clustered compositions
# ----------------------------------

def canon(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    subs = str.maketrans("₀₁₂₃₄₅₆₇₈₉⁰¹²³⁴⁵⁶⁷⁸⁹", "01234567890123456789")
    s = s.translate(subs)
    s = s.replace("–", "-").replace("—", "-")   # dash unification
    s = re.sub(r"\s+", " ", s).strip()
    # optional glass chatter
    s = re.sub(r"\bglass\b", "", s, flags=re.I)
    return s

def flags(s: str, flag_cols):
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
    # Check if model files exist
    model_files = {
        "vec": MODEL_DIR / "vec.pkl",
        "umap": MODEL_DIR / "umap.pkl",
        "hdbscan": MODEL_DIR / "hdbscan.pkl",
        "flag_cols": MODEL_DIR / "flag_cols.pkl"
    }
    
    for name, path in model_files.items():
        if not path.exists():
            print(f"Error: Model file {path} not found. Run cluster_compositions.py first.")
            return
    
    # Load models
    print("Loading models...")
    vec = joblib.load(model_files["vec"])
    umap_model = joblib.load(model_files["umap"])
    clusterer = joblib.load(model_files["hdbscan"])
    flag_cols = joblib.load(model_files["flag_cols"])
    
    # Load and preprocess data
    print(f"Loading composition data from {INPUT_JSON}...")
    try:
        with open(INPUT_JSON) as f:
            data = json.load(f)
            compositions = data["compositions"]
            num_piis = data.get("source_piis", "unknown")
        print(f"Loaded {len(compositions)} compositions from {num_piis} PIIs")
    except FileNotFoundError:
        print(f"Error: {INPUT_JSON} not found.")
        return
    except KeyError:
        print(f"Error: {INPUT_JSON} doesn't contain a 'compositions' key.")
        return
    
    # Process the compositions
    df = pd.DataFrame({"raw": compositions})
    df["canon"] = df["raw"].apply(canon)
    df[flag_cols] = df["canon"].apply(lambda x: pd.Series(flags(x, flag_cols))).astype(float)
    
    # Vectorize
    print("Vectorizing data...")
    X_text = vec.transform(df["canon"])
    X = hstack([X_text, df[flag_cols].values])
    
    # Apply dimensionality reduction
    print("Applying UMAP transformation...")
    X_umap = umap_model.transform(X)
    
    # Predict clusters
    print("Predicting clusters with HDBSCAN...")
    labels = clusterer.approximate_predict(X_umap)
    df["cluster"] = labels
    
    # Generate statistics
    unique_clusters = sorted(df["cluster"].unique())
    cluster_counts = df["cluster"].value_counts().sort_index()
    print(f"Compositions assigned to {len(unique_clusters)} clusters")
    print(cluster_counts)
    
    # Save results
    output = {
        "clustered_compositions": []
    }
    
    for _, row in df.iterrows():
        output["clustered_compositions"].append({
            "composition": row["raw"],
            "cluster": int(row["cluster"])
        })
    
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"Results saved to {OUTPUT_FILE}")
    
    # Print sample from each cluster
    print("\nSamples from each cluster:")
    for cluster in sorted(df["cluster"].unique()):
        if cluster >= 0:  # Skip noise cluster (-1)
            samples = df[df["cluster"] == cluster]["raw"].values[:3].tolist()
            print(f"Cluster {cluster} ({cluster_counts[cluster]} items): {samples}")

if __name__ == "__main__":
    main() 