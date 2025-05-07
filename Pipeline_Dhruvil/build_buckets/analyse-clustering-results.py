import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import os
from pathlib import Path
import seaborn as sns

def analyze_clusters(csv_path):
    """
    Analyze the clustering results from a CSV file
    
    Args:
        csv_path: Path to the CSV file containing clustered compositions
    """
    print(f"Loading clustering data from {csv_path}")
    # Load the data
    df = pd.read_csv(csv_path)
    
    # Count compositions per cluster
    cluster_counts = Counter(df['cluster'])
    
    # Sort by frequency (most common first)
    sorted_clusters = sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nFound {len(cluster_counts)} distinct clusters")
    print(f"Total compositions: {len(df)}")
    
    # Print the top 20 clusters by size
    print("\nTop 20 clusters by size:")
    for i, (cluster_id, count) in enumerate(sorted_clusters[:20]):
        percentage = (count / len(df)) * 100
        print(f"Cluster {cluster_id}: {count} compositions ({percentage:.2f}%)")
    
    # Generate a histogram of cluster sizes
    plt.figure(figsize=(12, 6))
    
    # Count how many clusters have each size
    sizes = list(cluster_counts.values())
    size_distribution = Counter(sizes)
    sorted_sizes = sorted(size_distribution.items())
    
    # Extract x and y data for the plot
    x_data = [x[0] for x in sorted_sizes]
    y_data = [x[1] for x in sorted_sizes]
    
    plt.bar(x_data, y_data)
    plt.xlabel('Cluster Size (Number of Compositions)')
    plt.ylabel('Number of Clusters')
    plt.title('Distribution of Cluster Sizes')
    plt.grid(True, alpha=0.3)
    
    # Optional: Use log scale for x-axis if there are very large clusters
    if max(sizes) > 100:
        plt.xscale('log')
        plt.title('Distribution of Cluster Sizes (Log Scale)')
    
    # Save plot
    plot_path = Path(csv_path).parent / 'cluster_size_distribution.png'
    plt.savefig(plot_path)
    print(f"\nCluster size distribution plot saved to {plot_path}")
    
    # Analyze feature distribution across clusters
    feature_cols = ['has_pct', 'has_wt', 'has_vol', 'has_at', 'has_mol', 
                   'has_colon', 'has_slash', 'many_dashes', 'paren', 'bracket']
    
    # Get the top 10 clusters by size for feature analysis
    top_clusters = [cluster_id for cluster_id, _ in sorted_clusters[:10]]
    
    # Create a summary for each top cluster
    print("\nFeature analysis for top 10 clusters:")
    for cluster_id in top_clusters:
        cluster_df = df[df['cluster'] == cluster_id]
        
        # Calculate feature frequency within this cluster
        feature_freq = {}
        for feature in feature_cols:
            feature_freq[feature] = cluster_df[feature].mean() * 100  # Convert to percentage
        
        print(f"\nCluster {cluster_id} ({len(cluster_df)} compositions):")
        
        # Get 3 example compositions from this cluster
        examples = cluster_df['raw'].head(3).tolist()
        print("Examples:")
        for ex in examples:
            print(f"  - {ex}")
        
        # Print feature statistics
        print("Feature percentages:")
        for feature, percentage in sorted(feature_freq.items(), key=lambda x: x[1], reverse=True):
            if percentage > 0:
                print(f"  - {feature}: {percentage:.1f}%")
    
    # Generate a heatmap of feature presence by cluster
    plt.figure(figsize=(14, 8))
    
    # Prepare data for top 20 clusters
    top20_clusters = [cluster_id for cluster_id, _ in sorted_clusters[:20]]
    heatmap_data = []
    
    for cluster_id in top20_clusters:
        cluster_df = df[df['cluster'] == cluster_id]
        row = [cluster_id, len(cluster_df)]  # Cluster ID and size
        
        # Add feature percentages
        for feature in feature_cols:
            row.append(cluster_df[feature].mean() * 100)
        
        heatmap_data.append(row)
    
    # Convert to DataFrame for the heatmap
    heatmap_df = pd.DataFrame(
        heatmap_data, 
        columns=['cluster_id', 'size'] + feature_cols
    )
    
    # Set cluster_id as index
    heatmap_df = heatmap_df.set_index('cluster_id')
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(heatmap_df[feature_cols], annot=True, cmap="YlGnBu", fmt=".0f")
    plt.title('Feature Distribution (%) Across Top 20 Clusters')
    plt.tight_layout()
    
    # Save heatmap
    heatmap_path = Path(csv_path).parent / 'cluster_feature_heatmap.png'
    plt.savefig(heatmap_path)
    print(f"\nCluster feature heatmap saved to {heatmap_path}")
    
    return df, cluster_counts

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze clustering results')
    parser.add_argument('--csv', default='clustered_compositions-dhruvil-sent.csv',
                        help='Path to the clustered compositions CSV file')
    
    args = parser.parse_args()
    
    # Run the analysis
    analyze_clusters(args.csv)
