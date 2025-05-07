import os
import sys
import json
import random
import argparse
from tqdm import tqdm
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from build_buckets.composition_sentences import process_paper

# ---------- USER CONFIG ----------
PROMPTING_DATA_DIR = Path("Prompting_data/research-paper-text")
PREPROCESSED_DIR = Path("preprocessed_test_set")
OUTPUT_JSON = "compositions.json"  # Output JSON for tfidf.py to use
# ----------------------------------

def process_selected_papers(num_piis=None):
    """
    Process papers and extract composition sentences.
    
    Args:
        num_piis (int, optional): Number of random PIIs (papers) to process. 
                                If None, process all papers.
    """
    os.makedirs(PREPROCESSED_DIR, exist_ok=True)
    
    # Get all text files in the directory
    paper_files = list(PROMPTING_DATA_DIR.glob('*.txt'))
    
    # Randomly select a subset of papers if num_piis is specified
    if num_piis is not None and num_piis > 0 and num_piis < len(paper_files):
        print(f"Randomly selecting {num_piis} papers from {len(paper_files)} available papers")
        paper_files = random.sample(paper_files, num_piis)
    
    print(f"Found {len(paper_files)} papers to process")
    
    all_compositions = []
    papers_processed = 0
    piis_used = []
    
    # Load existing compositions if OUTPUT_JSON exists
    if os.path.exists(OUTPUT_JSON):
        try:
            with open(OUTPUT_JSON, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                all_compositions = existing_data.get("compositions", [])
                print(f"Loaded {len(all_compositions)} existing composition sentences from {OUTPUT_JSON}")
        except:
            print(f"Error loading existing {OUTPUT_JSON}, starting fresh")
    
    # Process each paper
    for paper_path in tqdm(paper_files, desc="Processing papers"):
        pii = paper_path.stem
        output_path = PREPROCESSED_DIR / f"{pii}.json"
        piis_used.append(pii)
        
        # Skip if already processed
        if output_path.exists():
            print(f"Loading processed data for {pii}")
            with open(output_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_compositions.extend(data["composition_sentences"])
            papers_processed += 1
        else:
            # Process the paper
            composition_sentences = process_paper(paper_path)
            
            # Save results
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "pii": pii,
                    "composition_sentences": composition_sentences
                }, f, indent=2)
            
            all_compositions.extend(composition_sentences)
            papers_processed += 1
            print(f"Saved results for {pii} - {len(composition_sentences)} composition sentences found")
        
        # Incrementally save all compositions to OUTPUT_JSON after each paper
        with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
            json.dump({
                "compositions": all_compositions,
                "source_piis": len(piis_used)
            }, f, indent=2)
    
    print(f"\nProcessing complete!")
    print(f"Papers processed: {papers_processed}")
    print(f"Total composition sentences found: {len(all_compositions)}")
    
    # Also save a list of PIIs used for reference
    with open(PREPROCESSED_DIR / "piis_used.json", 'w') as f:
        json.dump({
            "piis": piis_used,
            "count": len(piis_used)
        }, f, indent=2)
    
    print(f"All compositions saved to {OUTPUT_JSON}")
    print(f"You can now run cluster_compositions.py to cluster the compositions")
    
    return all_compositions

def main():
    """Parse arguments and process papers"""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process research papers to extract composition sentences')
    parser.add_argument('--piis', type=int, help='Number of random PIIs (papers) to process. Default: process all')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility. Default: 42')
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Process papers
    process_selected_papers(args.piis)

if __name__ == "__main__":
    main() 