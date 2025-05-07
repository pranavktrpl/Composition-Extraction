import os
import json
import sys
from pathlib import Path

# Add parent directory to path to import module properly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from build_buckets.composition_sentences import split_into_sentences, contains_composition

# Test on a single paper from the dataset
TEST_PAPER = "S0167577X06001327.txt"  # Using the same PII as referenced in llm_utils.py

def test_composition_detector():
    """
    Test the composition sentence detector on a single paper.
    """
    # Path to the paper
    paper_path = Path(f"Prompting_data/research-paper-text/{TEST_PAPER}")
    
    if not paper_path.exists():
        print(f"Error: Test paper {TEST_PAPER} not found.")
        return
    
    print(f"Testing with paper: {TEST_PAPER}")
    
    # Read the paper
    with open(paper_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Split into sentences
    sentences = split_into_sentences(text)
    print(f"Found {len(sentences)} sentences")
    
    # Test the first few sentences to see if they contain composition information
    print("Testing the first 5 sentences:")
    for i, sentence in enumerate(sentences[:5]):
        print(f"\nSentence {i+1}: {sentence}")
        result = contains_composition(sentence)
        print(f"Contains composition: {result}")
    
    # Find all sentences with composition information (up to 10 for the test)
    print("\nFinding sentences with composition information (up to 10):")
    composition_sentences = []
    
    for i, sentence in enumerate(sentences):
        if len(composition_sentences) >= 10:
            break
        
        if contains_composition(sentence):
            composition_sentences.append(sentence)
            print(f"\nComposition sentence {len(composition_sentences)}: {sentence}")
    
    print(f"\nFound {len(composition_sentences)} sentences with composition information")
    
    # Save the results to a test file
    output_dir = Path("test_results")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(output_dir / f"{paper_path.stem}_test.json", 'w', encoding='utf-8') as f:
        json.dump({
            "pii": paper_path.stem,
            "composition_sentences": composition_sentences
        }, f, indent=2)
    
    print(f"Test results saved to {output_dir / f'{paper_path.stem}_test.json'}")

if __name__ == "__main__":
    test_composition_detector() 