import os
import json
import re
from tqdm import tqdm
from pathlib import Path
import spacy
from spacy.language import Language

# Import Gemini functions from the llm directory
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm.llm_utils import simple_prompt

# Configure paths
PROMPTING_DATA_DIR = Path("Prompting_data/research-paper-text")
OUTPUT_DIR = Path("composition_sentences")

# Custom component for scientific sentence boundaries
@Language.component("scientific_sentence_boundaries")
def scientific_sentence_boundaries(doc):
    """Custom component for handling scientific text sentence boundaries"""
    for token in doc:
        # Don't split on periods in chemical formulas
        if token.text.endswith('.'):
            if any(char.isupper() for char in token.text[:-1]):  # Chemical formula
                token.is_sent_start = False
            if any(char.isdigit() for char in token.text):  # Measurement
                token.is_sent_start = False
                
        # Don't split on chemical formula patterns
        if token.i > 0:
            prev_token = doc[token.i - 1]
            # Pattern like "H2O" or "NaCl"
            if (prev_token.text[0].isupper() and 
                any(c.isdigit() for c in prev_token.text)):
                token.is_sent_start = False
    return doc

def split_into_sentences(text):
    """
    Split the text into sentences using spaCy with scientific text awareness.
    """
    # Clean up text (remove excessive newlines, spaces, etc.)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Load English model
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading spaCy model...")
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    
    # Add custom scientific text component
    if "scientific_sentence_boundaries" not in nlp.pipe_names:
        nlp.add_pipe("scientific_sentence_boundaries", before="parser")
    
    doc = nlp(text)
    sentences = [str(sent).strip() for sent in doc.sents]
    
    # Filter out very short sentences (likely headings or artifacts)
    sentences = [s for s in sentences if len(s) > 15]
    
    return sentences

def contains_composition(sentence):
    """
    Use Gemini 2.0 to determine if a sentence contains chemical composition information.
    
    Args:
        sentence (str): The sentence to check
        
    Returns:
        bool: True if the sentence contains composition information, False otherwise
    """
    prompt = f"""
    You are a chemistry and materials science expert. Your task is to analyze the following scientific sentence and determine if it contains chemical composition information of any material.

    The composition can be in any form: formula, text description (like '20% by weight'), stoichiometric ratios, etc.
    
    IMPORTANT: If the sentence is describing the elemental or compound composition of ANY material, respond with 'YES'.
    If the sentence doesn't specifically describe material composition, respond with 'NO'.
    
    Sentence: "{sentence}"
    
    Answer (YES/NO):
    """
    
    try:
        # Using the simple_prompt function with custom/gemini-2point0 model
        response, _ = simple_prompt("custom/gemini-2point0", 0.0, prompt)
        # Check if the response contains YES (case insensitive)
        return "YES" in response.upper()
    except Exception as e:
        print(f"Error processing sentence with Gemini: {str(e)}")
        return False  # Default to False on error

def process_paper(paper_path):
    """
    Process a single research paper: read it, split into sentences, 
    identify composition sentences, and return results.
    
    Args:
        paper_path (Path): Path to the research paper text file
        
    Returns:
        list: List of sentences containing composition information
    """
    pii = paper_path.stem
    print(f"Processing {pii}...")
    
    # Read the paper
    with open(paper_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Split into sentences
    sentences = split_into_sentences(text)
    print(f"  Found {len(sentences)} sentences")
    
    # Check each sentence for composition information
    composition_sentences = []
    for sentence in tqdm(sentences, desc=f"  Analyzing {pii} sentences", leave=False):
        if contains_composition(sentence):
            composition_sentences.append(sentence)
    
    print(f"  Found {len(composition_sentences)} sentences with composition information")
    return composition_sentences

def main():
    """
    Main function to process all papers in the directory.
    """
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get all text files in the directory
    paper_files = list(PROMPTING_DATA_DIR.glob('*.txt'))
    print(f"Found {len(paper_files)} papers to process")
    
    # Process each paper
    for paper_path in tqdm(paper_files, desc="Processing papers"):
        pii = paper_path.stem
        output_path = OUTPUT_DIR / f"{pii}.json"
        
        # Skip if already processed
        if output_path.exists():
            print(f"Skipping {pii} - already processed")
            continue
        
        # Process the paper
        composition_sentences = process_paper(paper_path)
        
        # Save results
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                "pii": pii,
                "composition_sentences": composition_sentences
            }, f, indent=2)
        
        print(f"Saved results for {pii} - {len(composition_sentences)} composition sentences found")
    
    print("\nProcessing complete!")
    
    # Print summary
    processed_files = list(OUTPUT_DIR.glob('*.json'))
    total_sentences = sum(len(json.loads(open(f, 'r').read())["composition_sentences"]) for f in processed_files)
    print(f"Total papers processed: {len(processed_files)}")
    print(f"Total composition sentences found: {total_sentences}")
    print(f"Average composition sentences per paper: {total_sentences/len(processed_files) if processed_files else 0:.2f}")

if __name__ == "__main__":
    main()
