import os
import json
import csv
import re
from collections import defaultdict

def load_json_file(file_path):
    """Load a JSON file and return its contents."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def compare_compositions(test_compositions, pred_compositions):
    """Compare compositions between test and prediction files."""
    if not test_compositions or not pred_compositions:
        return False, 0, 0
    
    # Create dictionaries mapping placeholders to compositions
    test_dict = {item["placeholder"]: item["composition"] for item in test_compositions}
    pred_dict = {item["placeholder"]: item["composition"] for item in pred_compositions}
    
    # Count matches
    total_blanks = len(test_dict)
    correct_blanks = 0
    
    for placeholder, test_comp in test_dict.items():
        if placeholder in pred_dict and pred_dict[placeholder] == test_comp:
            correct_blanks += 1
    
    # Document is correct if all blanks are correct
    doc_correct = correct_blanks == total_blanks
    
    return doc_correct, correct_blanks, total_blanks

def update_csv(csv_path, results):
    """Update the CSV file with evaluation results."""
    # Read the existing CSV with error handling for encoding issues
    rows = []
    try:
        # First try with utf-8
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
    except UnicodeDecodeError:
        # If that fails, try with latin-1 (which can read any byte)
        with open(csv_path, 'r', encoding='latin-1') as f:
            reader = csv.reader(f)
            rows = list(reader)
    
    # Update the rows with results
    for i, row in enumerate(rows):
        if i == 0:  # Skip header
            continue
        
        pii = row[0]
        if pii in results:
            row[2] = "1" if results[pii]["doc_correct"] else "0"
    
    # Write back the updated CSV
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

def main():
    # Directories
    test_dir = "test-set"
    pred_dir = "outputs/gpt-4o-4context"  # Change this to the model output directory
    csv_path = "outputs/gpt-4o-4context/gpt-4o-4context.csv"  # CSV file to update
    
    # Results tracking
    results = {}
    total_docs = 0
    correct_docs = 0
    total_blanks = 0
    correct_blanks = 0
    
    # Get all test files
    test_files = [f for f in os.listdir(test_dir) if f.endswith('.json')]
    
    for test_file in test_files:
        pii = test_file.replace('.json', '')
        pred_file = os.path.join(pred_dir, test_file)
        test_file_path = os.path.join(test_dir, test_file)
        
        # Skip if prediction file doesn't exist
        if not os.path.exists(pred_file):
            print(f"Prediction file not found for {pii}")
            continue
        
        # Load files
        test_data = load_json_file(test_file_path)
        pred_data = load_json_file(pred_file)
        
        if not test_data or not pred_data:
            continue
        
        # Extract compositions
        test_compositions = test_data.get("response", {}).get("Compositions", [])
        pred_compositions = pred_data.get("response", {}).get("Compositions", [])
        
        # Compare
        doc_correct, doc_correct_blanks, doc_total_blanks = compare_compositions(
            test_compositions, pred_compositions
        )
        
        # Update counts
        total_docs += 1
        if doc_correct:
            correct_docs += 1
        total_blanks += doc_total_blanks
        correct_blanks += doc_correct_blanks
        
        # Store results
        results[pii] = {
            "doc_correct": doc_correct,
            "correct_blanks": doc_correct_blanks,
            "total_blanks": doc_total_blanks
        }
    
    # Update CSV
    update_csv(csv_path, results)
    
    # Calculate and print accuracy
    doc_accuracy = correct_docs / total_docs if total_docs > 0 else 0
    blank_accuracy = correct_blanks / total_blanks if total_blanks > 0 else 0
    
    print(f"Document-level accuracy: {doc_accuracy:.4f} ({correct_docs}/{total_docs})")
    print(f"Blank-level accuracy: {blank_accuracy:.4f} ({correct_blanks}/{total_blanks})")

if __name__ == "__main__":
    main()