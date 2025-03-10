import os
import json
import re
import pandas as pd
from collections import defaultdict

def load_json_file(file_path):
    """Load a JSON file and return its contents."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def normalize_composition(composition):
    """Normalize composition string to enable flexible matching."""
    if not composition:
        return ""
    
    # Convert to lowercase
    comp = composition.lower()
    
    # Handle special case for pure elements/compounds
    if comp == "c" or comp == "carbon" or comp == "graphite":
        return "c"
    
    # Extract components and percentages
    components = []
    percentages = []
    
    # Extract weight percentages
    wt_matches = re.findall(r'(\d+)(?:\.\d+)?wt%', comp)
    percentages.extend([f"{m}wt%" for m in wt_matches])
    
    # Extract material names (common in materials science)
    material_matches = re.findall(r'([a-z]+(?:-[a-z]+)?)', comp)
    components.extend([m for m in material_matches if m not in ['wt']])
    
    # Sort components and percentages for consistent comparison
    components.sort()
    percentages.sort()
    
    # Create normalized string
    normalized = "-".join(components + percentages)
    
    return normalized

def compare_compositions(test_compositions, pred_compositions):
    """Compare compositions between test and prediction files with flexible matching."""
    if not test_compositions or not pred_compositions:
        return False, 0, 0
    
    # Create dictionaries mapping placeholders to compositions
    test_dict = {item["placeholder"]: item["composition"] for item in test_compositions}
    pred_dict = {item["placeholder"]: item["composition"] for item in pred_compositions}
    
    # Count matches
    total_blanks = len(test_dict)
    correct_blanks = 0
    
    for placeholder, test_comp in test_dict.items():
        if placeholder in pred_dict:
            # Normalize both compositions for comparison
            norm_test = normalize_composition(test_comp)
            norm_pred = normalize_composition(pred_dict[placeholder])
            
            if norm_test == norm_pred:
                correct_blanks += 1
    
    # Document is correct if all blanks are correct
    doc_correct = correct_blanks == total_blanks
    
    return doc_correct, correct_blanks, total_blanks

def update_excel(excel_path, results):
    """Update the Excel file with evaluation results."""
    try:
        # Read the Excel file
        df = pd.read_excel(excel_path)
        
        # Update the Correct/Incorrect column
        for pii, result in results.items():
            # Find the row with matching PII
            mask = df.iloc[:, 0] == pii  # Assuming PII is in the first column
            if mask.any():
                # Update the Correct/Incorrect column (assuming it's the 3rd column, index 2)
                df.loc[mask, df.columns[2]] = "1" if result["doc_correct"] else "0"
        
        # Write back to Excel
        df.to_excel(excel_path, index=False)
        print(f"Updated Excel file: {excel_path}")
        
    except Exception as e:
        print(f"Error updating Excel file: {e}")

def main():
    # Directories
    test_dir = "test-set"
    pred_dir = "outputs/gemini-flash-4context-structured"  # Change this to the model output directory
    excel_path = "outputs/gemini-flash-4context-structured/Book3.xlsx"  # Excel file to update
    
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
        
        # Print detailed comparison for debugging
        print(f"File: {pii}")
        print(f"  Correct: {doc_correct} ({doc_correct_blanks}/{doc_total_blanks} blanks)")
        for t_comp in test_compositions:
            placeholder = t_comp["placeholder"]
            test_val = t_comp["composition"]
            pred_val = next((p["composition"] for p in pred_compositions if p["placeholder"] == placeholder), "N/A")
            norm_test = normalize_composition(test_val)
            norm_pred = normalize_composition(pred_val)
            match = norm_test == norm_pred
            print(f"    {placeholder}: {test_val} vs {pred_val} -> {match} (normalized: {norm_test} vs {norm_pred})")
        print()
    
    # Update Excel
    update_excel(excel_path, results)
    
    # Calculate and print accuracy
    doc_accuracy = correct_docs / total_docs if total_docs > 0 else 0
    blank_accuracy = correct_blanks / total_blanks if total_blanks > 0 else 0
    
    print(f"Document-level accuracy: {doc_accuracy:.4f} ({correct_docs}/{total_docs})")
    print(f"Blank-level accuracy: {blank_accuracy:.4f} ({correct_blanks}/{total_blanks})")

if __name__ == "__main__":
    main()