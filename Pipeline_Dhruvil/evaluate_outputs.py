import json
import os
from pathlib import Path
import re

def parse_complex_composition(comp_string):
    """
    Attempt to parse a composition string into a dictionary {component: fraction}.
    Handles a wide variety of notations:
      - numeric fraction + optional unit + formula, e.g. "8mol.%Y2O3"
      - just a formula, e.g. "Polyethylene Oxide"
      - multiple components separated by : or - or +, e.g. "2mol.%Ti:8mol.%Y2O3:ZrO2"
      - partial doping statements, e.g. "Zr-1.1Sn-0.19Fe"
      - etc.
    
    Returns a dict: { 'Y2O3': 8.0, 'ZrO2': 1.0 } as an example
    """
    # Quick handle for edge cases
    if not comp_string or comp_string.strip().upper() == "NULL":
        return {}

    # We'll replace parentheses to keep them from splitting the formula in awkward ways,
    # but we do want to consider them as separators if there's a fraction outside vs. inside.
    # Because forms like "Nd(1.48at.%):Ca0.08..." can be tricky, we do the broad approach:
    # 1) first break the composition string into "chunks" by typical delimiters.
    #    We treat colons, semicolons, plus signs, commas, parentheses (?), slashes, and dashes 
    #    all as potential delimiters.  Then parse each chunk with a more specialized regex.
    
    # Step 1: unify some unusual Unicode dashes or minus signs into a common ASCII '-'
    # (In real data, you might see \u2212, etc.)
    s = comp_string.replace('\u2013','-').replace('\u2212','-')
    
    # Common separators: : ; + - / , ( ) 
    # We will split on one or more of these characters
    # BUT watch out that doping often uses parentheses to indicate doping. 
    # This is a big heuristic approach, do what works best in your data:
    chunks = re.split(r'[\:\;\+\-,/()]+', s)
    
    # Let's keep a dictionary of parsed results
    parsed_dict = {}
    
    # A helper function to insert into parsed_dict with additive fractions
    def add_component(formula, fraction):
        formula = formula.strip()
        # If the formula is still empty, skip
        if not formula:
            return
        # You might standardize the formula, e.g. remove spaces or uppercase
        # Or do more advanced formula cleanup if needed:
        # e.g. "Polyethylene Oxide" => "PolyethyleneOxide" or a known short code.
        # For now, let's just keep it as is.
        parsed_dict[formula] = parsed_dict.get(formula, 0.0) + fraction

    # We'll define known fraction units in a single pattern:
    fraction_unit_pattern = r'(?:wt\.%|mol\.%|at\.%|vol\.%|%)'  # add more if needed
    
    # For each chunk, parse:
    for c in chunks:
        c = c.strip()
        if not c:
            continue
        
        # Try a more advanced approach:
        # 1) See if c matches something like "8mol.%Y2O3"
        #    We'll break that as numeric=8, unit=mol.%, formula=Y2O3
        #
        # 2) Or c might be e.g. "1.1Sn" => numeric=1.1, formula="Sn"
        #    or "Sn" => fraction=1.0, formula="Sn"
        #    or "Polyethylene Oxide" => fraction=1.0, formula as is
        #    or "15wt%MWCNT" => numeric=15, unit=wt%, formula=MWCNT
        
        match_pattern = r'^([\d\.]+)?([A-Za-z\.\%]*)?(.*)$'
        m = re.match(match_pattern, c)
        if m:
            # Extract groups
            raw_num, raw_unit, raw_formula = m.groups()
            raw_num = (raw_num or "").strip()
            raw_unit = (raw_unit or "").strip()
            raw_formula = (raw_formula or "").strip()
            
            if raw_num:
                try:
                    numeric_val = float(raw_num)
                except ValueError:
                    numeric_val = 1.0
            else:
                numeric_val = 1.0  # no numeric => default 1.0
            
            # Clean up raw_formula
            if raw_formula:
                add_component(raw_formula, numeric_val)
            else:
                # Skip empty formulas
                pass
        else:
            # Fallback: if we couldn't parse with the pattern, treat chunk as formula with fraction=1
            add_component(c, 1.0)

    return parsed_dict


def match_compositions_flex(test_str, model_list, tol=1e-4):
    """
    1) Parse the test_str using parse_complex_composition -> {component: fraction}
    2) Convert model_list -> {component: fraction}
    3) Compare float values within tolerance for all components in the union
    """
    try:
        def parse_model_composition(model_comp):
            """
            Convert something like:
              [['Li2O', 20.0], ['PbO', 20.0], ...]
            to:
              { 'Li2O': 20.0, 'PbO': 20.0, ... }
            """
            d = {}
            for oxide, fraction in model_comp:
                d[oxide] = d.get(oxide, 0.0) + float(fraction)
            return d

        def dicts_match(dict1, dict2, tolerance):
            # Skip if either dictionary is empty
            if not dict1 or not dict2:
                return False
                
            # Compare all keys in the union
            all_keys = set(dict1.keys()) | set(dict2.keys())
            for k in all_keys:
                v1 = dict1.get(k, 0.0)
                v2 = dict2.get(k, 0.0)
                if abs(v1 - v2) > tolerance:
                    return False
            return True

        # 1) Parse test string
        test_dict = parse_complex_composition(test_str)
        # 2) Parse model list
        model_dict = parse_model_composition(model_list)
        # 3) Compare
        return dicts_match(test_dict, model_dict, tol)
    except Exception as e:
        # For any parsing errors, return False
        print(f"  Error in match_compositions_flex: {str(e)}")
        return False


def check_exact_formula_match(test_comp, model_comp_str):
    """
    Check if a test composition formula matches exactly in the model output.
    Useful for chemical formulas that don't use percentage notation.
    
    :param test_comp: Test composition string like "Li1.5Al0.5Ge1.5(PO4)3"
    :param model_comp_str: String representation of model output
    :return: True if exact match found, False otherwise
    """
    try:
        # Clean up the test composition to handle special cases
        clean_test = test_comp.replace(' ', '').replace('%', '')
        
        # For simple cases, check if it appears directly in the string
        if clean_test in model_comp_str.replace(' ', ''):
            return True
        
        # For more complex cases (with slight format differences), try pattern matching
        # Convert the test composition to a regex pattern
        # Replace numbers with \d+\.?\d* pattern
        pattern = re.escape(clean_test)
        pattern = re.sub(r'\\d+\\.?\\d*', r'\\d+\\.?\\d*', pattern)
        
        # Try to match the pattern in the model output
        if re.search(pattern, model_comp_str.replace(' ', '')):
            return True
            
        return False
    except Exception as e:
        print(f"  Error in check_exact_formula_match: {str(e)}")
        return False


def evaluate_model_against_test_set(model_output_file, test_set_dir, tolerance=1e-4):
    """
    Compare model outputs with test set compositions using multiple matching strategies:
    1. Advanced composition parsing with match_compositions_flex
    2. Exact formula matching as fallback
    
    Args:
        model_output_file (str): Path to consolidated model output JSON file
        test_set_dir (str): Path to directory containing test set JSON files
        tolerance (float): Tolerance for numerical comparison
        
    Returns:
        dict: Statistics about matches
    """
    # Load model outputs
    print(f"Loading model outputs from {model_output_file}...")
    with open(model_output_file, 'r') as f:
        model_outputs = json.load(f)
    
    # Initialize counters and tracking
    all_test_compositions = []  # List to store all test set compositions
    found_matches = []  # List to track which test compositions were found
    match_methods = {"flex": 0, "exact": 0}  # Track which method worked
    
    # Process each file in the test set directory
    print(f"Processing test set files from {test_set_dir}...")
    total_files = len(list(Path(test_set_dir).glob('*.json')))
    print(f"Found {total_files} test files to process")
    
    for test_file in Path(test_set_dir).glob('*.json'):
        pii = test_file.stem
        print(f"Processing test file: {pii}")
        
        try:
            # Load test file
            with open(test_file, 'r') as f:
                test_data = json.load(f)
            
            # Extract compositions from test file - safely handle different formats
            test_compositions = []
            if "response" in test_data and "Compositions" in test_data["response"]:
                for item in test_data["response"]["Compositions"]:
                    if "composition" in item:
                        test_compositions.append(item["composition"])
            
            # Add to master list with PII for tracking
            for comp in test_compositions:
                all_test_compositions.append({"pii": pii, "composition": comp})
            
            # Check if model has outputs for this PII
            if pii in model_outputs:
                # For each test composition, check if it's found in model outputs
                for test_comp in test_compositions:
                    matched = False
                    match_method = None
                    
                    # Try both matching methods for each model composition
                    for model_comp_group in model_outputs[pii]:
                        # Try advanced composition parser first
                        for model_comp in model_comp_group:
                            if match_compositions_flex(test_comp, model_comp, tolerance):
                                found_matches.append({"pii": pii, "composition": test_comp})
                                matched = True
                                match_method = "flex"
                                print(f"  Matched with flex parsing: {test_comp}")
                                break
                        
                        if matched:
                            break
                            
                        # If flex matching fails, try exact string-based matching
                        model_comp_str = str(model_comp_group)
                        if check_exact_formula_match(test_comp, model_comp_str):
                            found_matches.append({"pii": pii, "composition": test_comp})
                            matched = True
                            match_method = "exact"
                            print(f"  Matched with exact string: {test_comp}")
                            break
                    
                    if matched and match_method:
                        match_methods[match_method] += 1
                    
                    if not matched:
                        print(f"  No match found for: {test_comp}")
            else:
                print(f"  No model outputs found for PII: {pii}")
                    
        except Exception as e:
            print(f"Error processing {test_file}: {str(e)}")
            continue
    
    # Remove duplicates from test compositions
    unique_test_comps = []
    seen = set()
    for comp_data in all_test_compositions:
        if comp_data["composition"] not in seen:
            seen.add(comp_data["composition"])
            unique_test_comps.append(comp_data)
    
    # Remove duplicates from matched compositions
    unique_matches = []
    seen = set()
    for match_data in found_matches:
        if match_data["composition"] not in seen:
            seen.add(match_data["composition"])
            unique_matches.append(match_data)
    
    # Calculate statistics
    stats = {
        "total_test_files": total_files,
        "total_unique_compositions_in_test_set": len(unique_test_comps),
        "total_matched_compositions": len(unique_matches),
        "match_percentage": (len(unique_matches) / len(unique_test_comps) * 100) if unique_test_comps else 0,
        "matches_by_method": match_methods
    }
    
    return stats, unique_test_comps, unique_matches


if __name__ == "__main__":
    # Configuration
    model_output_file = "consolidated_OGpipeline-AdditionalGPTdata-xLimited.json"
    test_set_dir = "test-set"
    tolerance = 1e-4  # Adjust as needed
    
    # Run evaluation
    stats, all_test_comps, matched_comps = evaluate_model_against_test_set(
        model_output_file, test_set_dir, tolerance
    )
    
    # Print results
    print("\n===== EVALUATION RESULTS =====")
    print(f"Total test files processed: {stats['total_test_files']}")
    print(f"Total unique compositions in test set: {stats['total_unique_compositions_in_test_set']}")
    print(f"Total matched compositions found: {stats['total_matched_compositions']}")
    print(f"Match percentage: {stats['match_percentage']:.2f}%")
    print(f"Matches by method: Flex parser: {stats['matches_by_method']['flex']}, Exact string: {stats['matches_by_method']['exact']}")
    
    # Save detailed results to a file
    output_file = "evaluation_results-OGpipeline-AdditionalGPTdata-newTestset.json"
    with open(output_file, "w") as f:
        json.dump({
            "statistics": stats,
            "unmatched_compositions": [
                comp for comp in all_test_comps 
                if comp not in matched_comps
            ]
        }, f, indent=2)
    
    print(f"\nDetailed results saved to {output_file}")
