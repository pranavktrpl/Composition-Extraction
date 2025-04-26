import json
import os
from pathlib import Path
import time
from typing import Dict, List, Any, Tuple
from tqdm import tqdm

from llm.llm_utils import simple_prompt

def create_judge_prompt(test_composition: str, model_compositions: List[Any]) -> str:
    """
    Create a prompt for the LLM judge to compare test and model compositions
    
    Args:
        test_composition: The ground truth composition from test set
        model_compositions: The model-generated compositions to evaluate
        
    Returns:
        A formatted prompt string for the LLM judge
    """
    prompt = f"""You are a chemistry expert judge evaluating material composition extraction results.

GROUND TRUTH COMPOSITION:
{test_composition}

MODEL EXTRACTED COMPOSITIONS:
{json.dumps(model_compositions, indent=2)}

Your task is to determine if the model has correctly extracted the ground truth composition. 
The compositions might be expressed differently (different formatting, notation, or order), 
but they should represent the same chemical composition.

Consider:
1. All components present (e.g., "Li2O3" vs "Li₂O₃")
2. Correct ratios/percentages (e.g., "8 mol% Y2O3" vs "Y2O3 (8 mol%)")
3. Equivalent notations (e.g., "Zr-1.1Sn-0.19Fe" vs "Zr1.1Sn0.19Fe" vs "Zr with 1.1 Sn and 0.19 Fe")

Respond with a JSON object with this format:
{{
  "is_match": true/false,
  "confidence": 0-100,
  "reasoning": "Brief explanation of your judgment"
}}

Only return the JSON object, no other text."""

    return prompt

def evaluate_with_llm_judge(model_output_file: str, test_set_dir: str, 
                           llm_model: str = "custom/gemini-2point0", temperature: float = 0.0,
                           max_test_files: int = None) -> Dict:
    """
    Evaluate model outputs against test set using an LLM as a judge
    
    Args:
        model_output_file: Path to consolidated model output JSON file
        test_set_dir: Path to directory containing test set JSON files
        llm_model: LLM model to use as judge
        temperature: Temperature for LLM generation
        max_test_files: Optional limit on number of test files to process
        
    Returns:
        Dictionary containing evaluation statistics
    """
    # Create a timestamped output directory for incremental saves
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = f"evaluation_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a checkpoint file to track progress
    checkpoint_file = os.path.join(output_dir, "checkpoint.json")
    results_file = os.path.join(output_dir, "llm_judge_results.json")
    
    # Check if we have an existing checkpoint to resume from
    processed_piis = set()
    all_test_compositions = []
    found_matches = []
    llm_judge_results = []
    
    if os.path.exists(checkpoint_file):
        print(f"Found checkpoint file at {checkpoint_file}, resuming from last state...")
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
                processed_piis = set(checkpoint_data.get("processed_piis", []))
                all_test_compositions = checkpoint_data.get("all_test_compositions", [])
                found_matches = checkpoint_data.get("found_matches", [])
                llm_judge_results = checkpoint_data.get("llm_judge_results", [])
                print(f"Resuming with {len(processed_piis)} already processed PIIs")
        except Exception as e:
            print(f"Error loading checkpoint: {str(e)}. Starting fresh.")
            processed_piis = set()
            all_test_compositions = []
            found_matches = []
            llm_judge_results = []
    
    # Load model outputs
    print(f"Loading model outputs from {model_output_file}...")
    with open(model_output_file, 'r') as f:
        model_outputs = json.load(f)
    
    # Process each file in the test set directory
    print(f"Processing test set files from {test_set_dir}...")
    test_files = list(Path(test_set_dir).glob('*.json'))
    
    if max_test_files:
        test_files = test_files[:max_test_files]
        
    total_files = len(test_files)
    print(f"Found {total_files} test files to process")
    
    # Add tqdm progress bar while preserving print statements
    for i, test_file in enumerate(tqdm(test_files, desc="Processing test files", unit="file")):
        pii = test_file.stem
        
        # Skip already processed PIIs if resuming
        if pii in processed_piis:
            print(f"Skipping already processed file {i+1}/{total_files}: {pii}")
            continue
            
        print(f"Processing test file {i+1}/{total_files}: {pii}")
        
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
                for test_comp_idx, test_comp in enumerate(tqdm(test_compositions, desc=f"Testing compositions for {pii}", unit="comp", leave=False)):
                    # Call LLM judge for each test composition against model composition groups
                    for model_comp_group in model_outputs[pii]:
                        # Create prompt for LLM judge
                        judge_prompt = create_judge_prompt(test_comp, model_comp_group)
                        
                        # Call LLM judge
                        try:
                            print(f"  Asking LLM judge about: {test_comp}")
                            judge_response, _ = simple_prompt(llm_model, temperature, judge_prompt)
                            
                            # Parse judge response
                            try:
                                # Clean up the response in case it contains markdown or extra text
                                judge_response = judge_response.strip()
                                if "```json" in judge_response:
                                    judge_response = judge_response.split("```json")[1].split("```")[0].strip()
                                elif "```" in judge_response:
                                    judge_response = judge_response.split("```")[1].split("```")[0].strip()
                                
                                judge_result = json.loads(judge_response)
                                
                                # Store result
                                result_entry = {
                                    "pii": pii,
                                    "test_composition": test_comp,
                                    "model_composition": model_comp_group,
                                    "is_match": judge_result["is_match"],
                                    "confidence": judge_result["confidence"],
                                    "reasoning": judge_result["reasoning"]
                                }
                                
                                llm_judge_results.append(result_entry)
                                
                                if judge_result["is_match"]:
                                    found_matches.append({"pii": pii, "composition": test_comp})
                                    print(f"  ✓ LLM judge found match with confidence {judge_result['confidence']}%")
                                    # If we found a match, no need to check other model compositions
                                    break
                                else:
                                    print(f"  ✗ LLM judge found no match (confidence: {judge_result['confidence']}%)")
                                
                            except json.JSONDecodeError:
                                print(f"  Error: Invalid JSON response from LLM judge: {judge_response}")
                                
                        except Exception as e:
                            print(f"  Error with LLM judge: {str(e)}")
                            # Add a small delay to avoid rate limits
                            time.sleep(2)
            else:
                print(f"  No model outputs found for PII: {pii}")
            
            # Mark this PII as processed
            processed_piis.add(pii)
            
            # Save incremental results after each PII
            checkpoint_data = {
                "processed_piis": list(processed_piis),
                "all_test_compositions": all_test_compositions,
                "found_matches": found_matches,
                "llm_judge_results": llm_judge_results,
                "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
                
            with open(results_file, 'w') as f:
                # Also save the formatted results
                # Calculate statistics at this point
                unique_test_comps = []
                seen_test = set()
                for comp_data in all_test_compositions:
                    comp_key = f"{comp_data['pii']}:{comp_data['composition']}"
                    if comp_key not in seen_test:
                        seen_test.add(comp_key)
                        unique_test_comps.append(comp_data)
                
                unique_matches = []
                seen_matches = set()
                for match_data in found_matches:
                    match_key = f"{match_data['pii']}:{match_data['composition']}"
                    if match_key not in seen_matches:
                        seen_matches.add(match_key)
                        unique_matches.append(match_data)
                
                # Calculate interim statistics
                match_percentage = (len(unique_matches) / len(unique_test_comps) * 100) if unique_test_comps else 0
                avg_confidence = sum(r["confidence"] for r in llm_judge_results if r["is_match"]) / len([r for r in llm_judge_results if r["is_match"]]) if [r for r in llm_judge_results if r["is_match"]] else 0
                
                interim_stats = {
                    "total_files_processed": len(processed_piis),
                    "total_files_to_process": total_files,
                    "progress_percentage": (len(processed_piis) / total_files * 100) if total_files else 0,
                    "total_unique_compositions_in_test_set": len(unique_test_comps),
                    "total_matched_compositions": len(unique_matches),
                    "match_percentage": match_percentage,
                    "average_confidence": avg_confidence,
                    "unmatched_compositions": get_unmatched_compositions(unique_test_comps, unique_matches),
                    "all_judge_results": llm_judge_results
                }
                
                json.dump(interim_stats, f, indent=2)
                
            print(f"  Saved progress checkpoint after processing PII: {pii}")
                    
        except Exception as e:
            print(f"Error processing {test_file}: {str(e)}")
            continue
    
    # Remove duplicates from test compositions and matches
    unique_test_comps = []
    seen_test = set()
    for comp_data in all_test_compositions:
        comp_key = f"{comp_data['pii']}:{comp_data['composition']}"
        if comp_key not in seen_test:
            seen_test.add(comp_key)
            unique_test_comps.append(comp_data)
    
    unique_matches = []
    seen_matches = set()
    for match_data in found_matches:
        match_key = f"{match_data['pii']}:{match_data['composition']}"
        if match_key not in seen_matches:
            seen_matches.add(match_key)
            unique_matches.append(match_data)
    
    # Calculate statistics
    stats = {
        "total_test_files": total_files,
        "total_unique_compositions_in_test_set": len(unique_test_comps),
        "total_matched_compositions": len(unique_matches),
        "match_percentage": (len(unique_matches) / len(unique_test_comps) * 100) if unique_test_comps else 0,
        "judge_model": llm_model,
        "average_confidence": sum(r["confidence"] for r in llm_judge_results if r["is_match"]) / len([r for r in llm_judge_results if r["is_match"]]) if [r for r in llm_judge_results if r["is_match"]] else 0
    }
    
    # Save final results to the output directory
    final_results_file = os.path.join(output_dir, "final_results.json")
    with open(final_results_file, 'w') as f:
        json.dump({
            "statistics": stats,
            "unmatched_compositions": get_unmatched_compositions(unique_test_comps, unique_matches),
            "all_judge_results": llm_judge_results
        }, f, indent=2)
    
    print(f"\nFinal results saved to {final_results_file}")
    
    return stats, unique_test_comps, unique_matches, llm_judge_results


def get_unmatched_compositions(all_comps, matched_comps):
    """
    Get compositions that were not matched by the model
    
    Args:
        all_comps: List of all test compositions
        matched_comps: List of compositions that were matched
    
    Returns:
        List of unmatched compositions
    """
    matched_keys = {f"{m['pii']}:{m['composition']}" for m in matched_comps}
    unmatched = []
    
    for comp in all_comps:
        comp_key = f"{comp['pii']}:{comp['composition']}"
        if comp_key not in matched_keys:
            unmatched.append(comp)
            
    return unmatched


if __name__ == "__main__":
    # Configuration
    model_output_file = "consolidated_OGpipeline-AdditionalGPTdata.json"
    test_set_dir = "test-set-OG"
    llm_model = "custom/gemini-2point0"  # Can also use "anthropic/claude-3-opus-20240229" or other models
    temperature = 0.0
    
    # Run evaluation
    stats, all_test_comps, matched_comps, judge_results = evaluate_with_llm_judge(
        model_output_file, test_set_dir, llm_model, temperature
    )
    
    # Print results
    print("\n===== EVALUATION RESULTS =====")
    print(f"Judge model: {llm_model}")
    print(f"Total test files processed: {stats['total_test_files']}")
    print(f"Total unique compositions in test set: {stats['total_unique_compositions_in_test_set']}")
    print(f"Total matched compositions found: {stats['total_matched_compositions']}")
    print(f"Match percentage: {stats['match_percentage']:.2f}%")
    print(f"Average confidence for matches: {stats['average_confidence']:.2f}%")
