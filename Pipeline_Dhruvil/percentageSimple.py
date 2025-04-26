import json

import os
from pathlib import Path
import time
from typing import Dict, List, Any
from tqdm import tqdm

from llm.llm_utils import simple_prompt

def create_complexity_prompt(composition: str) -> str:
    """
    Create a prompt for the LLM to determine if a composition is simple or complex
    
    Args:
        composition: The material composition to evaluate
        
    Returns:
        A formatted prompt string for the LLM
    """
    prompt = f"""You are a chemistry expert analyzing material compositions.

COMPOSITION:
{composition}

Your task is to determine if this is a simple composition (like a single element or simple compound with no doping) 
or a complex composition (with doping, multiple phases, complex structure descriptions, etc.).

Examples of SIMPLE compositions:
- Si
- TiO2
- Fe3O4
- SiO2
- CaCO3

Examples of COMPLEX compositions:
- (La0.8Sr0.2)0.95MnO3±δ
- Si doped with B (1%)
- 70% TiO2 - 30% SiO2
- Ni-5wt%Al-5wt%Mo
- Li1.3Al0.3Ti1.7(PO4)3 doped with 0.5 wt% SiO2

Respond with a JSON object with this format:
{{
  "is_simple": true/false,
  "confidence": 0-100,
  "reasoning": "Brief explanation of your judgment"
}}

Only return the JSON object, no other text."""

    return prompt

def analyze_composition_complexity(test_set_dir: str, 
                           llm_model: str = "custom/gemini-2point0", 
                           temperature: float = 0.0,
                           max_test_files: int = None) -> Dict:
    """
    Analyze compositions in test set and determine if they are simple or complex
    
    Args:
        test_set_dir: Path to directory containing test set JSON files
        llm_model: LLM model to use
        temperature: Temperature for LLM generation
        max_test_files: Optional limit on number of test files to process
        
    Returns:
        Dictionary containing analysis statistics
    """
    # Create a timestamped output directory for incremental saves
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = f"complexity_analysis_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create checkpoint and results files
    checkpoint_file = os.path.join(output_dir, "checkpoint.json")
    results_file = os.path.join(output_dir, "complexity_results.json")
    
    # Check if we have an existing checkpoint to resume from
    processed_comps = set()
    all_compositions = []
    complexity_results = []
    
    if os.path.exists(checkpoint_file):
        print(f"Found checkpoint file at {checkpoint_file}, resuming from last state...")
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
                processed_comps = set(checkpoint_data.get("processed_comps", []))
                all_compositions = checkpoint_data.get("all_compositions", [])
                complexity_results = checkpoint_data.get("complexity_results", [])
                print(f"Resuming with {len(processed_comps)} already processed compositions")
        except Exception as e:
            print(f"Error loading checkpoint: {str(e)}. Starting fresh.")
            processed_comps = set()
            all_compositions = []
            complexity_results = []
    
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
            
            # Process each composition
            for comp_idx, composition in enumerate(test_compositions):
                # Create a unique identifier for this composition
                comp_id = f"{pii}:{composition}"
                
                # Skip if already processed
                if comp_id in processed_comps:
                    print(f"  Skipping already processed composition: {composition}")
                    continue
                
                # Add to master list for tracking
                all_compositions.append({
                    "pii": pii,
                    "composition": composition
                })
                
                # Create prompt for LLM
                complexity_prompt = create_complexity_prompt(composition)
                
                # Call LLM
                try:
                    print(f"  Analyzing complexity of: {composition}")
                    llm_response, _ = simple_prompt(llm_model, temperature, complexity_prompt)
                    
                    # Parse response
                    try:
                        # Clean up the response in case it contains markdown or extra text
                        llm_response = llm_response.strip()
                        if "```json" in llm_response:
                            llm_response = llm_response.split("```json")[1].split("```")[0].strip()
                        elif "```" in llm_response:
                            llm_response = llm_response.split("```")[1].split("```")[0].strip()
                        
                        result = json.loads(llm_response)
                        
                        # Store result
                        result_entry = {
                            "pii": pii,
                            "composition": composition,
                            "is_simple": result["is_simple"],
                            "confidence": result["confidence"],
                            "reasoning": result["reasoning"]
                        }
                        
                        complexity_results.append(result_entry)
                        
                        if result["is_simple"]:
                            print(f"  ✓ Composition is SIMPLE (confidence: {result['confidence']}%)")
                        else:
                            print(f"  ✗ Composition is COMPLEX (confidence: {result['confidence']}%)")
                        
                    except json.JSONDecodeError:
                        print(f"  Error: Invalid JSON response from LLM: {llm_response}")
                        
                except Exception as e:
                    print(f"  Error with LLM: {str(e)}")
                    # Add a small delay to avoid rate limits
                    time.sleep(2)
                    continue
                
                # Mark as processed
                processed_comps.add(comp_id)
                
                # Save incremental results after each composition
                checkpoint_data = {
                    "processed_comps": list(processed_comps),
                    "all_compositions": all_compositions,
                    "complexity_results": complexity_results,
                    "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint_data, f, indent=2)
                
                # Calculate interim statistics
                simple_count = sum(1 for r in complexity_results if r["is_simple"])
                complex_count = len(complexity_results) - simple_count
                
                interim_stats = {
                    "total_compositions_processed": len(complexity_results),
                    "simple_compositions": simple_count,
                    "complex_compositions": complex_count,
                    "simple_percentage": (simple_count / len(complexity_results) * 100) if complexity_results else 0,
                    "complex_percentage": (complex_count / len(complexity_results) * 100) if complexity_results else 0,
                    "average_confidence": sum(r["confidence"] for r in complexity_results) / len(complexity_results) if complexity_results else 0,
                    "all_results": complexity_results
                }
                
                with open(results_file, 'w') as f:
                    json.dump(interim_stats, f, indent=2)
                    
                # Small delay to avoid rate limits
                time.sleep(0.5)
                
        except Exception as e:
            print(f"Error processing {test_file}: {str(e)}")
            continue
    
    # Calculate final statistics
    simple_count = sum(1 for r in complexity_results if r["is_simple"])
    complex_count = len(complexity_results) - simple_count
    
    stats = {
        "total_compositions": len(complexity_results),
        "simple_compositions": simple_count,
        "complex_compositions": complex_count,
        "simple_percentage": (simple_count / len(complexity_results) * 100) if complexity_results else 0,
        "complex_percentage": (complex_count / len(complexity_results) * 100) if complexity_results else 0,
        "judge_model": llm_model,
        "average_confidence": sum(r["confidence"] for r in complexity_results) / len(complexity_results) if complexity_results else 0
    }
    
    # Save final results 
    final_results_file = os.path.join(output_dir, "final_results.json")
    with open(final_results_file, 'w') as f:
        json.dump({
            "statistics": stats,
            "simple_compositions": [r for r in complexity_results if r["is_simple"]],
            "complex_compositions": [r for r in complexity_results if not r["is_simple"]],
            "all_results": complexity_results
        }, f, indent=2)
    
    print(f"\nFinal results saved to {final_results_file}")
    
    return stats, complexity_results

if __name__ == "__main__":
    # Configuration
    test_set_dir = "test-set-OG"
    llm_model = "custom/gemini-2point0"
    temperature = 0.0
    
    # Run analysis
    stats, results = analyze_composition_complexity(
        test_set_dir, llm_model, temperature
    )
    
    # Print summary results
    print("\n===== COMPOSITION COMPLEXITY ANALYSIS =====")
    print(f"Judge model: {llm_model}")
    print(f"Total compositions analyzed: {stats['total_compositions']}")
    print(f"Simple compositions: {stats['simple_compositions']} ({stats['simple_percentage']:.2f}%)")
    print(f"Complex compositions: {stats['complex_compositions']} ({stats['complex_percentage']:.2f}%)")
    print(f"Average confidence: {stats['average_confidence']:.2f}%")
