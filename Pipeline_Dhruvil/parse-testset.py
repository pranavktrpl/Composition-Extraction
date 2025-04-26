import json
import os
from pathlib import Path
from tqdm import tqdm
from dhruvil_prompter_new import DhruvilPrompter

def process_test_set(test_dir, output_file):
    """
    Process all composition data in the test directory using Dhruvil's pipeline.
    
    Args:
        test_dir (str): Path to directory containing test JSON files
        output_file (str): Path to save the results
    """
    # Initialize Dhruvil's pipeline
    dhruvil_pipeline = DhruvilPrompter(
        composition_classifier_path="mtp_trainClassifierWithout100_ratio1to6_run1_FlanT5Large.pt",
        direct_equational_classifier_path="mtp_trainClassifierWithout100_dm_vs_eqn_run1_FlanT5Large.pt",
        direct_extractor_path="mtp_CompExtractor_Without100_FlanT5Large_DirectMatch_AdditionalGpt4Data_run_2.pt",
        equational_extractor_path="mtp_CompExtractor_Without100_FlanT5Large_OnlyEqn_AdditionalGpt4Data_run_2.pt"
    )
    
    # Dictionary to store all results
    results = {}
    
    # Get all JSON files in the test directory
    test_files = list(Path(test_dir).glob('*.json'))
    print(f"Found {len(test_files)} test files in {test_dir}")
    
    # Process each file
    for test_file in tqdm(test_files, desc="Processing test files"):
        pii = test_file.stem
        print(f"Processing {pii}...")
        
        try:
            # Load the test file
            with open(test_file, 'r') as f:
                test_data = json.load(f)
            
            # Extract compositions
            compositions = []
            if "response" in test_data and "Compositions" in test_data["response"]:
                for item in test_data["response"]["Compositions"]:
                    if "composition" in item:
                        comp_str = item["composition"]
                        
                        # Process composition through Dhruvil's pipeline
                        pipeline_result = dhruvil_pipeline(comp_str)
                        
                        # Store results
                        compositions.append({
                            "composition": comp_str,
                            "model_output": pipeline_result
                        })
                        
                        print(f"  Processed: {comp_str}")
                        print(f"  Pipeline result: {pipeline_result}")
            
            # Store results for this PII
            if compositions:
                results[pii] = compositions
                
        except Exception as e:
            print(f"Error processing {pii}: {str(e)}")
    
    # Save all results to the output file
    print(f"Saving results for {len(results)} PIIs to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Processing complete!")
    
    # Print summary statistics
    total_compositions = sum(len(comps) for comps in results.values())
    print(f"\nSummary:")
    print(f"Total PIIs processed: {len(results)}")
    print(f"Total compositions processed: {total_compositions}")
    print(f"Average compositions per PII: {total_compositions/len(results) if results else 0:.2f}")

if __name__ == "__main__":
    # Configuration
    test_dir = "test-set-OG"
    output_file = "model_test_outputs.json"
    
    # Process test set
    process_test_set(test_dir, output_file)