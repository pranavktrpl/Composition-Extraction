import json
import os
from pathlib import Path

def extract_compositions(input_directory, output_file):
    """
    Extract non-empty compositions from all JSON files in the input directory
    and save them in a consolidated JSON file.
    
    Args:
        input_directory (str): Path to directory containing PII JSON files
        output_file (str): Path to save the consolidated output JSON file
    """
    # Dictionary to store all compositions
    all_compositions = {}
    
    # Process each JSON file in the directory
    for json_file in Path(input_directory).glob('*.json'):
        # Skip metadata files
        if json_file.name.endswith('_metadata.json'):
            continue
            
        pii = json_file.stem  # Get PII from filename
        print(f"Processing {pii}...")
        
        try:
            # Read the JSON file
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Extract non-empty compositions
            compositions = []
            for output_key, composition in data.items():
                if composition:  # If not empty
                    compositions.extend(composition)
            
            # Only add PIIs that have compositions
            if compositions:
                all_compositions[pii] = compositions
                print(f"Found {len(compositions)} compositions in {pii}")
            
        except Exception as e:
            print(f"Error processing {pii}: {str(e)}")
            continue
    
    # Save consolidated results
    print(f"\nSaving results for {len(all_compositions)} PIIs...")
    with open(output_file, 'w') as f:
        json.dump(all_compositions, f, indent=2)
    
    print(f"Results saved to {output_file}")
    
    # Print summary statistics
    total_compositions = sum(len(comps) for comps in all_compositions.values())
    print(f"\nSummary:")
    print(f"Total PIIs with compositions: {len(all_compositions)}")
    print(f"Total compositions found: {total_compositions}")
    print(f"Average compositions per PII: {total_compositions/len(all_compositions) if all_compositions else 0:.2f}")

if __name__ == "__main__":
    # Example usage
    input_dir = "outputs/dhruvil-t5"  # Directory containing your PII JSON files
    output_file = "consolidated_compositions.json"
    extract_compositions(input_dir, output_file)
