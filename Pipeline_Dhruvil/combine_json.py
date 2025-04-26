import os
import json

# Directory containing the JSON files
input_dir = 'test-set-OG'
output_file = 'combined_compositions.json'

# Initialize a set to track unique compositions
unique_compositions = set()

# Read each JSON file and extract compositions
for filename in os.listdir(input_dir):
    if filename.endswith('.json') and not filename.startswith('.'):
        file_path = os.path.join(input_dir, filename)
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # Extract compositions if they exist
            if 'response' in data and 'Compositions' in data['response']:
                compositions = data['response']['Compositions']
                for comp in compositions:
                    if 'composition' in comp:
                        unique_compositions.add(comp['composition'])
        except Exception as e:
            print(f"Error processing {filename}: {e}")

# Convert set to list and create the final JSON structure
combined_data = {"compositions": list(unique_compositions)}

# Save the combined data to a single JSON file
with open(output_file, 'w') as f:
    json.dump(combined_data, f, indent=2)

print(f"Combined {len(combined_data['compositions'])} unique compositions into {output_file}") 