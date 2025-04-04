import json
import os

def clean_json_files():
    # Get all JSON files in the current directory
    json_files = [f for f in os.listdir('.') if f.endswith('.json')]
    
    for filename in json_files:
        try:
            # Read the JSON file
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Keep only the response field
            cleaned_data = {'response': data['response']}
            
            # Write back the cleaned data
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(cleaned_data, f, indent=4)
                
            print(f"Cleaned {filename}")
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    clean_json_files()