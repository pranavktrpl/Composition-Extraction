import json

def count_empty_outputs():
    # Load the JSON file
    with open('cleaned_test_set.json', 'r') as f:
        data = json.load(f)
    
    total_compositions = 0
    empty_outputs = 0
    
    # Iterate through each PII and its compositions
    for pii, compositions in data.items():
        total_compositions += len(compositions)
        for comp in compositions:
            if not comp["model_output"] or comp["model_output"] == [[]]:
                empty_outputs += 1
    
    print(f"Total compositions: {total_compositions}")
    print(f"Empty outputs: {empty_outputs}")
    print(f"Percentage empty: {(empty_outputs/total_compositions)*100:.2f}%")
    print(f"Number of PIIs: {len(data)}")

# Run the analysis
count_empty_outputs()