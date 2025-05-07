import pandas as pd
import json
import os

# Load the pickle file
pickle_file = 'additional_composition_train_data_from_gpt4_6138.pkl'
print(f"Loading {pickle_file}...")
df = pd.read_pickle(pickle_file)

# Extract sentences
sentences = df['sentence'].tolist()

# Extract compositions and convert to string format
compositions = []
for comp_list in df['composition']:
    comp_strings = []
    for comp_group in comp_list:
        # Format each composition group like "(Element1, percentage1)(Element2, percentage2)..."
        comp_str = "".join([f"({element}, {percentage})" for element, percentage in comp_group])
        comp_strings.append(comp_str)
    # Join multiple composition groups with "|" if there are multiple
    compositions.append("|".join(comp_strings))

# Create the JSON objects
sentences_json = {"sentences": sentences}
compositions_json = {"compositions": compositions}

# Save to JSON files
with open('sentences.json', 'w', encoding='utf-8') as f:
    json.dump(sentences_json, f, ensure_ascii=False, indent=2)
    
with open('compositions.json', 'w', encoding='utf-8') as f:
    json.dump(compositions_json, f, ensure_ascii=False, indent=2)

print(f"Saved {len(sentences)} sentences to sentences.json")
print(f"Saved {len(compositions)} compositions to compositions.json")
