import os
import pandas as pd
import json

# Load the JSON file containing PIIs and table numbers
with open('test_piis_w_table.json', 'r') as f:
    pii_table_data = json.load(f)

df1_filtered = pd.read_csv('Current_database_whole.csv', na_filter=False)

# Filter rows where 'Composition' is empty or '[]'
only_prop = df1_filtered[df1_filtered['Composition'] == '[]']

# Set display options for pandas
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)

output_directory = "Matskraft-tables"
# Loop through all entries in the JSON file
for entry in pii_table_data:
    article_pii = entry['pii']
    table_no = f"[{entry['table']}]"

    # Filter the DataFrame based on the current PII and table number
    filtered_df = only_prop[(only_prop['Article PII'] == article_pii) & (only_prop['Table No.'] == table_no)]
    
    if not filtered_df.empty:
        print(f"Processing Article PII: {article_pii}, Table No: {table_no}")

        # Update the 'Composition' column with placeholder values
        filtered_df['Composition'] = [f"<blank_{i+1}>" for i in range(len(filtered_df))]
        
        # Save the updated DataFrame to a text file
        output_dir = os.path.join(f'{output_directory}', f"{article_pii}.txt")
        os.makedirs(os.path.dirname(output_dir), exist_ok=True)
        with open(output_dir, 'w') as output_file:
            output_file.write(filtered_df.to_string(index=False))
    else:
        print(f"No matching rows found for Article PII: {article_pii}, Table No: {table_no}")
