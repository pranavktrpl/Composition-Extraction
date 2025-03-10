from numbers_parser import Document
import pandas as pd

def read_numbers_file(file_path):
    doc = Document(file_path)
    sheets = doc.sheets
    tables = sheets[0].tables
    data = tables[0].rows(values_only=True)
    return pd.DataFrame(data[1:], columns=data[0])

def match_papers_and_validate(file1_path, file2_path, compostionUndefined):
    # Read both Numbers files
    df1 = read_numbers_file(file1_path)
    df2 = read_numbers_file(file2_path)
    
    # Create a dictionary to store results
    results = {}
    
    # Iterate through rows in file2 
    for _, row2 in df2.iterrows():
        if row2['Classic-chemical-Composition Undefined/Partially Defined'] == compostionUndefined:
            # Find matching PII in file1
            matching_rows = df1[df1['PII'] == row2['PII']]
            
            # If a match is found, store the 'correct/incorrect' value
            if not matching_rows.empty:
                results[row2['PII']] = matching_rows['Correct/Incorrect'].values[0]
    
    return results

result_defined = match_papers_and_validate('outputs/gemini-flash-inContext-structured/gemini-response.numbers', 'prelim_analysis_Jan2025.numbers', 0)
result_UN_defined = match_papers_and_validate('outputs/gemini-flash-inContext-structured/gemini-response.numbers', 'prelim_analysis_Jan2025.numbers', 1)

print(len(result_defined)+len(result_UN_defined))

# Write results to a text file
with open('match_papers_validation_results.txt', 'w') as f:
    f.write("Composition Well-Defined\n")
    for pii, validation in result_defined.items():
        f.write(f"{pii}: {validation}\n")

    f.write("Composition Not-Defined\n")
    for pii, validation in result_UN_defined.items():
        f.write(f"{pii}: {validation}\n")
