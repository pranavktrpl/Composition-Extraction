import json
import os
from pathlib import Path

def sum_json_stats(directory):
    total_prompt_tokens = 0
    total_candidate_tokens = 0
    total_tokens = 0
    total_processing_time = 0.0
    
    # Iterate through all JSON files in the directory
    for json_file in Path(directory).glob('*.json'):
        with open(json_file, 'r') as f:
            data = json.load(f)
            
            # Sum up the values
            total_prompt_tokens += data['metadata']['promptTokenCount']
            total_candidate_tokens += data['metadata']['candidatesTokenCount']
            total_tokens += data['metadata']['totalTokenCount']
            total_processing_time += data['processing-time']
    
    # Create the report content
    report_content = f"""Token and Processing Time Report:
Total Prompt Tokens: {total_prompt_tokens}
Total Candidate Tokens: {total_candidate_tokens}
Total Tokens: {total_tokens}
Total Processing Time: {total_processing_time} seconds
"""
    
    # Save the report to a file
    report_path = Path(directory) / 'report.txt'
    with open(report_path, 'w') as report_file:
        report_file.write(report_content)
    
    return {
        'total_prompt_tokens': total_prompt_tokens,
        'total_candidate_tokens': total_candidate_tokens,
        'total_tokens': total_tokens,
        'total_processing_time': total_processing_time
    }

# Example usage
directory_path = 'outputs/gpt-4o-4context'
results = sum_json_stats(directory_path)
print(results)