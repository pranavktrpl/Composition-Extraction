import json
import os
from pathlib import Path

def calculate_total_processing_time():
    # Path to the outputs directory
    output_dir = Path("outputs/dhruvil-t5")
    
    # Initialize total time
    total_time = 0
    total_sentences = 0
    processed_files = 0
    
    # Iterate through all metadata files
    for metadata_file in output_dir.glob("*_metadata.json"):
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                
            # Sum up processing times
            file_processing_time = sum(metadata["processing_times"])
            total_time += file_processing_time
            total_sentences += metadata["sentences_processed"]
            processed_files += 1
            
            print(f"File: {metadata_file.stem}")
            print(f"Processing time: {file_processing_time:.2f} seconds")
            print(f"Sentences processed: {metadata['sentences_processed']}")
            print("-" * 50)
            
        except Exception as e:
            print(f"Error processing {metadata_file}: {str(e)}")
    
    # Print summary
    print("\nSummary:")
    print(f"Total files processed: {processed_files}")
    print(f"Total sentences processed: {total_sentences}")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Average time per sentence: {total_time/total_sentences:.2f} seconds")
    print(f"Average time per file: {total_time/processed_files:.2f} seconds")

if __name__ == "__main__":
    calculate_total_processing_time() 