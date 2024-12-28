import os
import pickle
import argparse

from bs4 import BeautifulSoup
from tqdm import tqdm

from utils.mit_table_extractor import TableExtractor


def main(input_dir, output_dir):
    """
    Processes XML files from the input directory, extracts tables, and saves them as TXT files in the output directory.

    Args:
        input_dir (str): Path to the directory containing all XML files.
        output_dir (str): Path to the directory where extracted TXT files will be saved.
    """
    # Ensure the output directory exists; if not, create it
    os.makedirs(output_dir, exist_ok=True)

    te = TableExtractor()
    pii_table_dict = dict()

    # List all XML files in the input directory
    xml_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith('.xml')])

    # Iterate over each XML file with a progress bar
    for xml_file in tqdm(xml_files, desc="Processing XML files"):
        # Construct full path to the current XML file
        xml_path = os.path.join(input_dir, xml_file)
        
        # Define the output TXT file path with the same base name
        base_name = os.path.splitext(xml_file)[0]
        txt_path = os.path.join(output_dir, f"{base_name}.txt")
        
        # If the output TXT file already exists, remove it to ensure fresh processing
        if os.path.exists(txt_path):
            os.remove(txt_path)
        
        doi = None
        # Open and parse the XML file to extract DOI
        with open(xml_path, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file.read(), 'xml')
            doi_tags = soup.find_all('xocs:doi')
            if doi_tags:
                doi = doi_tags[0].text.strip()
        
        # Ensure DOI is found; if not, skip this file with a warning
        if doi is None:
            print(f"Warning: DOI not found in {xml_file}. Skipping this file.")
            continue
        
        te.doi = doi
        
        # Extract tables, captions, and footers using TableExtractor
        tables, captions, footers = te.get_xml_tables(xml_path)
        
        # Validate that the number of tables, captions, and footers match
        if not (len(tables) == len(captions) == len(footers)):
            print(f"Warning: Mismatch in counts for {xml_file}. Skipping this file.")
            continue
        
        processed_tables = []
        # Process each table along with its caption and footer
        for table, caption, footer in zip(tables, captions, footers):
            # Check for trivial table (only one cell)
            if len(table) == 1 and len(table[0]) == 1:
                print(f"Warning: Trivial table in {xml_file}: {table[0][0]}")
                # Continue processing to include the trivial table
                # Do not skip it; no 'continue' statement here
            
            # Determine the maximum number of columns in the current table
            max_cols = max(len(row) for row in table)
            # Pad each row with empty strings to ensure uniform column count
            for row in table:
                row += [''] * (max_cols - len(row))
            
            # Prepare a dictionary with extracted table information
            table_info = {
                'doi': doi,
                'act_table': table,
                'caption': caption if caption is not None else '',
                'footer': footer if footer is not None else dict(),
            }
            processed_tables.append(table_info)
        
        # Ensure that at least one valid table was extracted
        if not processed_tables:
            print(f"No valid tables found in {xml_file}.")
            continue
        
        # Save the processed tables to the TXT file
        with open(txt_path, 'w', encoding='utf-8') as txt_file:
            for idx, table_info in enumerate(processed_tables, 1):
                txt_file.write(f"Table {idx}:\n")
                txt_file.write(f"DOI: {table_info['doi']}\n")
                txt_file.write(f"Caption: {table_info['caption']}\n")
                txt_file.write(f"Footer: {table_info['footer']}\n")
                txt_file.write("Table Data:\n")
                for row in table_info['act_table']:
                    txt_file.write('\t'.join(str(cell) if cell is not None else '' for cell in row) + '\n')
                txt_file.write('\n')  # Add a newline for separation between tables
        
        # Optionally, store the processed tables in a dictionary for further use
        pii_table_dict[base_name] = processed_tables

    # Save the entire dictionary of tables to a pickle file in the output directory
    pickle_path = os.path.join(output_dir, 'pii_table_dict.pkl')
    with open(pickle_path, 'wb') as pkl_file:
        pickle.dump(pii_table_dict, pkl_file)
    
    print(f"Processing complete. Extracted tables are saved in {output_dir}.")


if __name__ == "__main__":
    # Set up argument parsing for input and output directories
    parser = argparse.ArgumentParser(description="Extract tables from XML files and save them as TXT files.")
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help="Path to the directory containing all XML files."
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help="Path to the directory where TXT files will be saved."
    )
    args = parser.parse_args()
    
    main(args.input_dir, args.output_dir)
