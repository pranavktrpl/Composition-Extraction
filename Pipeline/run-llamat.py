import json
import os
import argparse
from tqdm import tqdm
from time import sleep
from time import time
from dotenv import load_dotenv
import llm.llm_utils as llm

# Add command line argument parser
parser = argparse.ArgumentParser(description='Run Llamat model with optional in-context examples')
parser.add_argument('--use-incontext', action='store_true', help='Use in-context examples in the prompt')
args = parser.parse_args()

load_dotenv()

pii_list = json.load(open('test_piis.json'))
target_directory = "outputs/llamat-2-chat"
os.makedirs(target_directory, exist_ok=True)

# Load and format in-context examples only if the flag is set
incontextExamples_prompt = ""
if args.use_incontext:
    incontext_examples_dir = os.path.join(os.path.dirname(__file__), "prompts/in-contextExamples")
    incontext_examples_files = [f for f in os.listdir(incontext_examples_dir) if f.endswith('.txt') and os.path.isfile(os.path.join(incontext_examples_dir, f))]
    for i,example_file in enumerate(incontext_examples_files):
        example_pii = example_file.split('.')[0]
        example_path = os.path.join(incontext_examples_dir, example_file)
        
        # Get research paper content
        research_text_path = os.path.join(os.path.dirname(__file__), f"prompting_data/research-paper-text/{example_pii}.txt")
        research_tables_path = os.path.join(os.path.dirname(__file__), f"prompting_data/research-paper-tables/{example_pii}.txt")
        matskraft_path = os.path.join(os.path.dirname(__file__), f"prompting_data/Matskraft-tables/{example_pii}.txt")
        
        if all(os.path.isfile(path) for path in [research_text_path, research_tables_path, matskraft_path]):
            with open(research_text_path) as f:
                research_text = f.read()
            with open(research_tables_path) as f:
                research_tables = f.read()
            with open(matskraft_path) as f:
                matskraft_table = f.read()
            with open(example_path) as f:
                example_output = f.read()
                
            example_prompt = f"<Example-Paper {i+1}>\nResearch Paper:\n'''\n{research_text}\n\n{research_tables}\n'''\nIncomplete Table:\n'''\n{matskraft_table}\n'''\nExample Output:\n'''\n{example_output}\n'''"
            incontextExamples_prompt += example_prompt + "\n\n"

for pii in tqdm(pii_list, desc="Processing PIIs"):
    try:
        matskraft_path = os.path.join(os.path.dirname(__file__), f"prompting_data/Matskraft-tables/{pii}.txt")
        research_tables_path = os.path.join(os.path.dirname(__file__), f"prompting_data/research-paper-tables/{pii}.txt")
        research_text_path = os.path.join(os.path.dirname(__file__), f"prompting_data/research-paper-text/{pii}.txt")        
        
        if os.path.isfile(matskraft_path):
            with open(matskraft_path) as f:
                incomplete_table = f.read()
        else:
            print(f"Skipping {pii}: Matskraft file missing")
            continue
        if os.path.isfile(research_tables_path):
            with open(research_tables_path) as f:
                research_paper_tables = f.read()
        else:
            print(f"Skipping {pii}: Research tables file missing")
            continue
        if os.path.isfile(research_text_path):
            with open(research_text_path) as f:
                research_paper_text = f.read()
        else:
            print(f"Skipping {pii}: Research text file missing")
            continue
        
        # Split research text into paragraphs
        paragraphs = [p.strip() for p in research_paper_text.split('\n\n') if p.strip()]
        
        # Create output file for this PII
        output_file = os.path.join(target_directory, f"{pii}.txt")
        metadata_file = os.path.join(target_directory, f"{pii}_metadata.json")
        
        # Initialize metadata
        metadata = {
            "paragraphs_processed": 0,
            "total_paragraphs": len(paragraphs),
            "processing_times": [],
            "use_incontext": args.use_incontext
        }
        
        # Process each paragraph
        for i, paragraph in enumerate(paragraphs, 1):
            try:
                start_time = time()
                response_text, _ = llm.llamat_fluid_incontext_prompts(
                    model="llamat",
                    temperature=0.0,
                    in_context_examples=incontextExamples_prompt if args.use_incontext else "",
                    research_paper_context=paragraph,
                    incomplete_table=incomplete_table
                )
                end_time = time()
                
                # Append response to output file
                with open(output_file, 'a') as f:
                    f.write(f"\n\n=== Paragraph {i} ===\n")
                    f.write(f"Input: {paragraph}\n")
                    f.write(f"Output: {response_text}\n")
                
                # Update metadata
                metadata["paragraphs_processed"] += 1
                metadata["processing_times"].append(end_time - start_time)
                
                # Save metadata after each paragraph
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=4)
                
                print(f"Processed paragraph {i}/{len(paragraphs)} for {pii}")
                
            except Exception as e:
                print(f"Error processing paragraph {i} for {pii}: {str(e)}")
                continue
        
        print(f"Completed processing {pii}")
        
    except Exception as e:
        print(f"Error processing {pii}: {str(e)}")
        continue
