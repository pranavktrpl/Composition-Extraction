import json
import os
from tqdm import tqdm
from time import sleep
from time import time
from dotenv import load_dotenv
import llm.llm_utils as llm

load_dotenv()

pii_list = json.load(open('test_piis.json'))
target_directory = "outputs/gemini-1.5"
os.makedirs(target_directory, exist_ok=True)

incontextExamples_prompt = ""
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

for pii in tqdm(pii_list, desc="Prompting the model for all the pii's"):
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
        
        context = f"{research_paper_text}\n\n{research_paper_tables}"

        max_retries = 3
        for attempt in range(max_retries):
            try:
                start_time = time()
                json_response, metadata = llm.fluid_incontext_prompts(
                    model="gpt-4o",
                    temperature=0.0,
                    in_context_examples=incontextExamples_prompt, 
                    research_paper_context=context,
                    incomplete_table=incomplete_table
                )
                end_time = time()
                
                # Save response with metadata as JSON
                output = {
                    "metadata": metadata,
                    "processing-time": (end_time - start_time),
                    "response": json_response
                }
                output_file = os.path.join(target_directory, f"{pii}.json")
                with open(output_file, 'w') as f:
                    json.dump(output, f, indent=4, ensure_ascii=False)
                
                print(f"Response for {pii} saved to {target_directory}")
                break

            except Exception as e:
                if attempt < max_retries - 1:
                    sleep_time = 3 * (attempt)
                    print(f"Error processing {pii} (attempt {attempt + 1}/{max_retries}): {str(e)}")
                    print(f"Retrying in {sleep_time} seconds...")
                    sleep(sleep_time)
                else:
                    print(f"Failed to process {pii} after {max_retries} attempts: {str(e)}")
        
        # sleep(0)  # Wait between requests

    except Exception as e:
        print(f"Error processing {pii}: {str(e)}")
        continue
