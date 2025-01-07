import json
import os
from tqdm import tqdm
from time import sleep
from time import time
from dotenv import load_dotenv
import llm.llm_utils as llm

load_dotenv()

pii_list = json.load(open('test_piis.json'))
target_directory = "outputs/gemini-flash-inContext-structured"
os.makedirs(target_directory, exist_ok=True)

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

        start_time = time()
        json_response, metadata = llm.complete_the_table_in_context(
            model="custom/gemini-flash",
            temperature=0.0,
            research_paper_context=context,
            incomplete_table=incomplete_table
        )
        end_time = time()
        
        combined_output = {
            "metadata": metadata,
            "processing-time": (end_time - start_time),
            "response": json_response
        }

        output_file = os.path.join(target_directory, f"{pii}.json")
        with open(output_file, 'w') as f:
            json.dump(combined_output, f, indent=4, ensure_ascii=False)
        print(f"Response for {pii} saved to {output_file}")

        sleep(5)

    except Exception as e:
        print(f"Error processing {pii}: {e}")
