import os
import json
from typing import Optional
from tqdm import tqdm
from llm.llm_utils import simple_prompt

def process_json_directory(
    input_dir: str,
    system_prompt: str,
    model: str = "custom/gemini-pro",
    temperature: float = 0.0,
    output_dir: Optional[str] = None
) -> None:
    """
    Process all JSON files in a directory using an LLM with a given system prompt.
    
    Args:
        input_dir (str): Path to directory containing JSON files
        system_prompt (str): Prompt to use for LLM processing
        model (str): Model to use for LLM completion
        temperature (float): Temperature parameter for LLM
        output_dir (Optional[str]): Directory to save processed files. If None, overwrites input files
    """
    # Create output directory if specified and doesn't exist
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get all JSON files in the directory
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    
    for json_file in tqdm(json_files, desc="Processing JSON files"):
        input_path = os.path.join(input_dir, json_file)
        
        # Read input JSON
        try:
            with open(input_path, 'r') as f:
                input_data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error reading {json_file}: {e}")
            continue
            
        # Combine prompt with JSON data
        combined_prompt = f"{system_prompt}\n\nInput JSON:\n{json.dumps(input_data, indent=2)}"
        
        try:
            # Process with LLM
            response_text, _ = simple_prompt(
                model=model,
                temperature=temperature,
                prompt=combined_prompt
            )
            
            # Clean up response and parse JSON
            if isinstance(response_text, str):
                # Remove any markdown code blocks if present
                response_text = response_text.replace('```json\n', '')
                response_text = response_text.replace('```', '')
                response_text = response_text.strip()
                
                processed_data = json.loads(response_text)
                
                # Determine output path
                output_path = os.path.join(output_dir if output_dir else input_dir, json_file)
                
                # Save processed JSON
                with open(output_path, 'w') as f:
                    json.dump(processed_data, indent=2, fp=f)
                    
                print(f"Successfully processed {json_file}")
                
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue

if __name__ == "__main__":
    # Directory containing JSON files
    input_directory = "test-set/not-processed"
    
    # Your system prompt
    system_prompt = """ You are an LLM specifically instructed to parse, convert, and format chemical composition data. Your tasks include:

1. Parsing Composition Strings  
   - Read each composition string (e.g., `"20Li2O–20PbO–45B2O3–15P2O5"`, `"Ti0.9:N1.1:Si0.14"`, `"Zr-2.5wt.%Nb"`, etc.).
   - Identify components, dopants, and their amounts (e.g., `"Li2O"`, `"Ti"`, `"Nb"`, `"B2O3"`, `"Cr2O3"`, `"Tb3+"`), even if they appear in mixed notations (e.g., `'wt%'`, `'at%'`, `'mol'`, `'mol%'`, or as raw ratios/doping levels).
   - Differentiate between dopant vs. host components if needed (for example, doping via `: xCr2O3` or `: x mol% B2O3`).

2. Handling Units and Conversions 
   - Units: If a composition is described in weight percentages, atomic percentages, or molar fractions, interpret and convert them to the specified target format. For example:
     - `X wt% → [("X", fraction, "wt")]`
     - `X mol → [("X", fraction, "mol")]`
     - `X at% → [("X", fraction, "at%")]`
   - Ratios to Percentages:** When given raw ratios (e.g., `"Ti0.9:N1.1:Si0.14"`), normalize them to sum to 100 or 1.0 and express the result as percentages or fractions as requested.

3. Standardized Output Format  
   - For each JSON entry under `"Compositions"`, you must produce a *list of tuples* describing each component.  
   - The tuple format is typically:
     \[
       ("component_name", numeric_amount, "unit")
     \]
   - The numeric amount may sum to `1.0` (if fractional form) or to `100` (if in percentage form).  
   - Example:
     ```json
     {
       "placeholder": "<blank_1>",
       "composition": "[('Li2O', 0.2, 'mol'), ('PbO', 0.2, 'mol'), ('B2O3', 0.45, 'mol'), ('P2O5', 0.15, 'mol')]"
     }
     ```

4. Consistency Checks  
   - Make sure the final compositions sum to 1.0, depending on the specified output.
   - For doping additions like `+0.1wt%Cr2O3`, split it into separate components in the final list:
     ```json
     "[('Host', 0.999, 'wt'), ('Cr2O3', 0.001, 'wt')]"
     ```
   - Ensure that doping or additive content is scaled from the original base composition if necessary.

5. **Error Handling**  
   - If an entry states `"NULL"` or has no composition, maintain that in the final JSON (e.g., `"composition": "NULL"`).
   - Preserve placeholders (e.g., `"<blank_1>"`, `"<blank_2>"`) without changing them.
   - Do not remove or alter the original placeholders or the main JSON structure.

6. No Extra Comments or Explanations  
   - Output only the updated JSON. 
   - You should not restate the entire text. 
   - Do not include additional commentary or warnings.

Overall Goal:  
- Input: A JSON structure containing placeholders and composition strings with varied units (wt%, mol, at%, doping additions, partial fractions).  
- Output: The same JSON structure, but each `"composition"` field must be converted into the standardized list-of-tuples notation, with correct amounts, units, and sum normalization.

---

Example of the LLM’s Behavior:
- Given: 
  ```json
  { 
    "response": {
      "Compositions": [
        {
          "placeholder": "<blank_1>",
          "composition": "Ti29Zr34Ni5.3Cu9Be22.7"
        }
      ]
    }
  }
  ```
- Output: 
  ```json
  {
    "response": {
      "Compositions": [
        {
          "placeholder": "<blank_1>",
          "composition": "[('Ti', 0.29, 'at'), ('Zr', 0.34, 'at'), ('Ni', 0.053, 'at'), ('Cu', 0.09, 'at'), ('Be', 0.227, 'at')]"
        }
      ]
    }
  }
  ```

“You are a specialized LLM. Your job is to read the following JSON object, parse and interpret each composition string, then rewrite them into `[('Component', fraction, 'unit')]` tuples. Each composition should sum to 1.0 (or 100%) in the required unit. Maintain placeholders and the overall JSON structure. For doping or additives, separate them as new tuples. Return only the updated JSON with no extra comments.”
"""

    # Process the JSONs
    process_json_directory(
        input_dir=input_directory,
        system_prompt=system_prompt,
        model="custom/gemini-pro",  # You can change the model if needed
        temperature=0.0             # You can adjust temperature if needed
    )