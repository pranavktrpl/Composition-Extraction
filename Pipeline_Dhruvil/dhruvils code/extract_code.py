import os
import glob
import json

# Get all .py and .ipynb files in the current directory
py_files = glob.glob("*.py")
ipynb_files = glob.glob("*.ipynb")

# Process Python files - combine them into one file
if py_files:
    with open('combined_python.txt', 'w', encoding='utf-8') as outfile:
        for filename in py_files:
            # Write a header indicating the file name
            outfile.write(f"=== {filename} ===\n\n")
            try:
                with open(filename, 'r', encoding='utf-8') as infile:
                    content = infile.read()
                    outfile.write(content)
            except Exception as e:
                outfile.write(f"Error reading {filename}: {e}\n")
            outfile.write("\n\n")  # Add some spacing after each file's content
    print(f"Contents of {len(py_files)} Python files have been combined into 'combined_python.txt'.")

# Process Jupyter notebooks - create separate files for each
for notebook in ipynb_files:
    output_filename = f"{os.path.splitext(notebook)[0]}.txt"
    try:
        with open(notebook, 'r', encoding='utf-8') as infile:
            notebook_data = json.load(infile)
            
        with open(output_filename, 'w', encoding='utf-8') as outfile:
            for cell in notebook_data['cells']:
                if cell['cell_type'] == 'code':
                    # Write the code
                    outfile.write("=== CODE ===\n")
                    outfile.write(''.join(cell['source']))
                    outfile.write("\n\n")
                    
                    # # Write the outputs if they exist
                    # if cell['outputs']:
                    #     outfile.write("=== OUTPUT ===\n")
                    #     for output in cell['outputs']:
                    #         if output['output_type'] == 'stream':
                    #             outfile.write(''.join(output['text']))
                    #         elif output['output_type'] == 'execute_result':
                    #             if 'text/plain' in output['data']:
                    #                 outfile.write(''.join(output['data']['text/plain']))
                    #     outfile.write("\n\n")
                    # outfile.write("="*50 + "\n\n")
        
        print(f"Created {output_filename} from {notebook}")
    except Exception as e:
        print(f"Error processing {notebook}: {e}")
