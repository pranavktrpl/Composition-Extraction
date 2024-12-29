import os
import json
from dotenv import load_dotenv
import requests
from tqdm import tqdm 

load_dotenv()
Elsiver_API_KEY = os.getenv("ELSIVER_API_KEY")

print("Loading the pii's list")
pii_list = json.load(open('test_piis.json'))

os.makedirs('test_papers_plain', exist_ok=True)  # Ensure directory exists

print("Running for each pii loop")
for pii in tqdm(pii_list, desc="Downloading PLAIN files"):  # Wrap the loop with tqdm
    url = f'https://api.elsevier.com/content/article/pii/{pii}'
    headers = {
        'X-ELS-APIKey': Elsiver_API_KEY,
        'Accept': 'text/plain'
    }
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        if response.content.strip():  # Check if content is not empty
            with open(f'test_papers_plain/{pii}.txt', 'wb') as file:
                file.write(response.content)
        else:
            print(f'\nEmpty content received for PII: {pii}')
    else:
        print(f'\nFailed to retrieve article. Status code: {response.status_code}')
