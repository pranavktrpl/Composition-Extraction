import os
import json
from dotenv import load_dotenv
import requests

load_dotenv()
Elsiver_API_KEY = os.getenv("ELSIVER_API_KEY")

pii_list = json.load(open('test_piis.json'))

for pii in pii_list:
    url = f'https://api.elsevier.com/content/article/pii/{pii}'
    headers = {
        'X-ELS-APIKey': Elsiver_API_KEY,
        'Accept': 'application/plain'
    }
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        with open(f'test_papers/{pii}.xml', 'wb') as file:
            file.write(response.content)
        print(f'Successfully downloaded the full-text XML for PII: {pii}')
    else:
        print(f'Failed to retrieve article. Status code: {response.status_code}')
