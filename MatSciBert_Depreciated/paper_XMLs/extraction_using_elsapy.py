import os
import json
from elsapy.elsclient import ElsClient
from elsapy.elsdoc import FullDoc
from dotenv import load_dotenv

load_dotenv()
Elsiver_API_KEY = os.getenv("ELSIVER_API_KEY")

client = ElsClient(Elsiver_API_KEY)

pii_list = json.load(open('test_piis.json'))

for pii in pii_list:
    doc = FullDoc(sd_pii=pii)
    if doc.read(client):
        with open(f'test_papers/{pii}.xml', 'w', encoding='utf-8') as f:
            f.write(json.dumps(doc.data, indent=4))
        print(f"Downloaded and saved article with PII: {pii}")