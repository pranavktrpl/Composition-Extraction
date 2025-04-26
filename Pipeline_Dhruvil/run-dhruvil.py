import json
import os
# import nltk
from tqdm import tqdm
from time import time
from dhruvil_prompter_new import DhruvilPrompter
import spacy
from spacy.language import Language

# # Download NLTK data if not already downloaded
# try:
#     nltk.data.find('tokenizers/punkt')
# except LookupError:
#     nltk.download('punkt')

pii_list = json.load(open('test_piis.json'))
target_directory = "outputs/dhruvil-t5"
os.makedirs(target_directory, exist_ok=True)

# Initialize the Dhruvil model with the required model paths
dhruvil_model = DhruvilPrompter(
    composition_classifier_path="mtp_trainClassifierWithout100_ratio1to6_run1_FlanT5Large.pt",
    direct_equational_classifier_path="mtp_trainClassifierWithout100_dm_vs_eqn_run1_FlanT5Large.pt",
    direct_extractor_path="mtp_CompExtractor_Without100_FlanT5Large_DirectMatch_AdditionalGpt4Data_run_2.pt",
    equational_extractor_path="mtp_CompExtractor_Without100_FlanT5Large_OnlyEqn_AdditionalGpt4Data_run_2.pt"
)

@Language.component("scientific_sentence_boundaries")
def scientific_sentence_boundaries(doc):
    """Custom component for handling scientific text sentence boundaries"""
    for token in doc:
        # Don't split on periods in chemical formulas
        if token.text.endswith('.'):
            if any(char.isupper() for char in token.text[:-1]):  # Chemical formula
                token.is_sent_start = False
            if any(char.isdigit() for char in token.text):  # Measurement
                token.is_sent_start = False
                
        # Don't split on chemical formula patterns
        if token.i > 0:
            prev_token = doc[token.i - 1]
            # Pattern like "H2O" or "NaCl"
            if (prev_token.text[0].isupper() and 
                any(c.isdigit() for c in prev_token.text)):
                token.is_sent_start = False
    return doc

def split_sentences_scientific(text):
    """Split scientific text into sentences using spaCy with custom rules"""
    # Load the small English model
    nlp = spacy.load("en_core_web_sm")
    
    # Add custom scientific text component
    nlp.add_pipe("scientific_sentence_boundaries", before="parser")
    
    doc = nlp(text)
    return [str(sent) for sent in doc.sents]

for pii in tqdm(pii_list, desc="Processing PIIs"):
    try:
        # research_tables_path = os.path.join(os.path.dirname(__file__), f"prompting_data/research-paper-tables/{pii}.txt")
        research_text_path = os.path.join(os.path.dirname(__file__), f"Prompting_data/research-paper-text/{pii}.txt")        
        
        # if not all(os.path.isfile(path) for path in [research_tables_path, research_text_path]):
        #     print(f"Skipping {pii}: One or more required files missing")
        #     continue

        if not all(os.path.isfile(path) for path in [research_text_path]):
            print(f"Skipping {pii}: One or more required files missing")
            continue

        with open(research_text_path) as f:
            research_paper_text = f.read()
        
        # Split research text into sentences using spaCy
        sentences = split_sentences_scientific(research_paper_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Create output file for this PII
        output_file = os.path.join(target_directory, f"{pii}.json")
        metadata_file = os.path.join(target_directory, f"{pii}_metadata.json")
        
        # Initialize output dictionary
        outputs = {}
        
        # Initialize metadata
        metadata = {
            "sentences_processed": 0,
            "total_sentences": len(sentences),
            "processing_times": []
        }
        
        # Process each sentence
        for i, sentence in enumerate(sentences, 1):
            try:
                start_time = time()
                
                # Pass the sentence directly to the model without any additional prompting
                response_text = dhruvil_model(sentence)
                end_time = time()
                
                # Append response to output file
                # with open(output_file, 'a') as f:
                #     f.write(f"\n\n=== Sentence {i} ===\n")
                #     f.write(f"Input: {sentence}\n")
                #     f.write(f"Output: {response_text}\n")
                
                # Store output in dictionary
                outputs[f"output_{i}"] = response_text

                
                # Update metadata
                metadata["sentences_processed"] += 1
                metadata["processing_times"].append(end_time - start_time)
                
                # Save outputs and metadata after each sentence
                with open(output_file, 'w') as f:
                    json.dump(outputs, f, indent=4)
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=4)
                
                print(f"Processed sentence {i}/{len(sentences)} for {pii}")
                
            except Exception as e:
                print(f"Error processing sentence {i} for {pii}: {str(e)}")
                continue
        
        print(f"Completed processing {pii}")
        
    except Exception as e:
        print(f"Error processing {pii}: {str(e)}")
        continue 