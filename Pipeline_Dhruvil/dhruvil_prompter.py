import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class DhruvilPrompter:
    def __init__(self):
        # First check if MPS is available
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load initial classifier model and tokenizer
        print("Loading composition present classifier model and tokenizer...")
        self.initial_classifier_model = AutoModelForSeq2SeqLM.from_pretrained("path/to/initial/classifier/model")
        self.initial_classifier_tokenizer = AutoTokenizer.from_pretrained("path/to/initial/classifier/model")
        self.initial_classifier_model = self.initial_classifier_model.to(self.device)

        # Load classifier model and tokenizer
        print("Loading Direct vs Equational classifier model and tokenizer...")
        self.classifier_model = AutoModelForSeq2SeqLM.from_pretrained("path/to/classifier/model")
        self.classifier_tokenizer = AutoTokenizer.from_pretrained("path/to/classifier/model")
        self.classifier_model = self.classifier_model.to(self.device)

        # Load direct model and tokenizer
        print("Loading direct model and tokenizer...")
        self.direct_model = AutoModelForSeq2SeqLM.from_pretrained("path/to/direct/model")
        self.direct_tokenizer = AutoTokenizer.from_pretrained("path/to/direct/model")
        self.direct_model = self.direct_model.to(self.device)

        # Load equational model and tokenizer
        print("Loading equational model and tokenizer...")
        self.equational_model = AutoModelForSeq2SeqLM.from_pretrained("path/to/equational/model")
        self.equational_tokenizer = AutoTokenizer.from_pretrained("path/to/equational/model")
        self.equational_model = self.equational_model.to(self.device)

    def __call__(self, prompt):
        # First, check if we should process this input
        initial_inputs = self.initial_classifier_tokenizer(prompt, return_tensors="pt")
        initial_inputs = {k: v.to(self.device) for k, v in initial_inputs.items()}
        
        initial_outputs = self.initial_classifier_model.generate(
            **initial_inputs,
            temperature=0.0,
        )
        initial_classification = self.initial_classifier_tokenizer.batch_decode(initial_outputs, skip_special_tokens=True)[0]
        
        # If initial classification is 0, return empty string
        if initial_classification.strip() == "0":
            return ""

        # If we get here, proceed with the rest of the pipeline
        # First, classify the input
        classifier_inputs = self.classifier_tokenizer(prompt, return_tensors="pt")
        classifier_inputs = {k: v.to(self.device) for k, v in classifier_inputs.items()}
        
        classifier_outputs = self.classifier_model.generate(
            **classifier_inputs,
            temperature=0.0,
        )
        classification = self.classifier_tokenizer.batch_decode(classifier_outputs, skip_special_tokens=True)[0]
        
        # Based on classification, choose the appropriate model
        if "direct" in classification.lower():
            model = self.direct_model
            tokenizer = self.direct_tokenizer
        else:  # equational
            model = self.equational_model
            tokenizer = self.equational_tokenizer
        
        # Generate final output
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        print("Generating outputs...")
        outputs = model.generate(
            **inputs,
            # max_length=150,        # Longer for complex chemical descriptions
            temperature=0.0,        # Lower temperature for more precise outputs
            # top_p=0.95,            # High top_p for scientific accuracy
            # do_sample=True,
        )
        
        return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
