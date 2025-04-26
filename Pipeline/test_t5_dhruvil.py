import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# First check if MPS is available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load model and tokenizer
print("Loading model and tokenizer...")
model = AutoModelForSeq2SeqLM.from_pretrained("mtp_CompExtractor_Without100_FlanT5Large_DirectMatch_run_2.pt")
tokenizer = AutoTokenizer.from_pretrained("mtp_CompExtractor_Without100_FlanT5Large_DirectMatch_run_2.pt")

# Move model to device
model = model.to(device)

# Create input and move to device
inputs = tokenizer("In hexagonal perovskite materials, with a wide range of reports on B-deficient ceramics with the general formula AnBn-1O3n, e.g., Ba3LaNb3O12, Ba5Nb4O15, Ba8MTa6O24 (M = Zn, Co, Ni), Ba8M'Nb6O24 (M′ = Zn, Co, Mn, Fe), etc", return_tensors="pt")
# Move all input tensors to the same device as the model
inputs = {k: v.to(device) for k, v in inputs.items()}

print("Generating output...")
outputs = model.generate(
                        **inputs,
                        max_length=150,        # Longer for complex chemical descriptions
                        temperature=0.3,        # Lower temperature for more precise outputs
                        top_p=0.95,            # High top_p for scientific accuracy
                        do_sample=True,
                        )
# Move outputs back to CPU for decoding if needed
# outputs = outputs.cpu()

print("Generated text:")
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))