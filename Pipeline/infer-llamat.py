import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Check if MPS is available (only on Apple Silicon Macs with supported OS and PyTorch version)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print("Starting script...")
print(f"Using device: {device}")

# Path to your local model directory containing your downloaded files
model_dir = "llamat-2-chat"

# Load the tokenizer from the local directory.
# The 'trust_remote_code=True' flag is needed if the model uses custom classes.
print("Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

# Load the causal language model from the local directory.
print("Loading the model...")
model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)

# Move the model to the selected device (MPS or CPU)
print(f"Moving model to {device}...")
if device.type == "mps":
    model = model.half()
model.to(device)

# Prepare a conversation-style prompt using special tokens.
prompt = (
    "<|im_start|>system\nYou are a helpful assistant.\n<|im_end|>\n"
    "<|im_start|>question\nHi! Just trying to check are you working?\n<|im_end|>\n"
    "<|im_start|>answer\n"
)

# Tokenize the prompt.
inputs = tokenizer(prompt, return_tensors="pt").to(device)
print(f"inputs = {inputs}")

# Retrieve special token IDs if defined in the tokenizer.
bos_token_id = tokenizer.bos_token_id
eos_token_id = tokenizer.eos_token_id

# Generate text using the model with adjusted parameters.
print("Generating output.......")
outputs = model.generate(
    **inputs,
    max_length=100,             # Maximum length of the sequence to generate
    num_return_sequences=1,     # Number of sequences to generate
    top_p=0.95,                 # Use nucleus sampling
    do_sample=True,             # Enable sampling for more diverse outputs
    bos_token_id=bos_token_id,  # Start-of-sequence token ID (if applicable)
    eos_token_id=eos_token_id,  # End-of-sequence token ID (if applicable)
)

# Decode the generated tokens back to a string.
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated Text:")
print(generated_text)
