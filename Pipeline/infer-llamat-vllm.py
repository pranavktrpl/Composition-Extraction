import torch
import vllm

# --- Force CPU Device ---
device = torch.device("cpu")
print("Forced CPU device. Running on CPU.")

# --- Model Initialization ---
# Replace this with the path to your model checkpoint.
checkpoint = "llamat-2-chat"

# Setup keyword arguments for initializing the vLLM client.
# Since we're running on CPU, we set tensor_parallel_size to 1.
kwargs = {
    "model": checkpoint,
    "tokenizer": checkpoint,
    "trust_remote_code": True,
    "tensor_parallel_size": 1,  # CPU only, so we use a single device.
    "seed": 42,
    "gpu_memory_utilization": 0.9,  # Not used on CPU but kept for compatibility.
}

print("Initializing model from checkpoint:", checkpoint)
client = vllm.LLM(**kwargs)
print("Model initialized successfully.")

# --- Prompt and Generation ---
# Define the prompt to test the model.
prompt = "Hi! Just trying to check are you working?"
print("Sending prompt to the model:", prompt)

# Generate the response with sampling parameters.
response = client.generate(
    [prompt],
    sampling_params=vllm.SamplingParams(
        skip_special_tokens=True,
        best_of=1,
        top_k=50,
        top_p=1.0,
        temperature=0.0,
        stop=["<|im_start|>", "<|im_end|>"],
        max_tokens=50
    )
)

# Extract and print the output text.
output = response[0].outputs[0].text
print("Model response:", output)
