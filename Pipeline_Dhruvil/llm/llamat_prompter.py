import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
class llamatPrompter:
    def __init__(self):
        model_path = "llm/models/llamat-2-chat"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code = True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        self.model = self.model.to(torch.bfloat16).to("mps").eval() if torch.backends.mps.is_available() else self.model.to("cuda").eval() if torch.cuda.is_available() else self.model.to.torch.device("cpu")

    def __call__(self, prompt, images=None):

        prepare_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        bos_token_id = self.tokenizer.bos_token_id
        eos_token_id = self.tokenizer.eos_token_id

        outputs = self.model.generate(
            **prepare_inputs,
            max_length=7000,            # Maximum length of the sequence to generate
            num_return_sequences=1,     # Number of sequences to generate
            top_p=0.95,                 # Use nucleus sampling
            do_sample=True,             # Enable sampling for more diverse outputs
            bos_token_id=bos_token_id,  # Start-of-sequence token ID (if applicable)
            eos_token_id=eos_token_id,  # End-of-sequence token ID (if applicable)
        )
        
        return self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
