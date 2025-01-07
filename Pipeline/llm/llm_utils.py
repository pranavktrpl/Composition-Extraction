from .callLLM import text_completion, structured_completion, StructuredRequest, CompletionRequest
import os

with open(os.path.join(os.path.dirname(__file__), "../prompts/complete_the_table_w_in_context.txt")) as f:
    complete_the_table_w_in_context_prompt = f.read()

with open(os.path.join(os.path.dirname(__file__), "../prompts/complete_the_table_w_in_context_for_schema.txt")) as f:
    complete_the_table_w_in_context_schema_prompt = f.read()


with open(os.path.join(os.path.dirname(__file__), "../prompts/complete_the_table_w_out_in_context.txt")) as f:
    complete_the_table_w_out_in_context_prompt = f.read()


# with open(os.path.join(os.path.dirname(__file__), "../prompting_data/Matskraft-tables/S0167577X06001327.txt")) as f:
#     incomplete_table = f.read()

# with open(os.path.join(os.path.dirname(__file__), "../prompting_data/research-paper-tables/S0167577X06001327.txt")) as f:
#     research_paper_tables = f.read()

# with open(os.path.join(os.path.dirname(__file__), "../prompting_data/research-paper-text/S0167577X06001327.txt")) as f:
#     research_paper_text = f.read()


# context = f"{research_paper_text}\n\n{research_paper_tables}"


def complete_the_table_in_context(model: str, temperature: float, research_paper_context: str, incomplete_table: str):
    prompt = complete_the_table_w_in_context_schema_prompt.replace("{{Research Paper}}", research_paper_context)
    prompt = prompt.replace("{{Table}}", incomplete_table)
    
    return structured_completion(StructuredRequest(
        model=model,
        text=prompt,
        temperature=temperature,
        schema={
            "Compositions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "placeholder": {"type": "string", "description": "Placeholder in the table (e.g., <blank_1>)"},
                        "composition": {"type": "string", "description": "Extracted composition corresponding to the placeholder"}
                    },
                    "required": ["placeholder", "composition"]
                }
            }
        }
    ))

def complete_the_table_in_context_no_schema(model: str, temperature: float, research_paper_context: str, incomplete_table: str):
    prompt = complete_the_table_w_in_context_prompt.replace("{{Research Paper}}", research_paper_context)
    prompt = prompt.replace("{{Table}}", incomplete_table)
    return text_completion(CompletionRequest(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=100000
    ))

def complete_the_table(model: str, temperature: float, research_paper_context: str, incomplete_table: str):
    prompt = complete_the_table_w_out_in_context_prompt.replace("{{Research Paper}}", research_paper_context)
    prompt = prompt.replace("{{Table}}", incomplete_table)
    
    return structured_completion(StructuredRequest(
        model=model,
        text=prompt,
        temperature=temperature,
        schema={
            "Compositions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "placeholder": {"type": "string", "description": "Placeholder in the table (e.g., <blank_1>)"},
                        "composition": {"type": "string", "description": "Extracted composition corresponding to the placeholder"}
                    },
                    "required": ["placeholder", "composition"]
                }
            }
        }
    ))


def complete_the_table_no_schema(model: str, temperature: float, research_paper_context: str, incomplete_table: str):
    prompt = complete_the_table_w_out_in_context_prompt.replace("{{Research Paper}}", research_paper_context)
    prompt = prompt.replace("{{Table}}", incomplete_table)
    return text_completion(CompletionRequest(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=100000
    ))


def simple_prompt(model: str, temperature: float, prompt: str):
    return text_completion(CompletionRequest(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=7999
    ))


# print(simple_prompt("custom/gemini-flash", 0, "Hi, just checking my api, are you working?" ))
# print(complete_the_table_in_context_no_schema("custom/gemini-flash", 0.0, context, incomplete_table))
# print(complete_the_table_in_context_no_schema("custom/gemini-pro", 0.0, context, incomplete_table))
# print(complete_the_table_no_schema("groq/llama-3.3-70b-versatile", 0.0, context, incomplete_table))
# print(complete_the_table_in_context("custom/gemini-flash", 0.0, context, incomplete_table))
# print(complete_the_table_in_context("custom/gemini-pro", 0.0, context, incomplete_table))