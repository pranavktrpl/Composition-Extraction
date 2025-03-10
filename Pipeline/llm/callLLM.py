import json
import os
from time import sleep
from typing import Any, Dict, List, Optional, Union

import litellm
import requests
from litellm import completion
from pydantic import BaseModel, Field, create_model
from dotenv import load_dotenv

load_dotenv()

# Drop unsupported parameters for different model providers
litellm.drop_params = True

class Message(BaseModel):
    role: str
    content: str


class CompletionRequest(BaseModel):
    """Request model for LLM completion."""

    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.0
    max_tokens: Optional[int] = 100000


class StructuredRequest(BaseModel):
    """Request model for structured LLM completion."""

    model: str
    text: str
    temperature: Optional[float] = 0.0
    max_tokens: Optional[int] = 100000
    schema: Dict[str, Any]  # JSON Schema for output validation


def gemini_pro_completion(text: str) -> str:
    """Make a direct API call to Gemini Pro."""
    api_key = os.getenv("GEMINI_API_KEY") 
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-lite:generateContent?key={api_key}"
    payload = {"contents": [{"parts": [{"text": text}]}]}

    max_retries = 3
    base_delay = 2

    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            data = response.json()

            # Extract text from Gemini response
            try:
                # return data
                return data["candidates"][0]["content"]["parts"][0]["text"], data.get("usageMetadata", {"promptTokenCount": 0, "candidatesTokenCount": 0, "totalTokenCount": 0})
            except (KeyError, IndexError) as e:
                raise RuntimeError(f"Unexpected Gemini response format: {str(e)}")

        except requests.RequestException as e:
            if "429" in str(e).lower():
                if attempt == max_retries - 1:
                    raise RuntimeError("Too many requests")
                delay = base_delay * (2**attempt)
                sleep(delay)
                continue

            raise RuntimeError(f"Gemini API error: {str(e)}")


def gemini_2point0_flash(text: str) -> str:
    """Make a direct API call to Gemini 2.0 Flash."""
    api_key = os.getenv("GEMINI_API_KEY") 
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    payload = {"contents": [{"parts": [{"text": text}]}]}

    max_retries = 3
    base_delay = 2

    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            data = response.json()

            # Extract text from Gemini response
            try:
                # return data
                return data["candidates"][0]["content"]["parts"][0]["text"], data.get("usageMetadata", {"promptTokenCount": 0, "candidatesTokenCount": 0, "totalTokenCount": 0})
            except (KeyError, IndexError) as e:
                raise RuntimeError(f"Unexpected Gemini response format: {str(e)}")

        except requests.RequestException as e:
            if "429" in str(e).lower():
                if attempt == max_retries - 1:
                    raise RuntimeError("Too many requests")
                delay = base_delay * (2**attempt)
                sleep(delay)
                continue

            raise RuntimeError(f"Gemini API error: {str(e)}")



def gemini_flash_completion(text: str) -> str:
    """Make a direct API call to Gemini Flash."""
    api_key = os.getenv("GEMINI_API_KEY") 
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}"
    payload = {"contents": [{"parts": [{"text": text}]}]}

    max_retries = 3
    base_delay = 2

    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            data = response.json()

            # Extract text from Gemini response
            try:
                # return data
                return data["candidates"][0]["content"]["parts"][0]["text"], data.get("usageMetadata")
            except (KeyError, IndexError) as e:
                raise f"Error_500_INTERNAL_SERVER_ERROR; detail=Unexpected Gemini response format: {str(e)}"

        except requests.RequestException as e:
            if "429" in str(e).lower():
                if attempt == max_retries - 1:
                    raise "Error_429_TOO_MANY_REQUESTS"
                delay = base_delay * (2**attempt)
                sleep(delay)
                continue

            raise f"Error_500_INTERNAL_SERVER_ERROR; detail=Gemini API error: {str(e)}"
        
def text_completion(in_request: CompletionRequest) -> str:
    """
    Make a basic LLM completion call.
    """
    max_retries = 3
    base_delay = 1

    for attempt in range(max_retries):
        try:
            # Handle Gemini models specially
            if in_request.model == "custom/gemini-2point0":
                text = "\n".join(
                    f"{msg.role}: {msg.content}" for msg in in_request.messages
                )
                response, metadata = gemini_2point0_flash(text)
                return response, metadata
            elif in_request.model == "custom/gemini-flash":
                text = "\n".join(
                    f"{msg.role}: {msg.content}" for msg in in_request.messages
                )
                response, metadata = gemini_flash_completion(text)
                return response, metadata
            elif in_request.model == "custom/gemini-pro":
                text = "\n".join(
                    f"{msg.role}: {msg.content}" for msg in in_request.messages
                )
                response, metadata = gemini_pro_completion(text)
                return response, metadata
            else:
                # Use LiteLLM for other models
                response = completion(
                    model=in_request.model,
                    messages=[msg.dict() for msg in in_request.messages],
                    temperature=in_request.temperature,
                    max_tokens=in_request.max_tokens,
                )
                return response.choices[0].message.content, {
                    "promptTokenCount": response.usage.prompt_tokens,
                    "candidatesTokenCount": response.usage.completion_tokens,
                    "totalTokenCount": response.usage.total_tokens
                }

        except Exception as e:
            if attempt < max_retries - 1:
                sleep_time = base_delay * (2 ** attempt)
                print(f"Error: {str(e)}")
                print(f"Retrying in {sleep_time} seconds...")
                sleep(sleep_time)
                continue
            raise RuntimeError(f"Failed after {max_retries} attempts: {str(e)}")

    raise RuntimeError(f"Failed after {max_retries} attempts")


def structured_completion(in_request: StructuredRequest):
    """
    Make an LLM completion call that returns structured data according to a schema.
    Validates both input and output against provided schemas.
    """
    system_msg = {
        "role": "system",
        "content": f"""You are a structured data extraction system.
        Your response must be valid JSON that follows this schema:
        {json.dumps(in_request.schema, indent=2)}

        Do not include any explanations or text outside of the JSON structure.
        If you cannot extract certain fields, use null for those fields.""",
    }

    messages = [system_msg, {"role": "user", "content": in_request.text}]

    max_retries = 3
    base_delay = 1

    for attempt in range(max_retries):
        try:
            # Handle Gemini Flash specially
            if in_request.model == "custom/gemini-2point0":
                # Combine messages into a single text
                text = f"{system_msg['content']}\n\nUser request: {in_request.text}\n\nRespond with valid JSON only:"
                response_text, usageMetadata = gemini_2point0_flash(text)
                response_text = response_text.strip('```json\n')
                response_text = response_text.strip('```')
            elif in_request.model == "custom/gemini-flash":
                # Combine messages into a single text with clear JSON requirement
                text = f"{system_msg['content']}\n\nUser request: {in_request.text}\n\nRespond with valid JSON only:"
                response_text, usageMetadata = gemini_flash_completion(text)
                response_text = response_text.strip('```json\n')
                response_text = response_text.strip('```')
            elif in_request.model == "custom/gemini-pro":
                # Combine messages into a single text with clear JSON requirement
                text = f"{system_msg['content']}\n\nUser request: {in_request.text}\n\nRespond with valid JSON only:"
                response_text = gemini_pro_completion(text)
            else:
                # Use LiteLLM for other models
                response = completion(
                    model=in_request.model,
                    messages=messages,
                    temperature=in_request.temperature,
                    max_tokens=in_request.max_tokens,
                    response_format={"type": "json_object"},
                )
                response_text = response.choices[0].message.content

            try:
                result = json.loads(response_text)
                # metadata = json.loads(usageMetadata)
                return result, usageMetadata
            except json.JSONDecodeError:
                raise "ERROR_422_UNPROCESSABLE_ENTITY, detail=LLM response was not valid JSON"

        except Exception as e:
            if "rate limit" in str(e).lower():
                if attempt == max_retries - 1:
                    raise f"ERROR_429_TOO_MANY_REQUESTS, detail={str(e)}"
                delay = base_delay * (2**attempt)
                sleep(delay)
                continue

            raise f"ERROR_500_INTERNAL_SERVER_ERROR, detail={str(e)}"