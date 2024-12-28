
import json
import os
from time import sleep
from typing import Any, Dict, List, Optional, Union

import litellm
import requests
from litellm import completion
from pydantic import BaseModel, Field, create_model
from os import load_dotenv

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
    max_tokens: Optional[int] = 10000


class StructuredRequest(BaseModel):
    """Request model for structured LLM completion."""

    model: str
    text: str
    temperature: Optional[float] = 0.0
    max_tokens: Optional[int] = 10000
    schema: Dict[str, Any]  # JSON Schema for output validation


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
                return data["candidates"][0]["content"]["parts"][0]["text"]
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
            # Handle Gemini Flash specially
            if in_request.model == "custom/gemini-flash":
                # Combine messages into a single text
                text = "\n".join(
                    f"{msg.role}: {msg.content}" for msg in in_request.messages
                )
                return gemini_flash_completion(text)

            # Use LiteLLM for other models
            response = completion(
                model=in_request.model,
                messages=[msg.dict() for msg in in_request.messages],
                temperature=in_request.temperature,
                max_tokens=in_request.max_tokens,
            )
            return response.choices[0].message.content

        except Exception as e:
            if "rate limit" in str(e).lower():
                if attempt == max_retries - 1:
                    raise f"ERROR_429_TOO_MANY_REQUESTS, detail={str(e)}"
                delay = base_delay * (2**attempt)
                sleep(delay)
                continue

            raise f"HTTP_500_INTERNAL_SERVER_ERROR, detail={str(e)}"


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
            if in_request.model == "custom/gemini-flash":
                # Combine messages into a single text with clear JSON requirement
                text = f"{system_msg['content']}\n\nUser request: {in_request.text}\n\nRespond with valid JSON only:"
                response_text = gemini_flash_completion(text)
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
                return result
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