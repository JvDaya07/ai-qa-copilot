
import os

from dotenv import load_dotenv
from google import genai
from typing import Any
from google.genai import types

load_dotenv()


def generate_response(prompt: str) -> str:
    """Send a prompt to Gemini and return the generated text."""

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is missing. "
            "Add it to your local .env file."
        )

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt,
    )

    if not response.text:
        raise RuntimeError("Gemini returned an empty response.")

    return response.text



def generate_structured_response(
    prompt: str,
    response_schema: dict,
):
    """Generate JSON using a Gemini-compatible schema."""

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is missing. "
            "Add it to your local .env file."
        )

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=response_schema,
        ),
    )

    if response.parsed is not None:
        return response.parsed

    if response.text:
        import json
        return json.loads(response.text)

    raise RuntimeError(
        "Gemini returned neither parsed data nor JSON text."
    )


