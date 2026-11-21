from dataclasses import dataclass
from functools import cache

import anthropic
from google import genai
from openai import OpenAI

# Anthropic requires an explicit output limit. 256 was too low: longer
# answers were cut off before reaching the required terms and then
# failed the quality check.
ANTHROPIC_MAX_TOKENS = 1024

PRICING = {
    "gpt-5.6-luna": {
        "input": 1.00 / 1_000_000,
        "output": 6.00 / 1_000_000,
    },
    "gpt-5.6-sol": {
        "input": 5.00 / 1_000_000,
        "output": 30.00 / 1_000_000,
    },
    "claude-haiku-4-5": {
        "input": 1.00 / 1_000_000,
        "output": 5.00 / 1_000_000,
    },
    "claude-opus-4-8": {
        "input": 5.00 / 1_000_000,
        "output": 25.00 / 1_000_000,
    },
    "gemini-3.5-flash-lite": {
        "input": 0.30 / 1_000_000,
        "output": 2.50 / 1_000_000,
    },
    "gemini-3.6-flash": {
        "input": 1.50 / 1_000_000,
        "output": 7.50 / 1_000_000,
    },
}


@dataclass
class Completion:
    text: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost: float


# Clients are created on first use, not at import time, so importing this
# module needs neither API keys nor network access.
@cache
def openai_client() -> OpenAI:
    return OpenAI()


@cache
def anthropic_client() -> anthropic.Anthropic:
    return anthropic.Anthropic()


@cache
def google_client() -> genai.Client:
    return genai.Client()


def call_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    rates = PRICING[model]

    return (
        input_tokens * rates["input"]
        + output_tokens * rates["output"]
    )


def generate_openai(prompt: str, model: str) -> Completion:
    response = openai_client().responses.create(
        model=model,
        input=prompt,
    )

    return Completion(
        text=response.output_text,
        provider="openai",
        model=model,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        cost=call_cost(
            model,
            response.usage.input_tokens,
            response.usage.output_tokens,
        ),
    )


def generate_anthropic(prompt: str, model: str) -> Completion:
    response = anthropic_client().messages.create(
        model=model,
        max_tokens=ANTHROPIC_MAX_TOKENS,
        messages=[
            {"role": "user", "content": prompt},
        ],
    )

    return Completion(
        text=response.content[0].text,
        provider="anthropic",
        model=model,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        cost=call_cost(
            model,
            response.usage.input_tokens,
            response.usage.output_tokens,
        ),
    )


def generate_google(prompt: str, model: str) -> Completion:
    response = google_client().models.generate_content(
        model=model,
        contents=prompt,        # "contents", not "input" or "messages"
    )

    usage = response.usage_metadata
    thinking_tokens = usage.thoughts_token_count or 0
    output_tokens = usage.candidates_token_count + thinking_tokens

    return Completion(
        text=response.text,
        provider="google",
        model=model,
        input_tokens=usage.prompt_token_count,
        output_tokens=output_tokens,
        cost=call_cost(
            model,
            usage.prompt_token_count,
            output_tokens,
        ),
    )


PROVIDERS = {
    "openai": generate_openai,
    "anthropic": generate_anthropic,
    "google": generate_google,
}
