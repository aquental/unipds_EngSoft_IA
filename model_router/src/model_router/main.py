from dataclasses import dataclass

import anthropic
from google import genai
from openai import OpenAI

OPENAI_MODEL = "gpt-5.6-luna"
ANTHROPIC_MODEL = "claude-haiku-4-5"
GOOGLE_MODEL = "gemini-3.5-flash-lite"

openai_client = OpenAI()
anthropic_client = anthropic.Anthropic()
google_client = genai.Client()

PRICING = {
    OPENAI_MODEL: {
        "input": 1.00 / 1_000_000,
        "output": 6.00 / 1_000_000,
    },
    ANTHROPIC_MODEL: {
        "input": 1.00 / 1_000_000,
        "output": 5.00 / 1_000_000,
    },
    GOOGLE_MODEL: {
        "input": 0.30 / 1_000_000,
        "output": 2.50 / 1_000_000,
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
    response = openai_client.responses.create(
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
    response = anthropic_client.messages.create(
        model=model,
        max_tokens=256,
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
    response = google_client.models.generate_content(
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

models = [
    ("openai", OPENAI_MODEL),
    ("anthropic", ANTHROPIC_MODEL),
    ("google", GOOGLE_MODEL),
]

prompt = "Give one practical tip for writing maintainable Python."

for provider, model in models:
    result = PROVIDERS[provider](prompt, model)

    preview = result.text.replace("\n", " ").strip()[:80]

    print(
        f"{result.provider:10} "
        f"{result.model:24} "
        f"in={result.input_tokens:4} "
        f"out={result.output_tokens:4} "
        f"${result.cost:.6f}"
    )
    print(f"Preview: {preview}")
    print()