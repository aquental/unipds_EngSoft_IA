from dataclasses import dataclass

import anthropic
from google import genai
from openai import OpenAI

openai_client = OpenAI()
anthropic_client = anthropic.Anthropic()
google_client = genai.Client()

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

FALLBACKS = {
    "cheap": [
        ("openai", "gpt-5.6-luna"),
        ("anthropic", "claude-haiku-4-5"),
        ("google", "gemini-3.5-flash-lite"),
    ],
    "premium": [
        ("openai", "gpt-5.6-sol"),
        ("anthropic", "claude-opus-4-8"),
        ("google", "gemini-3.6-flash"),
    ],
}

HARD_KEYWORDS = {
    "analyze",
    "debug",
    "design",
    "prove",
    "refactor",
    "strategize",
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
        contents=prompt,
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


def choose_tier(prompt: str) -> str:
    words = set(prompt.lower().split())
    hard_task = bool(words & HARD_KEYWORDS)
    long_task = len(prompt) > 500

    return "premium" if hard_task or long_task else "cheap"


def passes_quality_check(
    result: Completion,
    required_terms: set[str],
) -> bool:
    text = result.text.lower()

    return all(term.lower() in text for term in required_terms)


def route(
    prompt: str,
    required_terms: set[str],
) -> Completion:
    tier = choose_tier(prompt)
    last_error = None

    for provider, model in FALLBACKS[tier]:
        try:
            result = PROVIDERS[provider](prompt, model)
        except Exception as error:
            last_error = error
            print(f"{provider}/{model} failed: {error}")
            continue

        if passes_quality_check(result, required_terms):
            return result

        print(f"{provider}/{model} failed quality check")

    raise RuntimeError(
        f"All candidates in tier '{tier}' failed the quality check "
        f"for required terms {required_terms}. Last error: {last_error}"
    )


requests = [
    {
        "prompt": "What is the capital of France? Include the word Paris.",
        "required_terms": {"paris"},
    },
    {
        "prompt": (
            "Analyze why a database index can speed up reads. "
            "Mention both lookup and writes."
        ),
        "required_terms": {"lookup", "writes"},
    },
]

for request in requests:
    result = route(
        request["prompt"],
        request["required_terms"],
    )

    print(
        f"\nAnswered by {result.provider}/{result.model} "
        f"for ${result.cost:.6f}"
    )
    print(result.text)