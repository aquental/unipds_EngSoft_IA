from model_router.completion_gate import passes_quality_check
from model_router.providers import PROVIDERS, Completion

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

DEMO_REQUESTS = [
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


def choose_tier(prompt: str) -> str:
    words = set(prompt.lower().split())
    hard_task = bool(words & HARD_KEYWORDS)
    long_task = len(prompt) > 500

    return "premium" if hard_task or long_task else "cheap"


def route(
    prompt: str,
    required_terms: set[str],
) -> Completion:
    tier = choose_tier(prompt)

    for provider, model in FALLBACKS[tier]:
        result = PROVIDERS[provider](prompt, model)

        if passes_quality_check(result, required_terms):
            return result

        print(f"{provider}/{model} failed quality check")

    raise RuntimeError(
        f"All candidates in tier '{tier}' failed the quality check "
        f"for required terms {required_terms}"
    )


def run_demo(route_fn) -> None:
    for request in DEMO_REQUESTS:
        result = route_fn(
            request["prompt"],
            request["required_terms"],
        )

        print(
            f"\nAnswered by {result.provider}/{result.model} "
            f"for ${result.cost:.6f}"
        )
        print(result.text)


if __name__ == "__main__":
    run_demo(route)
