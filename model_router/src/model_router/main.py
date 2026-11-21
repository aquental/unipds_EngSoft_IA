from model_router.providers import PROVIDERS

OPENAI_MODEL = "gpt-5.6-luna"
ANTHROPIC_MODEL = "claude-haiku-4-5"
GOOGLE_MODEL = "gemini-3.5-flash-lite"

models = [
    ("openai", OPENAI_MODEL),
    ("anthropic", ANTHROPIC_MODEL),
    ("google", GOOGLE_MODEL),
]


if __name__ == "__main__":
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
