from openai import OpenAI

CHEAP_MODEL = "gpt-5.6-luna"
PREMIUM_MODEL = "gpt-5.6-sol"

HARD_KEYWORDS = {
    "analyze",
    "debug",
    "design",
    "prove",
    "refactor",
    "strategize",
}

client = OpenAI()

PRICING = {
    CHEAP_MODEL: {
        "input": 1.00 / 1_000_000,
        "cache_write": 1.25 / 1_000_000,
        "cached_input": 0.10 / 1_000_000,
        "output": 6.00 / 1_000_000,
    },
    PREMIUM_MODEL: {
        "input": 5.00 / 1_000_000,
        "cache_write": 6.25 / 1_000_000,
        "cached_input": 0.50 / 1_000_000,
        "output": 30.00 / 1_000_000,
    },
}


def cost_for(model: str, usage) -> float:
    rates = PRICING[model]
    details = usage.input_tokens_details

    cached = details.cached_tokens
    cache_write = details.cache_write_tokens
    fresh = usage.input_tokens - cached - cache_write

    return (
        fresh * rates["input"]
        + cache_write * rates["cache_write"]
        + cached * rates["cached_input"]
        + usage.output_tokens * rates["output"]
    )


def is_hard_task(prompt: str) -> bool:
    words = set(prompt.lower().split())
    return bool(words & HARD_KEYWORDS)


def choose_model(prompt: str) -> str:
    hard_task = is_hard_task(prompt)
    long_task = len(prompt) > 500

    return PREMIUM_MODEL if hard_task or long_task else CHEAP_MODEL


def generate(prompt: str, model: str) -> float:
    response = client.responses.create(
        model=model,
        input=prompt,
    )
    return cost_for(model, response.usage)


workload = [
    "Classify this review as positive or negative: I loved it.",
    "Summarize this sentence in five words: Caching avoids repeated work.",
    "Debug a recursive function that overflows the stack.",
    "Design a migration strategy for a database with zero downtime.",
]

routed_total = 0.0
premium_total = 0.0

for prompt in workload:
    routed_model = choose_model(prompt)
    routed_cost = generate(prompt, routed_model)
    premium_cost = generate(prompt, PREMIUM_MODEL)

    routed_total += routed_cost
    premium_total += premium_cost

    print(f"{routed_model:14}  ${routed_cost:.6f}  {prompt}")

print(f"\nAll-premium baseline: ${premium_total:.6f}")
print(f"Routed workload:      ${routed_total:.6f}")
print(f"Savings:              ${premium_total - routed_total:.6f}")