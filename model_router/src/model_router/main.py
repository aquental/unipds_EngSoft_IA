from openai import OpenAI

CHEAP_MODEL = "gpt-5.6-luna"
MIDDLE_MODEL = "gpt-5.6-terra"
PREMIUM_MODEL = "gpt-5.6-sol"

client = OpenAI()

PRICING = {
    CHEAP_MODEL: {
        "input": 1.00 / 1_000_000,
        "cache_write": 1.25 / 1_000_000,
        "cached_input": 0.10 / 1_000_000,
        "output": 6.00 / 1_000_000,
    },
    MIDDLE_MODEL: {
        "input": 2.50 / 1_000_000,
        "cache_write": 3.125 / 1_000_000,
        "cached_input": 0.25 / 1_000_000,
        "output": 15.00 / 1_000_000,
    },
    PREMIUM_MODEL: {
        "input": 5.00 / 1_000_000,
        "cache_write": 6.25 / 1_000_000,
        "cached_input": 0.50 / 1_000_000,
        "output": 30.00 / 1_000_000,
    }
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


prompt = (
    "Explain why database indexes improve read performance "
    "and mention one tradeoff."
)

for model in (CHEAP_MODEL,MIDDLE_MODEL,PREMIUM_MODEL):
    response = client.responses.create(
        model=model,
        input=prompt,
    )

    print(f"\n{model}")
    print(f"Cost: ${cost_for(model, response.usage):.6f}")
    print(response.output_text)