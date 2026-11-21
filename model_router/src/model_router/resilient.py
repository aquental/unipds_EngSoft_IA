from model_router.completion_gate import passes_quality_check
from model_router.llm_completion_router import FALLBACKS, choose_tier, run_demo
from model_router.providers import PROVIDERS, Completion


def route(
    prompt: str,
    required_terms: set[str],
) -> Completion:
    """Like llm_completion_router.route, but an API error from one
    provider moves on to the next candidate instead of aborting."""
    tier = choose_tier(prompt)
    failures = []
    last_error = None

    for provider, model in FALLBACKS[tier]:
        try:
            result = PROVIDERS[provider](prompt, model)
        except Exception as error:
            last_error = error
            failures.append(f"{provider}/{model}: API error: {error}")
            print(f"{provider}/{model} failed: {error}")
            continue

        if passes_quality_check(result, required_terms):
            return result

        failures.append(
            f"{provider}/{model}: missing required terms {required_terms}"
        )
        print(f"{provider}/{model} failed quality check")

    raise RuntimeError(
        f"All candidates in tier '{tier}' failed:\n  "
        + "\n  ".join(failures)
    ) from last_error


if __name__ == "__main__":
    run_demo(route)
