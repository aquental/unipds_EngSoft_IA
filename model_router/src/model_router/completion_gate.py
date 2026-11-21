from dataclasses import dataclass


@dataclass
class Completion:
    text: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost: float


def passes_quality_check(
    result: Completion,
    required_terms: set[str],
) -> bool:
    text = result.text.lower()
    return all(term.lower() in text for term in required_terms)


examples = [
    (
        "case-insensitive required term",
        Completion(
            text="The capital of France is Paris.",
            provider="openai",
            model="gpt-5.6-luna",
            input_tokens=20,
            output_tokens=8,
            cost=0.000068,
        ),
        {"PARIS"},
    ),
    (
        "one required term is missing",
        Completion(
            text="An index creates a faster lookup path for reads.",
            provider="openai",
            model="gpt-5.6-sol",
            input_tokens=30,
            output_tokens=12,
            cost=0.000510,
        ),
        {"lookup", "writes"},
    ),
]

for label, result, required_terms in examples:
    accepted = passes_quality_check(result, required_terms)
    print(f"{str(accepted):5}  {label}")