"""Built-in list prices (USD per 1M tokens, standard tier).

Source: https://developers.openai.com/api/docs/pricing, checked 2026-10-04. Prices change;
override them per model in the experiment config with a ``price`` table.
"""

from __future__ import annotations

from workforce.models.base import Price

PRICES_AS_OF = "2026-10-04"

OPENAI_PRICES: dict[str, Price] = {
    "gpt-6-astra": Price(10.00, 50.00, 1.00),
    "gpt-6.1-sol": Price(2.00, 10.00, 0.10),
    "gpt-6-sol": Price(2.00, 10.00, 0.20),
    "gpt-6-luna": Price(0.10, 0.50, 0.01),
    "gpt-5.6-sol": Price(4.00, 20.00, 0.40),
    "gpt-5.6-terra": Price(2.00, 12.00, 0.20),
    "gpt-5.6-luna": Price(0.20, 1.20, 0.02),
    "gpt-5.5": Price(5.00, 30.00, 0.50),
    "gpt-5.4": Price(2.50, 15.00, 0.25),
    "gpt-5.4-mini": Price(0.75, 4.50, 0.075),
    "gpt-5.4-nano": Price(0.20, 1.25, 0.02),
    "gpt-5-mini": Price(0.25, 2.00, 0.025),
    "gpt-5-nano": Price(0.05, 0.40, 0.005),
    "gpt-4.1": Price(2.00, 8.00, 0.50),
    "gpt-4.1-mini": Price(0.40, 1.60, 0.10),
    "gpt-4.1-nano": Price(0.10, 0.40, 0.025),
    "gpt-4o-mini": Price(0.15, 0.60, 0.075),
}


def lookup_price(provider: str, model: str) -> Price | None:
    if provider == "mock":
        return Price(0.0, 0.0, 0.0)
    if provider == "openai":
        if model in OPENAI_PRICES:
            return OPENAI_PRICES[model]
        # Dated snapshots such as gpt-4.1-mini-2025-04-14 share their alias's price.
        for alias in sorted(OPENAI_PRICES, key=len, reverse=True):
            if model.startswith(alias + "-") and model[len(alias) + 1 :][:1].isdigit():
                return OPENAI_PRICES[alias]
    return None
