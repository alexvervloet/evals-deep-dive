"""
evals/providers.py: the ONLY file that talks to a model provider.

Evals are provider-agnostic: a dataset, a scorer, a metric, and a report don't
care who served the model. So we hide the one provider-specific call, turning a
prompt into an answer (`generate`), behind a single function. Everything else in
`evals/` and `examples/` is pure evaluation logic.

Pick your stack with `PROVIDER` in `.env`:

  PROVIDER=openai  ->  OpenAI chat     (needs OPENAI_API_KEY)
  PROVIDER=claude  ->  Claude messages (needs ANTHROPIC_API_KEY)

Unlike the RAG repo, evals never need embeddings, so there's just one call to
abstract and (for the claude stack) just one key.

Note on `temperature`: it's exposed here because evals care about it more than
most code. temperature=0 makes a task as repeatable as possible (good when you're
measuring the *task*); the LLM judges default to 0 so grading is stable. Example
09 deliberately turns it up to study run-to-run variance.
"""

import os
from functools import lru_cache

_OPENAI_CHAT = "gpt-4o-mini"
_CLAUDE_CHAT = "claude-haiku-4-5"

_KEYS = {
    "openai": ["OPENAI_API_KEY"],
    "claude": ["ANTHROPIC_API_KEY"],
}


def provider_name() -> str:
    """The active stack: 'openai' (default) or 'claude'. Set via PROVIDER in .env."""
    return os.getenv("PROVIDER", "openai").strip().lower()


def required_keys() -> list[str]:
    return _KEYS.get(provider_name(), [])


def describe() -> str:
    """One-line summary of the active stack, handy for examples to print."""
    p = provider_name()
    if p == "openai":
        return f"openai  (chat={_OPENAI_CHAT})"
    if p == "claude":
        return f"claude  (chat={_CLAUDE_CHAT})"
    return f"unknown provider {p!r}"


def ensure_ready() -> None:
    """Fail fast with a friendly message if the stack isn't configured.

    Call this at the top of any script *after* `load_dotenv()`.
    """
    import sys

    p = provider_name()
    if p not in _KEYS:
        sys.exit(f"PROVIDER={p!r} is not recognized. Set PROVIDER=openai or claude in .env.")
    missing = [k for k in required_keys() if not os.getenv(k)]
    if missing:
        sys.exit(
            f"PROVIDER={p} needs {', '.join(missing)} in the environment. "
            f"Provide them via secrun (see SECRETS.md), or run `secrun python check_setup.py`."
        )


@lru_cache(maxsize=1)
def _openai_client():
    from openai import OpenAI

    return OpenAI()


@lru_cache(maxsize=1)
def _anthropic_client():
    import anthropic

    return anthropic.Anthropic()


def generate(system: str, user: str, temperature: float = 0.0, max_tokens: int = 512) -> str:
    """Turn a (system, user) prompt into a text answer, normalized to a string.

    `temperature` defaults to 0 for repeatability, which matters when the thing
    you're measuring should not move run-to-run just because of sampling noise.
    """
    p = provider_name()
    if p == "openai":
        resp = _openai_client().chat.completions.create(
            model=_OPENAI_CHAT,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""
    if p == "claude":
        resp = _anthropic_client().messages.create(
            model=_CLAUDE_CHAT,
            temperature=temperature,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if b.type == "text")
    raise ValueError(f"Unknown PROVIDER={p!r} (expected 'openai' or 'claude').")
