"""Provider registry and free-tier model rules.

Only free-tier models are accepted: OpenRouter's ``:free`` catalogue and Groq's
documented Free Plan models. Checks run before any network call.
"""

from __future__ import annotations

from . import config
from .errors import DetectorError

DEFAULT_PROVIDER = "openrouter"
DEFAULT_MODEL = "google/gemma-4-31b-it:free"

PROVIDERS = {
    "openrouter": {
        "label": "OpenRouter",
        "base_url": config.OPENROUTER_BASE_URL,
        "keys_url": "https://openrouter.ai/keys",
        "key_placeholder": "sk-or-v1-…",
        "default_model": DEFAULT_MODEL,
        # Free OpenRouter models share a per-key request budget; modest
        # parallelism keeps long documents fast without tripping 429s.
        "concurrency": 3,
        # Free-tier limits are mostly per account, so spreading sections
        # across models helps less here than on Groq.
        "balance_width": 2,
    },
    "groq": {
        "label": "Groq",
        "base_url": config.GROQ_BASE_URL,
        "keys_url": "https://console.groq.com/keys",
        "key_placeholder": "gsk_…",
        "default_model": "openai/gpt-oss-120b",
        # The free plan is token-per-minute bound (~8K TPM); parallel calls
        # mostly convert into 429s.
        "concurrency": 2,
        # Limits are per model: three free chat models ≈ three times the
        # tokens-per-minute budget.
        "balance_width": 3,
    },
}

# Curated task-fit order. The first recommended entry becomes the default.
# Only models that actually appear in the provider's free catalogue are shown.
OPENROUTER_RECOMMENDED = {
    "google/gemma-4-31b-it:free": "instruction-tuned · structured JSON · 31B dense",
    "google/gemma-4-26b-a4b-it:free": "instruction-tuned MoE · structured JSON · fast",
    "nvidia/nemotron-3-super-120b-a12b:free": "structured outputs · 120B MoE",
    "nvidia/nemotron-3.5-super-vl:free": "structured outputs · 1M context",
    "dots-studio/dots-3-note-preview:free": "structured outputs · 512K context",
}
OPENROUTER_SPECIALIZED = {
    "nvidia/nemotron-3.5-content-safety:free": (
        "guardrail classifier · may emit safety labels instead of the schema"
    ),
}

# Groq Free Plan text models (from console.groq.com/docs/rate-limits).
# Audio, TTS, embedding, reranker and prompt-guard models are excluded: they
# cannot perform sentence classification in the required JSON shape.
GROQ_FREE_CHAT = {
    "openai/gpt-oss-120b": "120B · strongest JSON adherence on the free plan",
    "qwen/qwen3.8-27b": "27B · strong multilingual instruction following",
    "openai/gpt-oss-20b": "20B · fastest free option",
}
GROQ_FREE_SPECIALIZED = {
    "openai/gpt-oss-safeguard-20b": (
        "safety classifier · may emit safety labels instead of the schema"
    ),
}
GROQ_FREE_MODELS = set(GROQ_FREE_CHAT) | set(GROQ_FREE_SPECIALIZED)


def get_provider(provider: str) -> dict:
    key = (provider or DEFAULT_PROVIDER).strip().lower()
    if key not in PROVIDERS:
        raise DetectorError("Unknown model provider.", status=400)
    return {**PROVIDERS[key], "key": key}


def validate_model(provider: str, model: str) -> None:
    cfg = get_provider(provider)
    model = (model or "").strip()
    if not model:
        raise DetectorError("Select a free model from the sidebar.", status=400)
    if cfg["key"] == "openrouter":
        if not model.endswith(":free"):
            raise DetectorError(
                "Only OpenRouter :free models are allowed. Pick one from the sidebar.",
                status=400,
            )
        return
    if model not in GROQ_FREE_MODELS:
        allowed = ", ".join(sorted(GROQ_FREE_CHAT))
        raise DetectorError(
            f"Only Groq Free Plan models are allowed ({allowed}).",
            status=400,
        )


def validate_key(provider: str, api_key: str) -> None:
    if not api_key or len(api_key.strip()) < config.MIN_KEY_LENGTH:
        cfg = get_provider(provider)
        raise DetectorError(
            f"No valid {cfg['label']} API key. Add one in Settings or set "
            f"{config.ENV_KEY_NAMES[cfg['key']]} in .env.",
            status=400,
        )
