"""Provider registry.

Every provider here speaks the OpenAI-compatible chat-completions API, so
supporting a new one is data only: where it lives, how to recognise its API
keys, how it bills, and which of its models are allowed. ``keys.py`` uses the
key patterns to work out which provider a pasted key belongs to.

Base URLs can be overridden with ``<ID>_BASE_URL`` (e.g. ``GROQ_BASE_URL``),
which the test suite uses to point everything at a local stub.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from . import config
from .errors import DetectorError

# How the provider charges for the models we allow.
FREE = "free"            # every allowed model is free (OpenRouter :free, Groq free plan)
FREE_TIER = "free-tier"  # the key has a free allowance; heavy use may need billing
PAID = "paid"            # usage is billed to the key owner

# Which models a provider may use.
POLICY_FREE_SUFFIX = "free-suffix"  # model id must end in ":free"
POLICY_ALLOWLIST = "allowlist"      # only ids in ``allowlist`` / ``specialized``
POLICY_LISTED = "listed"            # anything the provider's /models lists


@dataclass(eq=False)
class ProviderSpec:
    id: str
    label: str
    default_base_url: str
    key_patterns: tuple[str, ...]
    env_vars: tuple[str, ...]
    keys_url: str = ""
    billing: str = PAID
    model_policy: str = POLICY_LISTED
    # Curated models in rank order (id -> short note). For allowlist providers
    # this *is* the list; for others it only boosts these ids in the ranking.
    allowlist: dict = field(default_factory=dict)
    specialized: dict = field(default_factory=dict)
    default_model: str = ""
    reasoning_models: str = ""   # regex: models that spend hidden thinking tokens
    low_effort_models: str = ""  # regex: models that accept reasoning_effort="low"
    extra_headers: dict = field(default_factory=dict)
    concurrency: int = 2         # parallel requests for one scan
    balance_width: int = 2       # models to spread one scan over

    @property
    def key(self) -> str:  # kept for call sites that read cfg.key
        return self.id

    @property
    def base_url(self) -> str:
        env = os.environ.get(f"{self.id.upper()}_BASE_URL", "")
        return (env or self.default_base_url).rstrip("/")

    def recognises(self, api_key: str) -> bool:
        return any(re.search(p, api_key) for p in self.key_patterns)


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
# Audio, TTS, embedding, reranker and prompt-guard models are excluded.
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

DEFAULT_PROVIDER = "openrouter"
DEFAULT_MODEL = "google/gemma-4-31b-it:free"

_SPECS = [
    ProviderSpec(
        id="openrouter",
        label="OpenRouter",
        default_base_url="https://openrouter.ai/api/v1",
        key_patterns=(r"^sk-or-",),
        env_vars=("OPENROUTER_API_KEY", "OPEN_ROUTER_KEY", "OPENROUTER_KEY", "OPEN_ROUTER_API_KEY"),
        keys_url="https://openrouter.ai/keys",
        billing=FREE,
        model_policy=POLICY_FREE_SUFFIX,
        allowlist=OPENROUTER_RECOMMENDED,
        specialized=OPENROUTER_SPECIALIZED,
        default_model=DEFAULT_MODEL,
        extra_headers={"HTTP-Referer": config.APP_URL, "X-Title": config.APP_TITLE},
        # Free OpenRouter limits are mostly per account, so modest
        # parallelism and narrow balancing.
        concurrency=3,
        balance_width=2,
    ),
    ProviderSpec(
        id="groq",
        label="Groq",
        default_base_url="https://api.groq.com/openai/v1",
        key_patterns=(r"^gsk_",),
        env_vars=("GROQ_API_KEY", "GROQ_KEY"),
        keys_url="https://console.groq.com/keys",
        billing=FREE,
        model_policy=POLICY_ALLOWLIST,
        allowlist=GROQ_FREE_CHAT,
        specialized=GROQ_FREE_SPECIALIZED,
        default_model="openai/gpt-oss-120b",
        reasoning_models=r"^(openai/gpt-oss|qwen/qwen3)",
        low_effort_models=r"^openai/gpt-oss",
        # Token-per-minute bound (~8K TPM): little parallelism, but limits
        # are per model, so balancing over 3 models ≈ 3x the budget.
        concurrency=2,
        balance_width=3,
    ),
    ProviderSpec(
        id="gemini",
        label="Google Gemini",
        default_base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        key_patterns=(r"^AIza[0-9A-Za-z_\-]{30,}$",),
        env_vars=("GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_KEY"),
        keys_url="https://aistudio.google.com/apikey",
        billing=FREE_TIER,
        reasoning_models=r"gemini-(2\.5|[3-9])",
        low_effort_models=r"gemini-(2\.5|[3-9])",
    ),
    ProviderSpec(
        id="cerebras",
        label="Cerebras",
        default_base_url="https://api.cerebras.ai/v1",
        key_patterns=(r"^csk-",),
        env_vars=("CEREBRAS_API_KEY",),
        keys_url="https://cloud.cerebras.ai",
        billing=FREE_TIER,
        reasoning_models=r"gpt-oss|thinking",
        low_effort_models=r"gpt-oss",
    ),
    ProviderSpec(
        id="mistral",
        label="Mistral",
        default_base_url="https://api.mistral.ai/v1",
        key_patterns=(r"^[A-Za-z0-9]{32}$",),
        env_vars=("MISTRAL_API_KEY",),
        keys_url="https://console.mistral.ai/api-keys",
        billing=FREE_TIER,
        reasoning_models=r"magistral",
    ),
    ProviderSpec(
        id="openai",
        label="OpenAI",
        default_base_url="https://api.openai.com/v1",
        # Generic "sk-" keys, minus prefixes that belong to other providers.
        key_patterns=(r"^sk-(proj|svcacct|admin)-", r"^sk-(?!or-|ant-)[A-Za-z0-9_\-]{40,}$"),
        env_vars=("OPENAI_API_KEY",),
        keys_url="https://platform.openai.com/api-keys",
        reasoning_models=r"^(o\d|gpt-5)",
        low_effort_models=r"^(o\d|gpt-5)",
    ),
    ProviderSpec(
        id="deepseek",
        label="DeepSeek",
        default_base_url="https://api.deepseek.com",
        key_patterns=(r"^sk-[0-9a-f]{32}$",),
        env_vars=("DEEPSEEK_API_KEY",),
        keys_url="https://platform.deepseek.com/api_keys",
        reasoning_models=r"reasoner",
    ),
    ProviderSpec(
        id="xai",
        label="xAI",
        default_base_url="https://api.x.ai/v1",
        key_patterns=(r"^xai-",),
        env_vars=("XAI_API_KEY",),
        keys_url="https://console.x.ai",
        reasoning_models=r"grok-(3-mini|4)",
        low_effort_models=r"grok-3-mini",
    ),
    ProviderSpec(
        id="together",
        label="Together AI",
        default_base_url="https://api.together.xyz/v1",
        key_patterns=(r"^tgp_v1_", r"^[0-9a-f]{64}$"),
        env_vars=("TOGETHER_API_KEY",),
        keys_url="https://api.together.ai/settings/api-keys",
        reasoning_models=r"(?i)deepseek-r1|qwq|thinking",
    ),
]

PROVIDERS: dict[str, ProviderSpec] = {spec.id: spec for spec in _SPECS}


def get_provider(provider: str) -> ProviderSpec:
    key = (provider or DEFAULT_PROVIDER).strip().lower()
    if key not in PROVIDERS:
        raise DetectorError("Unknown model provider.", status=400)
    return PROVIDERS[key]


def validate_model(provider: str, model: str) -> None:
    spec = get_provider(provider)
    model = (model or "").strip()
    if not model:
        raise DetectorError("Select a model in Settings.", status=400)
    if spec.model_policy == POLICY_FREE_SUFFIX and not model.endswith(":free"):
        raise DetectorError(
            f"Only free {spec.label} models (ending in :free) are allowed.", status=400
        )
    if spec.model_policy == POLICY_ALLOWLIST and model not in (
        set(spec.allowlist) | set(spec.specialized)
    ):
        allowed = ", ".join(sorted(spec.allowlist))
        raise DetectorError(
            f"Only {spec.label} free-plan models are allowed ({allowed}).", status=400
        )


def validate_key(provider: str, api_key: str) -> None:
    if not api_key or len(api_key.strip()) < config.MIN_KEY_LENGTH:
        spec = get_provider(provider)
        raise DetectorError(
            f"No API key for {spec.label} yet. Add one in Settings.", status=400
        )
