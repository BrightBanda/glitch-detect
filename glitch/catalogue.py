"""Free-model catalogue: listing, ranking, and fallback order."""

from __future__ import annotations

import threading
import time

import requests

from . import client
from .errors import DetectorError
from .providers import (
    DEFAULT_MODEL,
    GROQ_FREE_CHAT,
    GROQ_FREE_SPECIALIZED,
    OPENROUTER_RECOMMENDED,
    OPENROUTER_SPECIALIZED,
    PROVIDERS,
    get_provider,
)

TASK_HINTS = ("safety", "guard", "moderation", "classif", "detect", "forensic", "judge")
GROUP_ORDER = ("recommended", "specialized", "general")
GROUP_LABELS = {
    "recommended": "RECOMMENDED — BEST FOR SENTENCE JSON",
    "specialized": "SPECIALIZED CLASSIFIERS — EXPERIMENTAL",
    "general": "OTHER FREE MODELS",
}

_CACHE_TTL = 600
_cache_lock = threading.Lock()
_MODEL_CACHE: dict = {
    "openrouter": {"at": 0.0, "payload": None},
    "groq": {"at": 0.0, "payload": None},
}


def _entry(
    model_id: str,
    name: str,
    context,
    group: str,
    note: str,
    *,
    json_capable: bool = False,
    reasoning: bool = False,
) -> dict:
    return {
        "id": model_id,
        "name": name or model_id,
        "context": context if isinstance(context, int) else 0,
        "group": group,
        "group_label": GROUP_LABELS[group],
        "note": note,
        "json_capable": bool(json_capable),
        "reasoning": bool(reasoning),
        "free": True,
    }


def _sort_models(models: list[dict], priority: list[str]) -> list[dict]:
    def key(entry: dict):
        group = GROUP_ORDER.index(entry["group"])
        if entry["group"] in ("recommended", "specialized"):
            order = priority.index(entry["id"]) if entry["id"] in priority else 999
            return (group, order, entry["id"])
        return (group, 0 if entry["json_capable"] else 1, -entry["context"], entry["id"])

    return sorted(models, key=key)


def openrouter_models(items) -> list[dict]:
    models = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        model_id = str(item.get("id") or "")
        if not model_id.endswith(":free"):
            continue
        params = set(item.get("supported_parameters") or [])
        modality = str((item.get("architecture") or {}).get("modality") or "")
        name = str(item.get("name") or model_id)
        desc = str(item.get("description") or "")
        hay = f"{model_id} {name} {desc}".lower()
        context = item.get("context_length")
        json_capable = "response_format" in params or "structured_outputs" in params

        if model_id in OPENROUTER_RECOMMENDED:
            group = "recommended"
            note = OPENROUTER_RECOMMENDED[model_id]
        elif model_id in OPENROUTER_SPECIALIZED:
            group = "specialized"
            note = OPENROUTER_SPECIALIZED[model_id]
        elif any(hint in hay for hint in TASK_HINTS):
            group = "specialized"
            note = "classifier / guardrail · may not follow the JSON schema"
        elif "structured_outputs" in params and modality.startswith("text"):
            group = "recommended"
            note = "structured JSON output"
        else:
            group = "general"
            if modality and not modality.startswith("text->text"):
                note = "multimodal model"
            elif any(token in hay for token in ("coding", "agent", "terminal")):
                note = "coding / agent model · not task-specialised"
            elif json_capable:
                note = "JSON-capable"
            else:
                note = "general text model"

        models.append(
            _entry(
                model_id,
                name,
                context,
                group,
                note,
                json_capable=json_capable,
                reasoning="reasoning" in params,
            )
        )
    return _sort_models(models, list(OPENROUTER_RECOMMENDED))


def _groq_reasons(model_id: str) -> bool:
    return model_id.startswith(("openai/gpt-oss", "qwen/qwen3"))


def groq_entries(live_items=None) -> list[dict]:
    """Groq free-plan entries; live_items=None returns the curated fallback."""
    live = {}
    for item in live_items or []:
        if isinstance(item, dict) and item.get("id"):
            live[str(item["id"])] = item

    models = []
    for model_id, note in {**GROQ_FREE_CHAT, **GROQ_FREE_SPECIALIZED}.items():
        if live_items is not None and model_id not in live:
            continue
        item = live.get(model_id, {})
        group = "recommended" if model_id in GROQ_FREE_CHAT else "specialized"
        models.append(
            _entry(
                model_id,
                str(item.get("name") or model_id),
                item.get("context_window"),
                group,
                note,
                json_capable=group == "recommended",
                reasoning=_groq_reasons(model_id),
            )
        )
    priority = list(GROQ_FREE_CHAT) + list(GROQ_FREE_SPECIALIZED)
    return _sort_models(models, priority)


def _groq_fallback(note: str) -> dict:
    models = groq_entries()
    return {
        "provider": "groq",
        "label": "Groq",
        "models": models,
        "default": models[0]["id"] if models else PROVIDERS["groq"]["default_model"],
        "source": "fallback",
        "note": note,
    }


def _remember(provider: str, result: dict) -> dict:
    with _cache_lock:
        _MODEL_CACHE[provider].update(at=time.monotonic(), payload=result)
    return result


def list_models(provider: str, api_key: str = "", force: bool = False) -> dict:
    """Return only free-tier models, ranked for sentence-JSON classification.

    OpenRouter exposes a public catalogue and marks free models with ``:free``.
    Groq requires a key and does not flag plan eligibility in its API, so the
    documented Free Plan set is used and intersected with the live catalogue.
    """
    cfg = get_provider(provider)
    key = cfg["key"]
    cached = _MODEL_CACHE[key]

    if key == "openrouter":
        if (
            not force
            and cached["payload"]
            and time.monotonic() - cached["at"] < _CACHE_TTL
        ):
            return cached["payload"]
        try:
            response = client.get(f"{cfg['base_url']}/models", timeout=8)
            response.raise_for_status()
            payload = response.json()
            models = openrouter_models(
                payload.get("data") if isinstance(payload, dict) else []
            )
        except Exception as exc:
            if cached["payload"]:
                return {**cached["payload"], "source": "cached"}
            raise DetectorError(
                "Could not load the OpenRouter free-model catalogue.",
                status=502,
                detail=str(exc)[:300],
            ) from exc

        return _remember(
            key,
            {
                "provider": "openrouter",
                "label": "OpenRouter",
                "models": models,
                "default": models[0]["id"] if models else DEFAULT_MODEL,
                "source": "live",
                "note": "Only :free models · ranked by JSON task fit.",
            },
        )

    # Groq: the model list is account-scoped and key-protected.
    if not api_key or len(api_key.strip()) < 12:
        return _groq_fallback(
            "Groq Free Plan models. Add a key in Settings to verify availability."
        )

    # Not served from cache: the listing re-validates the key and plan.
    try:
        response = client.get(
            f"{cfg['base_url']}/models", api_key=api_key, provider="groq", timeout=10
        )
    except requests.RequestException as exc:
        return _groq_fallback(
            f"Could not reach Groq ({str(exc)[:80]}). Showing the documented Free Plan list."
        )

    if response.status_code in (401, 403):
        raise DetectorError(
            "Groq rejected the API key. Check it in Settings.",
            status=401,
            detail=client.upstream_detail(response),
        )

    if response.status_code != 200:
        return _groq_fallback(
            f"Groq returned HTTP {response.status_code}. Showing the documented Free Plan list."
        )

    try:
        payload = response.json()
    except ValueError:
        return _groq_fallback("Groq returned a non-JSON model list.")

    models = groq_entries(payload.get("data") if isinstance(payload, dict) else [])
    if not models:
        return _groq_fallback(
            "None of the documented Free Plan models were returned by Groq. "
            "The plan may have changed."
        )

    return _remember(
        key,
        {
            "provider": "groq",
            "label": "Groq",
            "models": models,
            "default": models[0]["id"],
            "source": "live",
            "note": "Only Groq Free Plan text models · 8K TPM, 1K RPD.",
        },
    )


def fallback_models(provider: str, limit: int = 4) -> list[str]:
    """Models worth switching to when the selected one fails, best first.

    Safety/guardrail classifiers are never used as fallbacks: they tend to
    answer with their own labels instead of the sentence schema.
    """
    key = get_provider(provider)["key"]
    if key == "groq":
        payload = _MODEL_CACHE["groq"]["payload"]
        models = payload["models"] if payload else groq_entries()
    else:
        try:
            models = list_models("openrouter")["models"]
        except DetectorError:
            models = [_entry(m, m, 0, "recommended", "") for m in OPENROUTER_RECOMMENDED]

    ranked = [m["id"] for m in models if m["group"] == "recommended"]
    ranked += [
        m["id"] for m in models if m["group"] == "general" and m["json_capable"]
    ]
    return ranked[:limit]


def is_reasoning(provider: str, model: str) -> bool:
    """Whether the model spends hidden reasoning tokens before answering."""
    key = get_provider(provider)["key"]
    if key == "groq":
        return _groq_reasons(model)
    payload = _MODEL_CACHE["openrouter"]["payload"]
    for entry in (payload or {}).get("models", []):
        if entry["id"] == model:
            return entry.get("reasoning", False)
    return False


def request_options(provider: str, model: str) -> dict:
    """Optional chat parameters that cut hidden reasoning tokens.

    Only sent where the model is known to support them; the client drops
    them and retries if a provider still rejects the request (HTTP 400).
    """
    key = get_provider(provider)["key"]
    if key == "groq" and model.startswith("openai/gpt-oss"):
        return {"reasoning_effort": "low"}
    if key == "openrouter" and is_reasoning(key, model):
        return {"reasoning": {"effort": "low", "exclude": True}}
    return {}
