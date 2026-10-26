"""Model catalogue: listing, ranking, fallback order, per-model request options.

OpenRouter and Groq have dedicated rules (free models only). Every other
provider is listed from its own ``/models`` endpoint, filtered to chat models,
and ranked for this job: fast, instruction-following models first.
"""

from __future__ import annotations

import re
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
    PAID,
    PROVIDERS,
    get_provider,
)

TASK_HINTS = ("safety", "guard", "moderation", "classif", "detect", "forensic", "judge")
GROUP_ORDER = ("recommended", "specialized", "general")
GROUP_LABELS = {
    "recommended": "RECOMMENDED — BEST FOR SENTENCE JSON",
    "specialized": "SPECIALIZED CLASSIFIERS — EXPERIMENTAL",
    "general": "OTHER MODELS",
}

# Models that cannot do chat-style text classification.
_NON_CHAT = re.compile(
    r"embed|whisper|tts|audio|speech|transcri|image|dall-e|imagen|veo|vision-preview|"
    r"moderation|guard|rerank|realtime|search|aqa|learnlm|ocr|codestral-mamba",
    re.IGNORECASE,
)
# Ranking hints for generic providers: fast instruction models suit this job.
_PREFER = re.compile(r"flash|mini|lite|small|instant|instruct|turbo|haiku|chat", re.IGNORECASE)
_AVOID = re.compile(
    r"preview|exp|thinking|reasoner|large|pro\b|opus|405b|deep-research|code", re.IGNORECASE
)

_CACHE_TTL = 600
_cache_lock = threading.Lock()
_MODEL_CACHE: dict[str, dict] = {pid: {"at": 0.0, "payload": None} for pid in PROVIDERS}


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


def groq_entries(live_items=None) -> list[dict]:
    """Groq free-plan entries; live_items=None returns the curated fallback."""
    spec = PROVIDERS["groq"]
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
                reasoning=bool(re.search(spec.reasoning_models, model_id)),
            )
        )
    priority = list(GROQ_FREE_CHAT) + list(GROQ_FREE_SPECIALIZED)
    return _sort_models(models, priority)


def generic_models(provider: str, items) -> list[dict]:
    """Chat models from any OpenAI-compatible ``/models`` list, ranked for this task."""
    spec = get_provider(provider)
    seen, scored = set(), []
    for item in items or []:
        if not isinstance(item, dict) or not item.get("id"):
            continue
        model_id = str(item["id"]).removeprefix("models/")
        if model_id in seen or _NON_CHAT.search(model_id):
            continue
        seen.add(model_id)
        curated = list(spec.allowlist).index(model_id) if model_id in spec.allowlist else None
        score = (
            0 if curated is not None else 1,
            curated or 0,
            0 if _PREFER.search(model_id) else 1,
            1 if _AVOID.search(model_id) else 0,
            model_id,
        )
        scored.append((score, model_id, item))
    scored.sort()

    models = []
    for rank, (_, model_id, item) in enumerate(scored):
        group = "recommended" if rank < 3 else "general"
        models.append(
            _entry(
                model_id,
                str(item.get("name") or item.get("display_name") or model_id),
                item.get("context_length") or item.get("context_window"),
                group,
                spec.allowlist.get(model_id, ""),
                json_capable=True,
                reasoning=bool(spec.reasoning_models and re.search(spec.reasoning_models, model_id)),
            )
        )
    return models


def _remember(provider: str, result: dict) -> dict:
    with _cache_lock:
        _MODEL_CACHE[provider].update(at=time.monotonic(), payload=result)
    return result


def _result(provider: str, models: list[dict], source: str, note: str) -> dict:
    spec = get_provider(provider)
    return {
        "provider": spec.id,
        "label": spec.label,
        "billing": spec.billing,
        "models": models,
        "default": models[0]["id"] if models else spec.default_model,
        "source": source,
        "note": note,
    }


def _groq_fallback(note: str) -> dict:
    return _result("groq", groq_entries(), "fallback", note)


def list_models(provider: str, api_key: str = "", force: bool = False) -> dict:
    """Allowed models for a provider, ranked for sentence-JSON classification."""
    spec = get_provider(provider)
    cached = _MODEL_CACHE[spec.id]

    if spec.id == "openrouter":
        if not force and cached["payload"] and time.monotonic() - cached["at"] < _CACHE_TTL:
            return cached["payload"]
        try:
            response = client.get(f"{spec.base_url}/models", timeout=8)
            response.raise_for_status()
            payload = response.json()
            models = openrouter_models(payload.get("data") if isinstance(payload, dict) else [])
        except Exception as exc:
            if cached["payload"]:
                return {**cached["payload"], "source": "cached"}
            raise DetectorError(
                "Could not load the OpenRouter free-model catalogue.",
                status=502,
                detail=str(exc)[:300],
            ) from exc
        return _remember(
            spec.id, _result(spec.id, models, "live", "Only :free models · ranked by JSON task fit.")
        )

    if spec.id == "groq" and (not api_key or len(api_key.strip()) < 12):
        return _groq_fallback("Groq Free Plan models. Add a key in Settings to verify availability.")
    if not api_key or len(api_key.strip()) < 12:
        raise DetectorError(f"Add a {spec.label} key to see its models.", status=400)

    if not force and cached["payload"] and time.monotonic() - cached["at"] < _CACHE_TTL:
        return cached["payload"]

    try:
        response = client.get(
            f"{spec.base_url}/models", api_key=api_key, provider=spec.id, timeout=10
        )
    except requests.RequestException as exc:
        if spec.id == "groq":
            return _groq_fallback(
                f"Could not reach Groq ({str(exc)[:80]}). Showing the documented Free Plan list."
            )
        raise DetectorError(f"Could not reach {spec.label}.", status=502, detail=str(exc)[:300]) from exc

    if response.status_code in (401, 403):
        raise DetectorError(
            f"{spec.label} rejected the API key. Check it in Settings.",
            status=401,
            detail=client.upstream_detail(response),
        )
    if response.status_code != 200:
        if spec.id == "groq":
            return _groq_fallback(
                f"Groq returned HTTP {response.status_code}. Showing the documented Free Plan list."
            )
        raise DetectorError(f"{spec.label} returned HTTP {response.status_code}.", status=502)

    try:
        payload = response.json()
    except ValueError:
        if spec.id == "groq":
            return _groq_fallback("Groq returned a non-JSON model list.")
        raise DetectorError(f"{spec.label} returned an unreadable model list.", status=502)

    items = payload.get("data") if isinstance(payload, dict) else []
    if spec.id == "groq":
        models = groq_entries(items)
        if not models:
            return _groq_fallback(
                "None of the documented Free Plan models were returned by Groq. "
                "The plan may have changed."
            )
        return _remember(
            spec.id, _result(spec.id, models, "live", "Only Groq Free Plan text models · 8K TPM, 1K RPD.")
        )

    models = generic_models(spec.id, items)
    note = "Billed to the key owner." if spec.billing == PAID else "Uses this key's free allowance."
    return _remember(spec.id, _result(spec.id, models, "live", note))


def fallback_models(provider: str, api_key: str = "", limit: int = 4) -> list[str]:
    """Models worth switching to when the selected one fails, best first.

    Safety/guardrail classifiers are never used as fallbacks: they tend to
    answer with their own labels instead of the sentence schema.
    """
    spec = get_provider(provider)
    payload = _MODEL_CACHE[spec.id]["payload"]
    if payload:
        models = payload["models"]
    elif spec.id == "groq":
        models = groq_entries()
    elif spec.id == "openrouter":
        try:
            models = list_models("openrouter")["models"]
        except DetectorError:
            models = [_entry(m, m, 0, "recommended", "") for m in OPENROUTER_RECOMMENDED]
    elif api_key:
        try:
            models = list_models(spec.id, api_key)["models"]
        except DetectorError:
            models = []
    else:
        models = []

    ranked = [m for m in models if m["group"] == "recommended"]
    ranked += [m for m in models if m["group"] == "general" and m["json_capable"]]
    # Prefer models that answer directly: a reasoning model can spend hundreds
    # of hidden tokens to emit a handful of labels. Stable sort keeps rank order.
    ranked.sort(key=lambda m: m.get("reasoning", False))
    return [m["id"] for m in ranked][:limit]


def is_reasoning(provider: str, model: str) -> bool:
    """Whether the model spends hidden reasoning tokens before answering."""
    spec = get_provider(provider)
    payload = _MODEL_CACHE[spec.id]["payload"]
    for entry in (payload or {}).get("models", []):
        if entry["id"] == model:
            return entry.get("reasoning", False)
    return bool(spec.reasoning_models and re.search(spec.reasoning_models, model))


def request_options(provider: str, model: str) -> dict:
    """Optional chat parameters that cut hidden reasoning tokens.

    Only sent where the model is known to support them; the client drops
    them and retries if a provider still rejects the request (HTTP 400).
    """
    spec = get_provider(provider)
    if spec.id == "openrouter":
        return {"reasoning": {"effort": "low", "exclude": True}} if is_reasoning(spec.id, model) else {}
    if spec.low_effort_models and re.search(spec.low_effort_models, model):
        return {"reasoning_effort": "low"}
    return {}
