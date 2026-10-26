"""API keys: work out which provider a key belongs to, and hold the server's keys.

Adding a key never requires naming its provider:

1. **Recognise** — the key is matched against each provider's key patterns
   (``gsk_…`` → Groq, ``AIza…`` → Gemini, ``sk-or-…`` → OpenRouter, …).
2. **Confirm** — the matching provider's ``/models`` endpoint is called with
   the key. If a key fits more than one pattern, only those candidates are
   tried. A key is never sent to a provider whose pattern it does not match.

Server keys come from two places, highest priority first:

* keys added at runtime (admin dashboard / ``KeyStore.add``), saved to
  ``instance/keys.json`` with owner-only permissions;
* environment / ``.env``: each provider's own variables (``GROQ_API_KEY``…)
  plus ``LLM_API_KEYS``, a comma-separated list recognised automatically.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import requests

from . import client, config
from .errors import DetectorError
from .providers import PROVIDERS, get_provider

log = logging.getLogger("glitch")

PROBE_TIMEOUT_S = 8
GENERIC_ENV_VAR = "LLM_API_KEYS"


@dataclass
class KeyMatch:
    provider: str
    verified: bool  # False when the provider could not be reached to confirm


def mask(api_key: str) -> str:
    api_key = api_key.strip()
    return f"{api_key[:4]}…{api_key[-4:]}" if len(api_key) > 12 else "…"


def identify(api_key: str) -> list[str]:
    """Providers whose key format matches, in registry order."""
    api_key = (api_key or "").strip()
    return [spec.id for spec in PROVIDERS.values() if spec.recognises(api_key)]


def probe(api_key: str, provider: str) -> bool | None:
    """True if the provider accepts the key, False if it rejects it, None if unknown."""
    spec = get_provider(provider)
    try:
        response = client.get(
            f"{spec.base_url}/models", api_key=api_key, provider=spec.id, timeout=PROBE_TIMEOUT_S
        )
    except requests.RequestException:
        return None
    if response.status_code == 200:
        return True
    if response.status_code in (401, 403):
        return False
    return None


def resolve(api_key: str, *, verify: bool = True) -> KeyMatch:
    """Work out which provider ``api_key`` belongs to."""
    api_key = (api_key or "").strip()
    if len(api_key) < config.MIN_KEY_LENGTH or any(c.isspace() for c in api_key):
        raise DetectorError("That doesn't look like an API key.", status=400)

    candidates = identify(api_key)
    if not candidates:
        raise DetectorError(
            "We don't recognise this kind of key yet. Supported: "
            + ", ".join(spec.label for spec in PROVIDERS.values())
            + ".",
            status=400,
        )
    if not verify:
        return KeyMatch(candidates[0], verified=False)

    with ThreadPoolExecutor(max_workers=len(candidates)) as pool:
        results = dict(zip(candidates, pool.map(lambda p: probe(api_key, p), candidates)))
    for provider in candidates:
        if results[provider] is True:
            return KeyMatch(provider, verified=True)
    unreachable = [p for p in candidates if results[p] is None]
    if len(candidates) == 1 and unreachable:
        # Provider down or offline: trust the unambiguous key format.
        return KeyMatch(candidates[0], verified=False)
    labels = " or ".join(get_provider(p).label for p in candidates)
    if unreachable:
        raise DetectorError(
            f"Couldn't reach {labels} to confirm this key. Try again in a moment.", status=503
        )
    raise DetectorError(f"{labels} rejected this key. Check that it's correct.", status=401)


class KeyStore:
    """The API keys the server may use, by provider."""

    def __init__(self, path=None):
        self._path = path or (config.INSTANCE_DIR / "keys.json")
        self._lock = threading.Lock()
        self._entries: list[dict] | None = None  # loaded lazily
        self._generic: dict[str, str] | None = None

    # -- persistence -------------------------------------------------------
    def _load(self) -> list[dict]:
        if self._entries is None:
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
                self._entries = [e for e in data.get("keys", []) if e.get("provider") in PROVIDERS]
            except (OSError, ValueError):
                self._entries = []
        return self._entries

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump({"keys": self._entries}, fh, indent=2)
        os.replace(tmp, self._path)

    # -- environment -------------------------------------------------------
    def _env_keys(self) -> dict[str, str]:
        out = {}
        for spec in PROVIDERS.values():
            for name in spec.env_vars:
                value = os.environ.get(name, "").strip()
                if value:
                    out[spec.id] = value
                    break
        if self._generic is None:
            # Recognised by format only: probing every key at startup would
            # slow the boot and the format is unambiguous for most providers.
            self._generic = {}
            for value in os.environ.get(GENERIC_ENV_VAR, "").split(","):
                value = value.strip()
                if not value:
                    continue
                try:
                    self._generic.setdefault(resolve(value, verify=False).provider, value)
                except DetectorError:
                    log.warning("%s: ignored an unrecognised key (%s)", GENERIC_ENV_VAR, mask(value))
        for provider, value in self._generic.items():
            out.setdefault(provider, value)
        return out

    # -- public API --------------------------------------------------------
    def key_for(self, provider: str) -> str:
        with self._lock:
            for entry in self._load():
                if entry["provider"] == provider:
                    return entry["key"]
        return self._env_keys().get(provider, "")

    def ready_providers(self) -> set[str]:
        with self._lock:
            stored = {e["provider"] for e in self._load()}
        return stored | set(self._env_keys())

    def add(self, api_key: str, label: str = "", *, verify: bool = True) -> dict:
        """Recognise, confirm and store a key. Returns the masked entry."""
        api_key = (api_key or "").strip()
        match = resolve(api_key, verify=verify)
        with self._lock:
            entries = self._load()
            for entry in entries:
                if entry["key"] == api_key:
                    return self._public(entry)
            entry = {
                "id": secrets.token_hex(6),
                "provider": match.provider,
                "key": api_key,
                "label": (label or "").strip()[:60],
                "verified": match.verified,
                "added_at": int(time.time()),
            }
            entries.insert(0, entry)  # newest key for a provider wins
            self._save()
        return self._public(entry)

    def remove(self, entry_id: str) -> bool:
        with self._lock:
            entries = self._load()
            kept = [e for e in entries if e["id"] != entry_id]
            if len(kept) == len(entries):
                return False
            self._entries = kept
            self._save()
        return True

    def list(self) -> list[dict]:
        """All keys, masked, including read-only ones from the environment."""
        with self._lock:
            stored = [self._public(e) for e in self._load()]
        env = [
            {
                "id": f"env:{provider}",
                "provider": provider,
                "provider_label": PROVIDERS[provider].label,
                "key": mask(value),
                "label": "From server environment",
                "source": "env",
            }
            for provider, value in self._env_keys().items()
        ]
        return stored + env

    @staticmethod
    def _public(entry: dict) -> dict:
        return {
            "id": entry["id"],
            "provider": entry["provider"],
            "provider_label": PROVIDERS[entry["provider"]].label,
            "key": mask(entry["key"]),
            "label": entry.get("label", ""),
            "verified": entry.get("verified", False),
            "added_at": entry.get("added_at", 0),
            "source": "admin",
        }


store = KeyStore()


def resolve_request_keys(user_keys: dict | None = None) -> dict[str, str]:
    """Key per provider for one request: the user's own key, else the server's."""
    user_keys = user_keys if isinstance(user_keys, dict) else {}
    out = {}
    for provider in PROVIDERS:
        value = str(user_keys.get(provider) or "").strip()
        out[provider] = value or store.key_for(provider)
    return out
