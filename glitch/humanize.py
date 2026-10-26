"""Humanize: rewrite flagged text so it reads as natural human prose.

Placeholder. The route, UI panel and signature exist so the feature can be
built without touching the rest of the app.
"""

from __future__ import annotations

from .errors import DetectorError


def humanize(text: str, *, provider: str, model: str, api_key: str) -> dict:
    raise DetectorError("Humanize is not available yet.", status=501)
