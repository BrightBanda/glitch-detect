"""Error hierarchy shared by the transport, parser and analyzer."""

from __future__ import annotations


class DetectorError(Exception):
    """Raised for configuration, transport, or parsing failures."""

    def __init__(self, message: str, *, status: int = 502, detail: str = "", code: str = ""):
        super().__init__(message)
        self.message = message
        self.status = status
        self.detail = detail
        # Machine-readable reason for the UI, e.g. "quota", "auth_required".
        self.code = code


class ParseError(DetectorError):
    def __init__(self, message: str, detail: str = ""):
        super().__init__(message, status=502, detail=detail)


class ModelFailure(DetectorError):
    """A failure tied to one model; a different model may still succeed."""


class RateLimited(ModelFailure):
    def __init__(self, message: str, *, detail: str = "", retry_after: float = 0.0,
                 scope: str = "model"):
        super().__init__(message, status=429, detail=detail)
        self.retry_after = retry_after
        # "model": just this model is limited; "provider": an account-wide cap
        # (e.g. OpenRouter's free models per day) blocks every model there.
        self.scope = scope
