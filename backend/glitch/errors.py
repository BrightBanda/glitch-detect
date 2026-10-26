"""Error hierarchy shared by the transport, parser and analyzer."""

from __future__ import annotations


class DetectorError(Exception):
    """Raised for configuration, transport, or parsing failures."""

    def __init__(self, message: str, *, status: int = 502, detail: str = ""):
        super().__init__(message)
        self.message = message
        self.status = status
        self.detail = detail


class ParseError(DetectorError):
    def __init__(self, message: str, detail: str = ""):
        super().__init__(message, status=502, detail=detail)


class ModelFailure(DetectorError):
    """A failure tied to one model; a different model may still succeed."""


class RateLimited(ModelFailure):
    def __init__(self, message: str, *, detail: str = "", retry_after: float = 0.0):
        super().__init__(message, status=429, detail=detail)
        self.retry_after = retry_after
