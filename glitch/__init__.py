"""Glitch//Detect — local-first, sentence-level AI writing forensics.

Module map:
    config      .env loading, endpoints, limits, key resolution
    providers   provider registry and free-tier model rules
    catalogue   free-model listing, ranking, fallback order
    client      pooled HTTP transport and upstream error mapping
    prompt      forensic system prompt and per-chunk message
    parsing     tolerant JSON extraction and label normalisation
    text        sentence splitting, chunking, highlight HTML
    analyzer    parallel scan, model fallback, scoring
    humanize    placeholder for the rewrite feature
    web         Flask app factory and routes
"""

from .web import create_app

__all__ = ["create_app"]
