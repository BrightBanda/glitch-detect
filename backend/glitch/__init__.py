"""Glitch//Detect — local-first, sentence-level AI writing forensics.

Module map:
    config      .env loading, limits, paths, admin token
    providers   provider registry: URLs, key patterns, billing, model rules
    keys        recognise a key's provider, verify it, server key store
    catalogue   model listing and ranking, fallback order, reasoning options
    routing     shared job runner: balancing, fallback, cooldowns
    client      pooled HTTP transport and upstream error mapping
    cache       thread-safe TTL/LRU cache keyed by text fingerprints
    prompt      detection prompts (quick + verbatim classic) and rewrite prompt
    parsing     tolerant parsers for labels, sentences and rewrites
    text        sentence splitting, chunking, highlight HTML, span replacement
    style       zero-token style analysis: rhythm, AI phrases, voice, grounding
    analyzer    detection: jobs, sentence/section caches, scoring
    humanize    rewrite: passages, notes, validation, re-scoring
    api         FastAPI app: routes, request models, frontend serving
"""

from .api import create_app

__all__ = ["create_app"]
