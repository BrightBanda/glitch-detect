#!/usr/bin/env python3
"""Test double for the OpenRouter / Groq chat-completions endpoints.

Used only by the end-to-end UI test. It mimics the wire format closely enough
to exercise the real parser, scoring and highlighting code paths without
spending API credits. Never imported by the application.

Set ``STUB_RATE_LIMITED`` to a comma-separated list of model IDs to make those
models answer HTTP 429, which exercises automatic model fallback.
"""

from __future__ import annotations

import json
import os
import re

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="Glitch//Detect provider stub")

SPLIT = re.compile(r'(?<=[.!?])\s+')
RATE_LIMITED = {
    m.strip() for m in os.environ.get("STUB_RATE_LIMITED", "").split(",") if m.strip()
}


def classify(sentence: str) -> str:
    lowered = sentence.lower()
    if any(marker in lowered for marker in ("ultimately", "moreover", "in conclusion")):
        return "REFINED"
    if any(
        marker in lowered
        for marker in ("crucial", "landscape", "testament", "leverage", "foster", "stakeholders")
    ):
        return "AI"
    if any(marker in lowered for marker in ("i tried", "we fixed", "colleague", "tuesday", "i only")):
        return "HUMAN"
    return "HUMAN"


STOCK_PHRASES = {
    "ultimately, ": "", "moreover, ": "", "in today's rapidly evolving digital landscape, ": "these days ",
    "plays a crucial role in": "matters for", "crucial": "big", "landscape": "scene",
    "a testament to": "proof of", "testament": "proof", "leveraging": "using", "leverage": "use",
    "foster": "build", "stakeholders": "people involved",
}


def humanize(sentence: str) -> str:
    """Crude stand-in rewrite: strips stock phrases so classify() reads it as HUMAN."""
    out = sentence
    for phrase, plain in STOCK_PHRASES.items():
        out = re.sub(re.escape(phrase), plain, out, flags=re.IGNORECASE)
    return out[:1].upper() + out[1:]


@app.get("/v1/models")
def models():
    """Union of OpenRouter-style free models and Groq Free Plan models."""
    return {
            "object": "list",
            "data": [
                {
                    "id": "google/gemma-4-31b-it:free",
                    "name": "Google: Gemma 4 31B (free)",
                    "context_length": 262144,
                    "supported_parameters": ["response_format"],
                    "architecture": {"modality": "text->text"},
                },
                {
                    "id": "nvidia/nemotron-3-super-120b-a12b:free",
                    "name": "NVIDIA: Nemotron 3 Super (free)",
                    "context_length": 262144,
                    "supported_parameters": ["response_format", "structured_outputs"],
                    "architecture": {"modality": "text->text"},
                },
                {"id": "openai/gpt-oss-120b", "context_window": 131072, "active": True},
                {"id": "qwen/qwen3.8-27b", "context_window": 131072, "active": True},
                {"id": "openai/gpt-oss-20b", "context_window": 131072, "active": True},
                {
                    "id": "openai/gpt-oss-safeguard-20b",
                    "context_window": 131072,
                    "active": True,
                },
                {"id": "llama-3.3-70b-versatile", "context_window": 131072, "active": True},
                {"id": "whisper-large-v3", "context_window": 0, "active": True},
            ],
        }


@app.post("/v1/chat/completions")
async def completions(request: Request):
    try:
        body = await request.json()
    except ValueError:
        body = {}
    if body.get("model") in RATE_LIMITED:
        return JSONResponse(
            {"error": {"message": "Rate limit exceeded (stub)", "code": 429}},
            status_code=429,
            headers={"Retry-After": "30"},
        )
    messages = body.get("messages") or []
    user = ""
    for message in messages:
        if message.get("role") == "user":
            user = message.get("content") or ""

    match = re.search(r"===DATA START===\s*(.*?)\s*===DATA END===", user, re.DOTALL)
    data = match.group(1) if match else user
    system = messages[0].get("content") or ""
    compact = '"labels"' in system

    if '"rewrites"' in system:
        targets = re.findall(r"^\[(\d+)\] (.*)$", data, re.MULTILINE)
        payload = {"rewrites": {n: humanize(text) for n, text in targets}}
    elif compact:
        sentences = [re.sub(r"^\d+\.\s*", "", line) for line in data.splitlines() if line.strip()]
        payload = {"labels": [classify(sentence) for sentence in sentences]}
    else:
        sentences = [s.strip() for s in SPLIT.split(re.sub(r"\s+", " ", data)) if s.strip()]
        payload = {
            "sentences": [
                {"text": sentence, "classification": classify(sentence)}
                for sentence in sentences
            ]
        }

    # Deliberately fenced: the production parser must strip it.
    content = "```json\n" + json.dumps(payload) + "\n```"
    prompt_tokens = sum(len(m.get("content") or "") for m in messages) // 4

    return {
            "id": "gen-test",
            "model": body.get("model", "stub"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": len(content) // 4},
        }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8599, log_level="warning")
