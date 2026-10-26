#!/usr/bin/env python3
"""Test double for the OpenRouter chat-completions endpoint.

Used only by the end-to-end UI test. It mimics the wire format closely enough
to exercise the real parser, scoring and highlighting code paths without
spending API credits. Never imported by the application.
"""

from __future__ import annotations

import json
import re

from flask import Flask, jsonify, request

app = Flask(__name__)

SPLIT = re.compile(r'(?<=[.!?])\s+')


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
    if len(sentence.split()) < 6:
        return "HUMAN"
    return "HUMAN"


@app.get("/v1/models")
def models():
    """Union of OpenRouter-style free models and Groq Free Plan models."""
    return jsonify(
        {
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
    )


@app.post("/v1/chat/completions")
def completions():
    body = request.get_json(force=True, silent=True) or {}
    messages = body.get("messages") or []
    user = ""
    for message in messages:
        if message.get("role") == "user":
            user = message.get("content") or ""

    match = re.search(r"===DATA START===\s*(.*?)\s*===DATA END===", user, re.DOTALL)
    data = match.group(1) if match else user

    sentences = [s.strip() for s in SPLIT.split(re.sub(r"\s+", " ", data)) if s.strip()]
    payload = {
        "sentences": [
            {"text": sentence, "classification": classify(sentence)}
            for sentence in sentences
        ]
    }

    # Deliberately fenced: the production parser must strip it.
    content = "```json\n" + json.dumps(payload, indent=2) + "\n```"

    return jsonify(
        {
            "id": "gen-test",
            "model": body.get("model", "stub"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8599, debug=False, use_reloader=False)
