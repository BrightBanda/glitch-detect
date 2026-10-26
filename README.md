# Glitch//Detect

A local-first, sentence-level AI writing detector with a cyberpunk interface.
Text is analysed remotely by a **free OpenRouter or Groq model**; the app runs
entirely on your machine and keeps API keys in your browser tab.

![status](https://img.shields.io/badge/node-local-22D3EE) ![python](https://img.shields.io/badge/python-3.10%2B-3B82F6) ![transport](https://img.shields.io/badge/transport-OpenRouter%20%2B%20Groq-F59E0B)

---

## Quick start

```bash
cd glitch-detect
./run.sh
```

`run.sh` creates a virtualenv on first run, installs Flask + requests, and prints:

```
  LOCAL FORENSICS NODE  ·  OpenRouter + Groq transport
  ➜  http://127.0.0.1:8501
```

Open that URL, choose **OPENROUTER** or **GROQ**, paste the matching key
([OpenRouter](https://openrouter.ai/keys) · [Groq](https://console.groq.com/keys)),
paste text, and hit **RUN FORENSIC SCAN**. Only free-tier models are listed and
accepted.

Manual equivalent:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python app.py
```

Environment variables: `PORT` (default `8501`), `OPENROUTER_BASE_URL`
(default `https://openrouter.ai/api/v1`), `GROQ_BASE_URL`
(default `https://api.groq.com/openai/v1`).

---

## What it does

1. Splits the document into API-sized chunks on paragraph boundaries.
2. Sends each chunk to the selected provider's OpenAI-compatible
   chat-completions endpoint with the mandated forensic system prompt (verbatim,
   see `detector.py`).
3. Parses the returned JSON — with fence stripping, trailing-comma repair, and a
   structural regex fallback for truncated or prose-wrapped responses.
4. Scores the document as `(AI + REFINED) / total × 100`.
5. Rebuilds the original text as inline `<span>`s, preserving paragraphs, colour-coded:

| Class | Colour | Meaning |
|---|---|---|
| `AI` | `#EF4444` red | Machine-generated: high predictability, flat rhythm |
| `REFINED` | `#F59E0B` amber | Paraphrased / polished over an AI base |
| `HUMAN` | `#3B82F6` blue | Organic variance, idiom, imperfection |

---

## Architecture

```
browser (static/app.js)
   │  POST /api/analyze  { text, provider, api_key, model, chunk_words }
   ▼
Flask (app.py) ── validation, rate/error mapping, security headers
   │
   ▼
detector.py ── chunking → provider transport → fence/JSON/repair parsing → scoring → HTML
   │
   ├── https://openrouter.ai/api/v1/chat/completions
   └── https://api.groq.com/openai/v1/chat/completions
```

| File | Role |
|---|---|
| `app.py` | Flask routes, validation, error envelope, CSP headers, launcher |
| `detector.py` | System prompt, chunking, transport, parsing, scoring, highlighting |
| `templates/index.html` | Dashboard markup, SVG gauge, legend |
| `static/style.css` | Design tokens, glitch animations, highlight classes |
| `static/app.js` | Fetch flow, rendering, gauge animation, clipboard/export |
| `tests/test_detector.py` | 28 unit tests: parsing, escaping, scoring, free-model rules, error mapping |
| `tests/stub_openrouter.py` | Local test double used by the E2E UI check |

### Parsing ladder

The free model is instructed to return raw JSON, but it frequently does not.
`parse_analysis()` degrades in four stages:

1. `json.loads` on the outermost `{…}` (after stripping ```` ```json ```` fences).
2. Trailing-comma repair, then parse again.
3. Structural regex over `{"text": …, "classification": …}` objects, in either
   key order — survives truncation and prose wrappers.
4. Fail with a diagnostic containing the first 280 characters of the response.

Labels are normalised (`paraphrased`, `machine-generated`, `polished` → the three
canonical classes). Unknown labels default to `HUMAN` and raise a warning rather
than inflating the AI score.

---

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Dashboard |
| `GET` | `/api/health` | Node status, providers, defaults |
| `POST` | `/api/models` | `{provider, api_key?, force?}` → ranked free-model list |
| `POST` | `/api/analyze` | `{text, provider, api_key, model, chunk_words}` → score + sentences + HTML |

Model listing (OpenRouter needs no key; Groq needs the key to verify availability):

```bash
curl -s http://127.0.0.1:8501/api/models \
  -H 'Content-Type: application/json' \
  -d '{"provider":"groq","api_key":"gsk_…"}'
```

Analysis:

```bash
curl -s http://127.0.0.1:8501/api/analyze \
  -H 'Content-Type: application/json' \
  -d '{"provider":"openrouter","api_key":"sk-or-v1-…","model":"google/gemma-4-31b-it:free","text":"Your text here. Second sentence."}'
```

Non-free models are rejected **before** any network call: OpenRouter IDs must end
in `:free`, and Groq IDs must be on the documented Free Plan list.

---

## Security & privacy

- Keys are **never written to disk by the server**. They are held in the browser
  tab and sent with each analysis request; only the key for the selected provider
  is used.
- Optional "keep in this tab" uses per-provider `sessionStorage` entries (cleared
  when the tab closes). Switching providers keeps each key separate.
- The server binds to `127.0.0.1` only and sets `Cache-Control: no-store`,
  `X-Frame-Options: DENY`, and a restrictive CSP.
- Model output is HTML-escaped before it is wrapped in highlight spans, so a
  document containing `<script>` cannot inject markup.
- API keys are not logged; failures log only status codes and messages.

---

## Free-model notes

Free endpoints are rate-limited, queue aggressively, and rotate. The picker only
shows free models and ranks them for this specific job — sentence-level JSON
classification:

1. **Recommended** — structured-output / instruction-tuned models that reliably
   follow the sentence schema.
2. **Specialized classifiers** — guardrail/safety models. They are surfaced for
   experimentation but may emit safety labels instead of the required JSON.
3. **Other free models** — kept for fallback, ranked lower.

**OpenRouter** — the public catalogue is filtered to `:free` on every load
(cached 10 minutes). Example defaults: `google/gemma-4-31b-it:free`.

**Groq** — Groq does not expose plan eligibility in its API, so the list is the
documented **Free Plan** set, intersected with the models your key can see:
`openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `qwen/qwen3.8-27b`, plus the
`openai/gpt-oss-safeguard-20b` classifier. Paid/enterprise models (including
`llama-3.3-70b-versatile`), audio models, and prompt-guard classifiers are not
offered. The free plan is rate-limited to roughly 8K TPM / 1K RPD for the chat
models, so long documents may pause on `429` retries.

Handled failures, with actionable messages in the UI:

| Condition | Behaviour |
|---|---|
| `429` | Retries up to 3× with backoff / `Retry-After`, then explains the limit |
| `401/403` | "Key rejected" |
| `402` | "Plan or credits refused the request" |
| `404` | "Model unavailable — pick another `:free` model" |
| `5xx` | Retries twice |
| Timeout | 150 s per chunk; the UI aborts at 240 s |
| Reasoning-only reply | Explicit error instead of silent empty result |

---

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```

The suite covers fenced/wrapped/truncated JSON, synonym labels, XSS escaping,
paragraph reconstruction, chunk limits, score arithmetic, the exact system prompt,
free-model enforcement, provider routing, and upstream error mapping.

End-to-end UI check without spending credits:

```bash
# terminal 1
.venv/bin/python tests/stub_openrouter.py
# terminal 2
OPENROUTER_BASE_URL=http://127.0.0.1:8599/v1 \
GROQ_BASE_URL=http://127.0.0.1:8599/v1 \
.venv/bin/python app.py
```

---

## Customising the palette

All colours live in `:root` in `static/style.css`:

```css
--red:   #EF4444;   /* AI generated  */
--amber: #F59E0B;   /* AI refined    */
--blue:  #3B82F6;   /* human         */
--cyan:  #22D3EE;   /* glitch accent */
```

---

## Limitations

A detector score is evidence about statistical typicality, **not proof of
authorship**. False positives concentrate on second-language writers, formulaic
genres (lab reports, legal boilerplate), and heavily edited drafts. The UI states
this explicitly; keep it that way in any deployment.
