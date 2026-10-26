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

Open that URL, choose **OPENROUTER** or **GROQ**, paste text, and hit
**RUN FORENSIC SCAN**. Only free-tier models are listed and accepted.

### API keys

Put default keys in `.env` (see `.env.example`):

```
OPENROUTER_API_KEY=sk-or-v1-…
GROQ_API_KEY=gsk_…
```

They stay on the server; the browser only learns that a default exists. Keys
entered on the **Settings** page override the `.env` defaults per provider
([OpenRouter keys](https://openrouter.ai/keys) · [Groq keys](https://console.groq.com/keys)).

Manual equivalent:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python app.py
```

Environment variables (or `.env`): `OPENROUTER_API_KEY`, `GROQ_API_KEY`,
`PORT` (default `8501`), `OPENROUTER_BASE_URL` (default
`https://openrouter.ai/api/v1`), `GROQ_BASE_URL` (default
`https://api.groq.com/openai/v1`).

---

## What it does

1. Splits the document into API-sized chunks on paragraph boundaries.
2. Sends the chunks **in parallel** to the selected provider's OpenAI-compatible
   chat-completions endpoint with the mandated forensic system prompt (verbatim,
   see `glitch/prompt.py`). If a model is rate limited, gone, or failing, the
   scan switches to the next free model immediately (see *Rate limits*).
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
   │  POST /api/analyze  { text, provider, model, api_keys?, chunk_words,
   │                       concurrency, fallback, cross_provider }
   ▼
glitch/web.py ── local-only guard, validation, key resolution (.env / user), CSP
   │
   ▼
glitch/analyzer.py ── chunk → parallel workers ─┬─ ModelRouter (fallback + cooldowns)
   │                                            └─ client.py → parsing.py
   ▼
score → text.py highlight HTML
   │
   ├── https://openrouter.ai/api/v1/chat/completions
   └── https://api.groq.com/openai/v1/chat/completions
```

| File | Role |
|---|---|
| `app.py` | Launcher: banner, free-port search, `app.run` |
| `glitch/config.py` | `.env` loading, endpoints, limits, key resolution |
| `glitch/providers.py` | Provider registry, free-model lists, model/key validation |
| `glitch/catalogue.py` | Free-model listing, ranking, fallback order |
| `glitch/client.py` | Pooled HTTP session, retries, upstream error mapping |
| `glitch/prompt.py` | The verbatim system prompt and per-chunk message |
| `glitch/parsing.py` | Tolerant JSON extraction, label normalisation |
| `glitch/text.py` | Sentence splitting, chunking, highlight HTML |
| `glitch/analyzer.py` | Parallel scan, model fallback and cooldowns, scoring |
| `glitch/humanize.py` | Placeholder for the upcoming rewrite feature |
| `glitch/web.py` | Flask app factory and routes |
| `templates/index.html` | Sidebar app: Check text, Humanize (preview), Settings, Account |
| `static/style.css` | Design tokens, highlight classes, layout |
| `static/app.js` | Page routing, settings/keys, checks, plain-language results and errors, report download |
| `tests/test_glitch.py` | Unit tests: parsing, fallback, keys, web guard, error mapping |
| `tests/stub_provider.py` | Local provider double for the E2E check (can simulate 429s) |

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
| `POST` | `/api/models` | `{provider, api_keys?, force?}` → ranked free-model list |
| `POST` | `/api/analyze` | `{text, provider, model, api_keys?, chunk_words?, concurrency?, fallback?, cross_provider?, compact?, balance?}` → score + sentences + HTML |
| `POST` | `/api/humanize` | Placeholder — returns `501` |

`api_keys` is `{"openrouter": "…", "groq": "…"}`; omitted providers use the
`.env` key. The legacy single `api_key` field is still accepted. Results include
`model`/`provider` (the model that did most of the work), `requested_model`,
`models_used`, and `fallbacks`.

Model listing (OpenRouter needs no key; Groq needs the key to verify availability):

```bash
curl -s http://127.0.0.1:8501/api/models \
  -H 'Content-Type: application/json' \
  -d '{"provider":"groq"}'
```

Analysis:

```bash
curl -s http://127.0.0.1:8501/api/analyze \
  -H 'Content-Type: application/json' \
  -d '{"provider":"openrouter","model":"google/gemma-4-31b-it:free","text":"Your text here. Second sentence."}'
```

Non-free models are rejected **before** any network call: OpenRouter IDs must end
in `:free`, and Groq IDs must be on the documented Free Plan list.

---

## Security & privacy

- `.env` keys are read by the server and **never sent to the browser**; the UI
  only receives an "env key available" flag.
- Custom keys from Settings live in the tab's memory and are sent to the local
  node with each request. Optional "keep until this tab closes" uses
  `sessionStorage`. Non-secret settings use `localStorage`.
- The server binds to `127.0.0.1` only and rejects requests whose `Host` is not
  `127.0.0.1`/`localhost` (blocks DNS-rebinding pages from spending `.env` keys).
  It sets `Cache-Control: no-store`, `X-Frame-Options: DENY`, and a strict CSP.
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

### Saving tokens

| Technique | What it does | Setting |
|---|---|---|
| **Quick (compact) mode** | Sentences are split locally and sent numbered; the model replies with labels only (`{"labels": [...]}`) instead of echoing the text. About 93% less output and about 50% fewer total tokens. | Advanced → Quick mode (default on). Off = the original verbatim prompt. |
| **Sized `max_tokens`** | Each request asks only for what its reply needs (plus headroom for reasoning models), instead of a flat 4096. Groq counts the requested budget against tokens-per-minute. A cut-off reply is retried once with a larger budget. | Automatic |
| **Low reasoning effort** | `reasoning_effort: "low"` for gpt-oss on Groq; `reasoning.effort: "low"` for OpenRouter models that advertise reasoning. Dropped automatically if a provider rejects it. | Automatic |
| **Result cache** | Each section's result is cached in memory (1 h, 512 entries) by text, model, mode, and prompt version. Re-checking unchanged text costs nothing. | Automatic |
| **Load balancing** | Sections of a long document are spread round-robin over the provider's top free models (Groq: 3, OpenRouter: 2), multiplying per-model rate-limit budgets. | Advanced → Share long documents across models (default on) |

Each result reports `mode`, `cached_chunks`, `usage` (prompt / completion /
total tokens), and `models_used`; the server log prints them per scan.

### Rate limits and fallback

With **auto-switch** on (default, Settings → Rate limits):

- A `429`, `404`, repeated `5xx`, timeout, or empty/reasoning-only reply retires
  that model for the rest of the scan; every chunk moves to the next candidate
  **without sleeping**. A chunk the model can't format as JSON is retried on the
  next model too.
- Candidates: the selected model, then the provider's recommended free models
  (never guardrail classifiers), then — if allowed and a key exists — the top
  models of the other provider.
- Rate-limited models enter a cooldown (`Retry-After`, else 60 s) shared across
  scans, so the next scan starts on a model that is likely to answer.
- Only the last remaining model waits on `Retry-After` (capped at 15 s, 2 retries).
- Each switch is reported in the result warnings and `fallbacks`.

Other failures:

| Condition | Behaviour |
|---|---|
| `401/403` on the selected provider | "Key rejected" — stops the scan |
| `402` | "Plan or credits refused the request" |
| Whole scan | 225 s server budget; the UI aborts at 240 s |

---

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```

The suite covers fenced/wrapped/truncated JSON, synonym labels, XSS escaping,
paragraph reconstruction, chunk limits, parallel ordering, score arithmetic, the
exact system prompt, free-model enforcement, model fallback and cooldowns,
`.env`/key resolution, the local-only guard, and upstream error mapping.

End-to-end UI check without spending credits:

```bash
# terminal 1 (optionally make a model answer 429 to exercise fallback)
STUB_RATE_LIMITED=google/gemma-4-31b-it:free .venv/bin/python tests/stub_provider.py
# terminal 2
OPENROUTER_BASE_URL=http://127.0.0.1:8599/v1 \
GROQ_BASE_URL=http://127.0.0.1:8599/v1 \
OPENROUTER_API_KEY=sk-or-stub-key-0000 \
.venv/bin/python app.py
```

---

## Customising the palette

All colours live in `:root` in `static/style.css`:

```css
--red:   #EF4444;  --red-text:   #FCA5A5;  /* AI generated  */
--amber: #F59E0B;  --amber-text: #FCD34D;  /* AI refined    */
--blue:  #3B82F6;  --blue-text:  #93C5FD;  /* human         */
--cyan:  #22D3EE;                          /* accent        */
```

The base colours mark borders and underlines; the `*-text` tints are used for
highlighted text so it stays readable on the dark background.

---

## Limitations

A detector score is evidence about statistical typicality, **not proof of
authorship**. False positives concentrate on second-language writers, formulaic
genres (lab reports, legal boilerplate), and heavily edited drafts. The UI states
this explicitly; keep it that way in any deployment.
