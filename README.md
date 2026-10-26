# Glitch//Detect

A local-first, sentence-level AI writing detector and humanizer with a
cyberpunk interface. Text is analysed by a model from any supported provider
(OpenRouter, Groq, Gemini, OpenAI, Mistral, Cerebras, DeepSeek, xAI, Together);
the app runs on your machine and API keys never reach the browser unless the
user adds their own.

![status](https://img.shields.io/badge/node-local-22D3EE) ![python](https://img.shields.io/badge/python-3.10%2B-3B82F6) ![providers](https://img.shields.io/badge/providers-9-F59E0B)

---

## Quick start

```bash
cd glitch-detect
cp .env.example .env      # add at least one API key
./run.sh
```

`run.sh` creates a virtualenv on first run, installs the backend's requirements
(FastAPI, uvicorn, requests) whenever they change, and prints:

```
  LOCAL FORENSICS NODE  ·  multi-provider · FastAPI
  ➜  http://127.0.0.1:8501
  API docs: http://127.0.0.1:8501/docs
  ready: Groq, OpenRouter
```

Open the URL, paste text, press **CHECK TEXT**. Flagged sentences can be sent to
**Humanize** with one click.

### API keys — no provider needed

Just add the key; the provider is recognised automatically.

```
# .env
LLM_API_KEYS=gsk_…, AIza…, sk-or-v1-…
```

Provider-specific variables (`GROQ_API_KEY`, `GEMINI_API_KEY`, …) also work. In
the app, **Settings → Your own API keys** takes any supported key the same way.
See [Providers and keys](#providers-and-keys).

Manual equivalent of `run.sh`:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r backend/requirements.txt
.venv/bin/python backend/main.py
# or, with auto-reload while developing:
.venv/bin/uvicorn main:app --app-dir backend --reload --port 8501
```

---

## What it does

**Check text**
1. Splits the document into sentences (quick mode) or sections (classic mode).
2. Looks every sentence up in the cache; only new or edited sentences are sent.
3. Sends them in parallel through the shared router (balancing, fallback,
   sized token budgets).
4. Parses the reply tolerantly, scores `(AI + REFINED) / total × 100`, and
   rebuilds the original text as colour-coded `<span>`s.

| Class | Colour | Meaning |
|---|---|---|
| `AI` | `#EF4444` red | Machine-generated: high predictability, flat rhythm |
| `REFINED` | `#F59E0B` amber | Paraphrased / polished over an AI base |
| `HUMAN` | `#3B82F6` blue | Organic variance, idiom, imperfection |

**Humanize** — designed from `mechanics-of-ai-writing-detection.md`. Detectors
weigh rhythm and discourse shape (T2) and missing particulars (T3) far more than
individual words (T1), and sentence-by-sentence paraphrase makes text *more*
uniform (§4.4.3). So the humanizer:

1. Uses the labels from the last check (or labels the text in quick mode).
2. Groups each run of consecutive flagged sentences into a **passage** and
   rewrites it as a unit, free to merge, split and cut, which is the only way to
   change sentence-length rhythm.
3. Gives each passage **specific notes** from `backend/glitch/style.py` (zero tokens):
   flat rhythm, LLM-lexicon words, tricolons, stock openers and closers,
   trailing participial clauses, uniform hedging.
4. **Validates locally** (zero tokens): lost numbers/names → retry, then keep
   the original; leftover AI patterns, flat rhythm (CV < 0.5) or, in
   *personal* voice, no voice markers → one targeted retry.
5. Re-scores only the new sentences, and by default (*Extra polish*) rewrites
   passages the detector still flags once more, showing the model its last
   attempt.
6. Returns before/after **sentence variety** and **AI-phrase counts**, plus
   **grounding tips**: paragraphs that only make general claims. No facts are
   invented; specific detail is the writer's to add.

Voices: *personal* (default: first person, takes a position, may add
opinions; the UI asks the user to check them) and *faithful* (meaning and
perspective kept). Tones: natural, casual, academic, professional. Rewrites
are cached per passage + tone + voice + model; *Try another version* skips the
cache.

Measured on a live model (Groq `gpt-oss-120b`) with a fully generic AI
paragraph: the sentence-level rewriter left it at 100% in every run; the
passage-level design gave 18%, 50% and 67% over three runs (median 50%),
using about 7K tokens with Extra polish (about 3.5K without). The same claims
with real, specific details scored 0%, which is why the UI points users to the
grounding tips.

Both features are limited to **2,500 words** per request (`MAX_WORDS` in
`backend/glitch/config.py`); the UI warns live and the server enforces it.

---

## Architecture

```
glitch-detect/
├── backend/                 FastAPI app (Python)
│   ├── main.py              launcher: banner, free port, uvicorn
│   ├── glitch/              the application package
│   ├── tests/               unit tests + provider stub
│   └── requirements*.txt
├── frontend/                static site, no build step, no templating
│   ├── index.html
│   └── static/              style.css, app.js
├── docs/                    ADMIN_DASHBOARD.md, screenshot
└── run.sh  .env  instance/  (instance/ = admin-added keys, git-ignored)
```

The backend serves the frontend from the same origin (`/`, `/static/…`, and
any `frontend/<name>.html` at `/<name>`), so the local-only host check and the
strict CSP keep working and no CORS is needed. Point `GLITCH_FRONTEND_DIR` at
another folder to serve a different build. The frontend gets its limits,
defaults and tone list from `GET /api/config`.

```
browser (frontend/static/app.js)
   │  /api/config · /api/analyze · /api/humanize · /api/models · /api/keys/identify
   ▼
glitch/api.py ─ FastAPI: host guard, JSON-only writes, request models,
   │            error envelope, CSP, admin API, frontend serving
   ├─ analyzer.py ─┐   (detection: jobs + caches)
   ├─ humanize.py ─┤   (rewrite: passages + style.py notes/validation)
   │               ▼
   │          routing.py ─ jobs → parallel runner → ModelRouter
   │               │       (balancing · fallback · cooldowns · budgets)
   │               ▼
   │          client.py ─ pooled HTTP, retries, error mapping
   │
   └─ keys.py ─ recognise key → provider · KeyStore (.env + instance/keys.json)
                     │
                providers.py ─ data-only registry of OpenAI-compatible providers
```

Endpoints are plain `def` functions (model calls use blocking HTTP); FastAPI
runs them in its worker thread pool.

| File | Role |
|---|---|
| `backend/main.py` | Launcher: banner, free-port search, `uvicorn.run` |
| `backend/glitch/api.py` | FastAPI app: routes, request models, middleware, frontend serving |
| `backend/glitch/config.py` | `.env` loading, limits, paths, admin token |
| `backend/glitch/providers.py` | Provider registry: URLs, key patterns, billing, model rules |
| `backend/glitch/keys.py` | Key recognition, verification, server key store |
| `backend/glitch/catalogue.py` | Model listing and ranking, fallback order, reasoning options |
| `backend/glitch/routing.py` | Shared job runner: balancing, fallback, cooldowns |
| `backend/glitch/client.py` | Pooled HTTP session, retries, upstream error mapping |
| `backend/glitch/cache.py` | Thread-safe TTL/LRU cache keyed by SHA-256 fingerprints |
| `backend/glitch/prompt.py` | Detection prompts (quick + verbatim classic) and rewrite prompt |
| `backend/glitch/parsing.py` | Tolerant parsers for labels, sentences, and rewrites |
| `backend/glitch/text.py` | Sentence splitting, chunking, highlight HTML, span replacement |
| `backend/glitch/analyzer.py` | Detection: jobs, sentence/section caches, scoring |
| `backend/glitch/humanize.py` | Rewrite: passages, notes, validation, re-scoring, second pass |
| `backend/glitch/style.py` | Zero-token style analysis: rhythm, AI phrases, voice, grounding tips |
| `frontend/index.html` | Sidebar app: Check text, Humanize, Settings, Account |
| `frontend/static/style.css` | Design tokens, highlight classes, layout |
| `frontend/static/app.js` | Routing, settings/keys, checks, rewrites, plain-language errors |
| `backend/tests/test_glitch.py` | Unit + API tests (offline: network is blocked in tests) |
| `backend/tests/stub_provider.py` | FastAPI provider double for E2E checks (labels, rewrites, 429s) |

### Parsing ladder

Free models are told to return raw JSON but often don't. Every parser degrades
the same way:

1. `json.loads` on the outermost `{…}` (after stripping ```` ```json ```` fences).
2. Trailing-comma repair, then parse again.
3. Structural regex — survives truncation and prose wrappers.
4. Fail with a diagnostic containing the first 280 characters of the response
   (the router then tries the next model).

Labels are normalised (`paraphrased`, `machine-generated`, `polished` → the three
canonical classes). Unknown labels default to `HUMAN` with a warning.

---

## Providers and keys

Every provider speaks the OpenAI-compatible chat API, so each one is a data
entry in `backend/glitch/providers.py`:

| Provider | Key format | Billing | Models allowed |
|---|---|---|---|
| OpenRouter | `sk-or-…` | free | `:free` models only |
| Groq | `gsk_…` | free | documented Free Plan list |
| Google Gemini | `AIza…` | free tier | anything its `/models` lists |
| Cerebras | `csk-…` | free tier | 〃 |
| Mistral | 32 letters/digits | free tier | 〃 |
| OpenAI | `sk-proj-…`, `sk-…` | paid | 〃 |
| DeepSeek | `sk-` + 32 hex | paid | 〃 |
| xAI | `xai-…` | paid | 〃 |
| Together AI | `tgp_v1_…`, 64 hex | paid | 〃 |

**How a key is resolved** (`backend/glitch/keys.py`):

1. *Recognise* — match the key against each provider's `key_patterns`.
2. *Confirm* — call that provider's `/models` with the key. A key is only ever
   sent to providers whose pattern it matches, never to the others.
3. If the provider can't be reached and the format is unambiguous, the key is
   accepted unverified.

**Server keys**, highest priority first: keys added at runtime (admin API,
stored in `instance/keys.json` with `0600` permissions, git-ignored) →
provider variables (`GROQ_API_KEY`, …) → `LLM_API_KEYS`. A user's own key from
Settings overrides the server key for that user only.

For generic providers the model list comes from the provider and is ranked for
this task (fast instruction models first; embedding/audio/image/moderation
models removed). Fallback **never moves work onto a paid provider** the user
didn't select.

**Adding a provider**: append a `ProviderSpec` with its base URL, key regexes,
env var names, billing, and (optionally) curated models and reasoning-model
patterns. Nothing else changes.

### Admin API (for a future dashboard)

Disabled unless `ADMIN_TOKEN` is set; every call needs
`Authorization: Bearer $ADMIN_TOKEN`. Building the dashboard? Start with
[docs/ADMIN_DASHBOARD.md](docs/ADMIN_DASHBOARD.md).

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/admin/keys` | All keys (masked) with source, plus provider readiness |
| `POST` | `/api/admin/keys` | `{key, label?}` → recognised, verified, stored; `201` |
| `DELETE` | `/api/admin/keys/<id>` | Remove a stored key (env keys are read-only) |

```bash
curl -s -X POST http://127.0.0.1:8501/api/admin/keys \
  -H "Authorization: Bearer $ADMIN_TOKEN" -H 'Content-Type: application/json' \
  -d '{"key":"AIza…","label":"team Gemini"}'
# → {"ok":true,"key":{"provider":"gemini","key":"AIza…abcd","verified":true,…}}
```

---

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` · `/<page>` | Frontend (`frontend/index.html`, `frontend/<page>.html`) |
| `GET` | `/docs` · `/openapi.json` | Interactive API docs and schema |
| `GET` | `/api/config` | Limits, defaults, tones and voices for the frontend |
| `GET` | `/api/health` · `/api/providers` | Providers and whether each is ready (never keys) |
| `POST` | `/api/keys/identify` | `{key}` → `{provider, label, verified}`; nothing stored |
| `POST` | `/api/models` | `{provider, api_keys?, force?}` → ranked model list |
| `POST` | `/api/analyze` | `{text, provider, model, api_keys?, …options}` → score, sentences, HTML |
| `POST` | `/api/humanize` | `{text, provider, model, api_keys?, tone?, voice?, only_flagged?, second_pass?, fresh?, labels?, …options}` → rewritten text, HTML, changes, before/after score and style, grounding tips |

Options shared by analyze and humanize: `chunk_words`, `concurrency`,
`fallback`, `cross_provider`, `compact`, `balance`. `api_keys` maps provider id
to the user's own key; omitted providers use the server key. Results include
`usage` (tokens), `models_used`, and `fallbacks`.

```bash
curl -s http://127.0.0.1:8501/api/humanize -H 'Content-Type: application/json' \
  -d '{"provider":"groq","model":"openai/gpt-oss-120b","tone":"casual","text":"…"}'
```

---

## Security & privacy

- Server keys (`.env`, admin store) are **never sent to the browser**; the UI
  only learns whether a provider is ready. Admin listings show masked keys.
- A pasted key is only sent to providers whose key format it matches.
- User keys from Settings live in the tab's memory (optionally
  `sessionStorage`) and are sent to the local node with each request.
- The admin API is off unless `ADMIN_TOKEN` is set, and compares tokens in
  constant time.
- The server binds to `127.0.0.1` only and rejects requests whose `Host` is not
  `127.0.0.1`/`localhost` (blocks DNS-rebinding pages from spending server keys).
  It sets `Cache-Control: no-store`, `X-Frame-Options: DENY`, and a strict CSP.
- Model output is HTML-escaped before it is wrapped in spans.
- Caches hold SHA-256 fingerprints of text; client-supplied labels are never
  written into the shared cache.
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
| **Edit-aware cache** | In quick mode every sentence's label is cached (in memory, 6 h) under the selected service + model. A re-check sends only new or edited sentences, duplicates once; an unchanged document makes no calls at all. Labels are saved as each group finishes, so a scan interrupted by limits resumes where it stopped. Classic mode caches whole sections. Only SHA-256 fingerprints of text are kept. | Automatic |
| **Load balancing** | Sections of a long document are spread round-robin over the provider's top free models (Groq: 3, OpenRouter: 2), multiplying per-model rate-limit budgets. | Advanced → Share long documents across models (default on) |

Each result reports `mode`, `reused_sentences`, `cached_chunks`, `usage` (prompt / completion /
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
.venv/bin/python -m pip install -r backend/requirements-dev.txt   # once (httpx for TestClient)
.venv/bin/python -m unittest discover -s backend/tests -v
```

The suite covers fenced/wrapped/truncated JSON, synonym labels, XSS escaping,
paragraph reconstruction, chunk limits, parallel ordering, score arithmetic, the
exact system prompt, free-model enforcement, model fallback and cooldowns,
`.env`/key resolution, the admin API, the humanizer, the local-only guard,
JSON-only writes, the error envelope, page serving, and upstream error mapping.

End-to-end UI check without spending credits:

```bash
# terminal 1 (optionally make a model answer 429 to exercise fallback)
STUB_RATE_LIMITED=google/gemma-4-31b-it:free .venv/bin/python backend/tests/stub_provider.py
# terminal 2
OPENROUTER_BASE_URL=http://127.0.0.1:8599/v1 \
GROQ_BASE_URL=http://127.0.0.1:8599/v1 \
OPENROUTER_API_KEY=sk-or-stub-key-0000 \
.venv/bin/python backend/main.py
```

---

## Customising the palette

All colours live in `:root` in `frontend/static/style.css`:

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

---

## Business model

See [BUSINESS_MODEL.md](BUSINESS_MODEL.md) for the commercial model: Free and
Premium limits, MWK pricing, PayChangu payments (Airtel Money, TNM Mpamba,
bank/card), family plans, renewal notifications, unit economics and projections.
