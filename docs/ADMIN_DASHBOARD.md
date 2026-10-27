# Admin dashboard — brief for the implementing agent

## Status (26 Oct 2026)

**Built:** `/admin` (`frontend/admin.html`, `static/admin.js`, `static/admin.css`)
with a token gate and four views:

- **API keys**: add (provider recognised automatically), list masked, remove.
- **Users**: search and filter, plan/status badges, today's usage, manage
  panel (grant Premium/Family for N days, change plan/status/notes, reset
  today's usage, sign out everywhere, suspend, delete), add an account.
- **Analytics**: anonymous usage (`backend/glitch/analytics.py`) plus account
  counts by plan and status (`users.store.summary()`).
- **Providers**: readiness and billing.

Accounts, sessions, plans and quotas live in `backend/glitch/users.py`; the
admin routes are listed in the README's *Admin dashboard* section.

**Still to build** (each needs an external service or a product decision):
PayChangu checkout and payment webhooks (moving `pending_payment` → `active`
automatically), renewal reminders, ads on the free tier, Family seat
invitations, and the optional key extensions in §6.

The rest of this brief is the original specification; keep following its
constraints (§3) and look-and-feel rules (§4) for further admin work.

You are building the admin dashboard for **Glitch//Detect**, a local-first
AI-writing detector and humanizer (FastAPI backend + static vanilla-JS
frontend, no build step). The
backend for key management already exists and is tested; your job is mainly
the UI, plus any small backend additions listed under *Optional extensions*.

Read this file fully before writing code. Then skim, in this order:
`backend/glitch/api.py` (routes), `backend/glitch/keys.py` (key store),
`backend/glitch/providers.py` (provider registry), `frontend/index.html` +
`frontend/static/app.js` (the user app you are matching), and
`backend/tests/test_glitch.py` (`AdminApiTests`, `KeyStoreTests`). The live
API schema is at `/docs` while the app runs.

---

## 1. Goal

An admin opens the dashboard, enters the admin token once, and can:

1. **See every API key** the server can use — masked — with its provider,
   label, source (added by admin vs. server environment), verification state,
   and date added.
2. **Add a key by pasting it.** No provider dropdown: the backend recognises
   the provider from the key format and confirms it with that provider.
3. **Remove a key** that was added through the dashboard.
4. **See which providers are ready** (have at least one key) and how each bills
   (free / free tier / paid).

The principle the product owner asked for: *"I should just add the API key and
it automatically resolves which provider it's for."* Never ask the admin to
pick a provider.

---

## 2. Backend contract (exists today)

All routes return JSON. Errors always look like:

```json
{ "ok": false, "error": { "message": "Human-readable text", "detail": "optional technical detail" } }
```

Show `error.message` to the admin; put `error.detail` behind a "Technical
details" disclosure (the user app does this — see `.alert-detail-wrap`).

### Authentication

- Admin routes exist **only when `ADMIN_TOKEN` is set** in `.env`. Without it
  they return **404** — show "Admin access isn't enabled on this server. Set
  ADMIN_TOKEN in .env and restart." rather than a generic error.
- Every admin request must send `Authorization: Bearer <token>`.
  Wrong or missing token → **401**. Tokens are compared in constant time.
- There are no sessions or cookies. The dashboard asks for the token, keeps it
  in **`sessionStorage`** (cleared when the tab closes), and offers "Sign out"
  which deletes it. Never put the token in a URL, `localStorage`, or a log.

### `GET /api/admin/keys` (admin)

```json
{
  "ok": true,
  "keys": [
    {
      "id": "3f9a1c2b7d4e",
      "provider": "gemini",
      "provider_label": "Google Gemini",
      "key": "AIza…QQQQ",
      "label": "team Gemini",
      "verified": true,
      "added_at": 1793000000,
      "source": "admin"
    },
    {
      "id": "env:groq",
      "provider": "groq",
      "provider_label": "Groq",
      "key": "gsk_…abcd",
      "label": "From server environment",
      "source": "env"
    }
  ],
  "providers": [
    { "id": "openrouter", "label": "OpenRouter", "billing": "free", "keys_url": "https://openrouter.ai/keys",
      "default_model": "google/gemma-4-31b-it:free", "ready": true }
  ]
}
```

- `key` is **always masked** (first 4 + last 4 characters). The full key never
  leaves the server after it is added. Do not try to reveal it.
- `source: "env"` entries come from `.env` and are **read-only**: no delete
  button, and explain "Set in the server's .env file".
- `env` entries have no `verified` / `added_at` fields — handle their absence.
- `added_at` is Unix seconds.
- `billing` is one of `"free"`, `"free-tier"`, `"paid"`.
- Precedence when several keys exist for one provider: newest admin-added key
  → provider env var (e.g. `GROQ_API_KEY`) → `LLM_API_KEYS`. Only the first is
  used. Consider showing an "in use" marker on that one.

### `POST /api/admin/keys` (admin)

Request: `{ "key": "<pasted key>", "label": "optional, ≤60 chars" }`

Success: **201** `{ "ok": true, "key": { …entry as above… } }`. Adding a key
that already exists returns the existing entry (also 201) — no duplicate.

Failures you must present clearly:

| Status | Meaning | Suggested UI copy |
|---|---|---|
| 400 | Not a key, or format not recognised | Show `error.message` (it lists supported providers) |
| 401 | The recognised provider rejected the key | "{Provider} rejected this key. Check it and try again." (`error.message` already says this) |
| 503 | Key format is ambiguous and the providers couldn't be reached | "Couldn't confirm the key right now. Try again in a moment." |

Verification calls the provider, so this request can take a few seconds:
disable the button and show progress. If `verified` is `false` in the result,
the provider was unreachable but the format was unambiguous — show a muted
"Not yet verified" badge, not an error.

### `DELETE /api/admin/keys/<id>` (admin)

Success: `{ "ok": true }`. Unknown id or an `env:` id → **404**.
Ask for confirmation before deleting ("Remove the Gemini key ending QQQQ?").

### Public routes you may also use

| Route | Use |
|---|---|
| `GET /api/providers` | Provider list + readiness, no auth. Same `providers` shape as above. |
| `POST /api/keys/identify` `{key}` | Recognise a key **without storing it** → `{provider, label, verified}`. Useful for a live "This looks like a Groq key" preview before saving. Note it also calls the provider to verify, so debounce it (≥600 ms) or call it only on blur/submit. |
| `POST /api/models` `{provider}` | Model list for a provider using the server key. Good for a "Test key / show models" detail view. |

Supported providers and key formats live in `backend/glitch/providers.py`
(`ProviderSpec.key_patterns`). Do not hard-code them in JS — read labels and
billing from `/api/providers`.

---

## 3. Hard constraints

These come from the existing security model. Breaking any of them is a bug.

1. **Same app, same origin.** Create `frontend/admin.html` — the backend
   already serves any `frontend/<name>.html` at `/<name>`, so it appears at
   `/admin` with no backend change.
   The server rejects requests whose `Host` isn't `127.0.0.1`/`localhost`, and
   there is no CORS — a separate dev server on another port will not work.
2. **Content-Security-Policy** (`_CSP` in `backend/glitch/api.py`) allows scripts and
   styles **only from `'self'`** (plus Google Fonts CSS). Therefore:
   no inline `<script>`, no `onclick=""` attributes, no `style=""` attributes,
   no CDN libraries. Put JS in `frontend/static/admin.js`, CSS in
   `frontend/static/style.css` (or `frontend/static/admin.css`), and reference
   them as `/static/…`. Setting `element.style.x` from JS is fine.
   API writes must be sent as `application/json` (others get 415).
3. **Never render a full key**, never log one, never store one in browser
   storage. The admin types a key once; clear the input after a successful add.
4. **Escape everything** you put into the DOM. Use `textContent` /
   `createElement`, as `frontend/static/app.js` does. Labels are admin-supplied text.
5. **Don't weaken the admin gate**: keep the 404-when-disabled and 401
   behaviour, and don't add "remember me" persistence for the token.
6. Don't change provider rules, prompts, or the free/paid fallback policy as
   part of this work.

---

## 4. Look and feel

Match the user app exactly — same dark cyber aesthetic, plain-language copy.
Reuse `frontend/static/style.css`; don't fork the design.

- **Layout:** the user app is a `.shell` grid with a left `.sidebar`
  (`.sb-brand`, `.sb-nav` of `.sb-link`, `.sb-foot` status) and `.main` holding
  `.view` sections with a `.page-head` (`.eyebrow`, `h1`, `.page-sub`). Mirror
  it. Suggested sidebar: *API keys*, *Providers*, a "Back to app" link to `/`,
  and *Sign out* at the bottom.
- **Cards:** `.panel.settings-card` with an `h2` and `.card-sub`.
- **Controls:** `.btn-primary` (main action), `.btn-ghost` (secondary,
  `.danger` for delete), `.mini-btn` (inline), `.switch` toggles,
  `.field-label`, `.hint`. Inputs are styled globally.
- **Lists:** `.key-list` (`li` with `.key-provider`, `.key-mask`, trailing
  `.mini-btn`) already exists for the user's own keys — extend it.
- **Status:** `.choice-status[data-state="ready"|"needs-key"]`, `.badge`,
  `.soon-tag`; errors use the `.alert` block (see markup in
  `frontend/index.html`; the JS helper `showAlert(box, …)` in `app.js` shows
  how it's driven).
- **Tokens** (`:root` in `style.css`): `--cyan` accent, `--red/--amber/--blue`
  plus `*-text` tints for readable coloured text, `--green` for success.
  Headings use `--font-display` (Orbitron); **numbers use the body font**
  (Orbitron renders 0 as Ø).
- **Copy:** human, short, no jargon. "Add key", "Remove", "Ready", "Not yet
  verified". Avoid ".env", "HTTP 401", provider IDs in the main UI; it's fine
  in a "Technical details" disclosure.
- **Responsive:** must work at 360 px wide (the user app turns the sidebar
  into a bottom tab bar under 860 px — follow that pattern or stack).
- **Motion:** respect `prefers-reduced-motion` (already handled globally).

---

## 5. Suggested screens

1. **Sign-in gate** — single password field "Admin token", "Sign in" button.
   On submit call `GET /api/admin/keys`; 401 → "That token didn't work";
   404 → admin disabled message (see §2). On success store token and show app.
2. **API keys** (default view)
   - "Add a key" card: one password-type input (with Show/Hide), optional
     label input, "Add key" button. On success: green confirmation
     "✓ Google Gemini key added and working", clear input, refresh list.
   - Key list grouped by provider: provider label, masked key, label, source
     badge ("Added here" / "Server environment"), verified badge, added date,
     Remove (admin keys only), "In use" marker on the active key.
   - Empty state: "No keys yet. Paste any supported key above."
3. **Providers** — every provider from `/api/providers`: name, billing badge
   (Free / Free tier / Paid), Ready / Needs a key, "Get a key ↗" link
   (`keys_url`, `target="_blank" rel="noopener noreferrer"`).

---

## 6. Optional extensions (backend work required)

Not implemented. If you build any, add the route in `backend/glitch/api.py`
with `dependencies=admin` (the `require_admin` dependency), use a pydantic
request model, raise `DetectorError` for failures so the error envelope is
kept, and add tests.

| Feature | Notes |
|---|---|
| Edit label | `PATCH /api/admin/keys/<id> {label}` → add `KeyStore.relabel()`. |
| Re-verify a key | `POST /api/admin/keys/<id>/verify` → `keys.probe(key, provider)`; update `verified`. |
| Usage stats | **Done** — `GET /api/admin/analytics` (anonymous counters) and per-account usage in `GET /api/admin/users/<id>`. |
| Key priority | Let the admin choose which key is active per provider (store an `order` field). |
| Multiple keys per provider rotation | Would need routing changes (`keys.resolve_request_keys`, `routing.run_job`); discuss before building. |

---

## 7. Working in this repo

```bash
./run.sh                                                  # serve on http://127.0.0.1:8501
.venv/bin/python -m pip install -r backend/requirements-dev.txt   # once, for tests
.venv/bin/python -m unittest discover -s backend/tests    # all tests (offline, <1 s)
node --check frontend/static/admin.js                     # JS syntax check
```

Enable the admin API locally: add `ADMIN_TOKEN=some-long-random-string` to
`.env` and restart.

Test without real keys or network — run the stub and point providers at it:

```bash
.venv/bin/python backend/tests/stub_provider.py  # terminal 1: fake provider on :8599
S=http://127.0.0.1:8599/v1
PORT=8611 OPENROUTER_BASE_URL=$S GROQ_BASE_URL=$S GEMINI_BASE_URL=$S \
  ADMIN_TOKEN=test-admin GLITCH_INSTANCE_DIR=/tmp/glitch-instance \
  .venv/bin/python backend/main.py               # terminal 2
```

With that setup any well-formed key (e.g. `AIza` + 35 letters) verifies
against the stub, and admin-added keys go to `/tmp/glitch-instance`, not the
real `instance/` folder.

Conventions:
- Python: type hints, `from __future__ import annotations`, docstrings on
  modules and non-obvious functions, comments only where intent isn't obvious.
- JS: one IIFE per page, `"use strict"`, an `els` map of elements, a `state`
  object, small named functions, no frameworks, no build step. Model it on
  `frontend/static/app.js` and reuse its patterns (`store` wrapper with try/catch,
  `postJson`, `showAlert`).
- Tests: the test module blocks all network access and swaps `keys.store` for a
  temp store — follow `AdminApiTests` for new admin tests. Mock `keys.probe`
  instead of calling providers.
- `instance/` and `.env` are git-ignored. Never commit keys or tokens, and
  don't create fixtures containing real-looking secrets.

---

## 8. Done when

- [ ] `/admin` loads, passes CSP (no console CSP errors), and matches the app's look.
- [ ] Token gate handles: admin disabled (404), wrong token (401), success.
- [ ] Token kept only in `sessionStorage`; Sign out clears it.
- [ ] Adding a key never asks for a provider; result shows the recognised provider.
- [ ] 400 / 401 / 503 add-key failures each show a clear, human message.
- [ ] Env keys are visible but can't be removed; admin keys can, with confirmation.
- [ ] No full key ever appears in the DOM, storage, URLs, or logs.
- [ ] All DOM text set via `textContent`/`createElement`.
- [ ] Works at 360 px width and with keyboard only (focus states visible).
- [ ] Existing tests pass; new backend routes (if any) have tests.
- [ ] README "Admin API" section updated if the API changed.
