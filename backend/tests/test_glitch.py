import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from glitch import analyzer, cache, catalogue, client, config, create_app, keys, routing  # noqa: E402
from glitch.providers import PROVIDERS  # noqa: E402
from glitch import humanize as humanize_mod  # noqa: E402
from glitch import style  # noqa: E402
from glitch.errors import DetectorError, ModelFailure, ParseError, RateLimited  # noqa: E402
from glitch.parsing import content_to_text, parse_analysis, parse_labels  # noqa: E402
from glitch.prompt import COMPACT_SYSTEM_PROMPT, SYSTEM_PROMPT, build_messages  # noqa: E402
from glitch.providers import validate_model  # noqa: E402
from glitch.text import Sentence, build_highlight_html, chunk_text  # noqa: E402

GOOD = {
    "sentences": [
        {"text": "The system is robust and scalable.", "classification": "AI"},
        {"text": "I rewrote it twice before it worked.", "classification": "HUMAN"},
        {"text": "Ultimately, the results underscore its importance.", "classification": "REFINED"},
    ]
}
GOOD_TEXT = (
    "The system is robust and scalable. "
    "I rewrote it twice before it worked. "
    "Ultimately, the results underscore its importance."
)
OR_KEY = "sk-or-test-key-1234"
GROQ_KEY = "gsk-test-key-1234"

def _offline(*args, **kwargs):
    raise ConnectionError("network access is disabled in tests")


# Never let the developer's real keys (.env or instance/keys.json), or any
# request, leave a test. Tests patch these again where they need a response.
_TMP = tempfile.TemporaryDirectory()
_ENV_NAMES = [name for spec in PROVIDERS.values() for name in spec.env_vars] + [keys.GENERIC_ENV_VAR]
_env_patches = [
    mock.patch.dict(os.environ, {name: "" for name in _ENV_NAMES}),
    mock.patch.object(keys, "store", keys.KeyStore(Path(_TMP.name) / "keys.json")),
    mock.patch.object(client._session, "get", side_effect=_offline),
    mock.patch.object(client._session, "post", side_effect=_offline),
]


def setUpModule():
    for patch in _env_patches:
        patch.start()


def tearDownModule():
    for patch in reversed(_env_patches):
        patch.stop()


def make_client() -> TestClient:
    """A TestClient that looks local, so the real host guard stays active."""
    return TestClient(create_app(), base_url="http://127.0.0.1:8501")


def reset_state():
    routing.clear_cooldowns()
    cache.clear_all()
    for entry in catalogue._MODEL_CACHE.values():
        entry.update(at=0.0, payload=None)


def fake_response(status, payload, headers=None):
    class R:
        status_code = status
        text = json.dumps(payload)

        def __init__(self):
            self.headers = headers or {}

        @staticmethod
        def json():
            return payload

    return R()


def chat_ok(content=json.dumps(GOOD)):
    return fake_response(200, {"choices": [{"message": {"content": content}}]})


class ParsingTests(unittest.TestCase):
    def test_plain_json(self):
        sentences, warnings = parse_analysis(json.dumps(GOOD))
        self.assertEqual([s.classification for s in sentences], ["AI", "HUMAN", "REFINED"])
        self.assertEqual(warnings, [])

    def test_fenced_json(self):
        sentences, _ = parse_analysis("```json\n" + json.dumps(GOOD) + "\n```")
        self.assertEqual(len(sentences), 3)

    def test_prose_wrapper(self):
        raw = "Sure! Here is the analysis:\n" + json.dumps(GOOD) + "\nLet me know if you need more."
        sentences, _ = parse_analysis(raw)
        self.assertEqual(len(sentences), 3)

    def test_trailing_commas(self):
        raw = '{"sentences": [{"text": "One.", "classification": "AI"},]}'
        sentences, warnings = parse_analysis(raw)
        self.assertEqual(sentences[0].classification, "AI")
        self.assertTrue(any("Repaired" in w for w in warnings))

    def test_regex_fallback_reverse_order(self):
        # Truncated JSON cannot be repaired, so the structural regex recovers it.
        raw = '{"sentences": [{"classification": "HUMAN", "text": "I walked home at dusk."}'
        sentences, warnings = parse_analysis(raw)
        self.assertEqual(sentences[0].text, "I walked home at dusk.")
        self.assertEqual(sentences[0].classification, "HUMAN")
        self.assertTrue(any("fallback regex" in w for w in warnings))

    def test_synonyms(self):
        raw = json.dumps(
            {
                "sentences": [
                    {"text": "A.", "classification": "machine-generated"},
                    {"text": "B.", "classification": "paraphrased"},
                    {"text": "C.", "classification": "authentic"},
                ]
            }
        )
        sentences, _ = parse_analysis(raw)
        self.assertEqual([s.classification for s in sentences], ["AI", "REFINED", "HUMAN"])

    def test_unknown_label_defaults_human(self):
        raw = json.dumps({"sentences": [{"text": "A.", "classification": "???"}]})
        sentences, warnings = parse_analysis(raw)
        self.assertEqual(sentences[0].classification, "HUMAN")
        self.assertTrue(any("unrecognised" in w for w in warnings))

    def test_empty_response_raises(self):
        with self.assertRaises(ParseError):
            parse_analysis("")

    def test_no_sentences_raises(self):
        with self.assertRaises(ParseError):
            parse_analysis("I cannot help with that.")

    def test_content_parts(self):
        self.assertEqual(content_to_text([{"type": "text", "text": "a"}, {"text": "b"}]), "ab")


class ChunkTests(unittest.TestCase):
    def test_short_text_single_chunk(self):
        self.assertEqual(chunk_text("One two three."), ["One two three."])

    def test_chunks_respect_limit(self):
        text = "\n\n".join(["word " * 60 for _ in range(10)])
        chunks = chunk_text(text, max_words=120)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk.split()), 130)

    def test_long_paragraph_splits_on_sentences(self):
        paragraph = " ".join(["This is a sentence about systems."] * 60)
        self.assertGreater(len(chunk_text(paragraph, max_words=100)), 1)


class HtmlTests(unittest.TestCase):
    def test_escapes_untrusted_text(self):
        sentences = [
            Sentence('<script>alert("x")</script>', "AI"),
            Sentence("Normal sentence.", "HUMAN"),
        ]
        html = build_highlight_html(sentences, "<script>alert('x')</script> Normal sentence.")
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)
        self.assertIn('class="hl ai"', html)
        self.assertIn('class="hl human"', html)

    def test_paragraph_reconstruction(self):
        sentences = [
            Sentence("First para one.", "HUMAN"),
            Sentence("First para two.", "AI"),
            Sentence("Second para.", "REFINED"),
        ]
        source = "First para one. First para two.\n\nSecond para."
        self.assertEqual(build_highlight_html(sentences, source).count("doc-para"), 2)

    def test_source_wording_is_shown(self):
        html = build_highlight_html([Sentence("the CAT sat.", "AI")], "The cat\nsat.")
        self.assertIn(">The cat sat.</span>", html)


class PromptTests(unittest.TestCase):
    def test_prompt_is_verbatim(self):
        self.assertTrue(SYSTEM_PROMPT.startswith(
            "You are a highly advanced computational linguistics model specialized in deep text forensics"
        ))
        self.assertIn("1. \"AI\" - Clearly machine-generated text", SYSTEM_PROMPT)
        self.assertIn("2. \"REFINED\" - Text that shows heavy AI optimization", SYSTEM_PROMPT)
        self.assertIn("3. \"HUMAN\" - Authentic human text", SYSTEM_PROMPT)
        self.assertIn("You must respond ONLY with a raw, valid JSON object.", SYSTEM_PROMPT)
        self.assertNotIn("```json\n{", SYSTEM_PROMPT.split("format must look exactly like this:")[0])

    def _capture(self, **kwargs):
        captured = {}

        def fake_post(url, headers=None, json=None, timeout=None):
            captured.update(url=url, headers=headers, json=json)
            return chat_ok()

        with mock.patch.object(client._session, "post", side_effect=fake_post):
            out = client.request_chunk(build_messages("Some text.", 1, 1), **kwargs)
        return captured, out

    def test_payload_shape(self):
        captured, out = self._capture(api_key=OR_KEY, model="test/model:free")
        self.assertEqual(captured["url"], "https://openrouter.ai/api/v1/chat/completions")
        self.assertEqual(captured["json"]["messages"][0]["content"], SYSTEM_PROMPT)
        self.assertEqual(captured["json"]["messages"][1]["role"], "user")
        self.assertEqual(captured["json"]["temperature"], 0.0)
        self.assertIn(f"Bearer {OR_KEY}", captured["headers"]["Authorization"])
        self.assertIn("HTTP-Referer", captured["headers"])
        self.assertEqual(out, json.dumps(GOOD))

    def test_groq_payload_shape(self):
        captured, _ = self._capture(api_key=GROQ_KEY, model="openai/gpt-oss-120b", provider="groq")
        self.assertEqual(captured["url"], "https://api.groq.com/openai/v1/chat/completions")
        self.assertNotIn("HTTP-Referer", captured["headers"])
        self.assertEqual(captured["json"]["messages"][0]["content"], SYSTEM_PROMPT)


class ScoreTests(unittest.TestCase):
    def setUp(self):
        reset_state()

    def test_score_math_and_html(self):
        with mock.patch.object(client, "request_chunk", return_value=json.dumps(GOOD)):
            result = analyzer.analyze(
                GOOD_TEXT, model="test/model:free", keys={"openrouter": OR_KEY},
                options=analyzer.ScanOptions(fallback=False),
            )
        self.assertEqual(result["counts"], {"AI": 1, "REFINED": 1, "HUMAN": 1})
        self.assertEqual(result["score"], 67)  # (1 + 1) / 3 -> 66.67 rounds to 67
        self.assertEqual(result["chunks"], 1)
        self.assertEqual(result["fallbacks"], [])
        self.assertIn("hl ai", result["html"])

    def test_validation(self):
        keys = {"openrouter": OR_KEY, "groq": GROQ_KEY}
        cases = [
            dict(text="", keys=keys),
            dict(text="Long enough.", keys={"openrouter": "short"}),
            dict(text="Long enough.", keys=keys, model="bad-model"),
            # Paid models are refused even if a caller sends the ID.
            dict(text="Long enough.", keys=keys, model="google/gemma-4-31b-it"),
            dict(text="Long enough.", keys=keys, model="llama-3.3-70b-versatile", provider="groq"),
        ]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(DetectorError):
                analyzer.analyze(case.pop("text"), **case)

    def test_parallel_chunks_keep_order(self):
        paragraphs = [f"Paragraph {i} " + "word " * 130 + "end." for i in range(4)]

        def fake(messages, api_key, model, provider, **kwargs):
            body = messages[1]["content"]
            tag = body.split("===DATA START===\n")[1].split(" ")[1]
            return json.dumps({"sentences": [{"text": f"P{tag}.", "classification": "AI"}]})

        with mock.patch.object(client, "request_chunk", side_effect=fake):
            result = analyzer.analyze(
                "\n\n".join(paragraphs), model="test/model:free", keys={"openrouter": OR_KEY},
                options=analyzer.ScanOptions(
                    chunk_words=150, fallback=False, balance=False, compact=False, concurrency=4
                ),
            )
        self.assertEqual(result["chunks"], 4)
        self.assertEqual([s["text"] for s in result["sentences"]], ["P0.", "P1.", "P2.", "P3."])


class FallbackTests(unittest.TestCase):
    def setUp(self):
        reset_state()
        catalogue._MODEL_CACHE["openrouter"].update(
            at=1e12,
            payload={"models": [
                catalogue._entry("a/first:free", "", 0, "recommended", ""),
                catalogue._entry("b/second:free", "", 0, "recommended", ""),
                catalogue._entry("c/guard:free", "", 0, "specialized", ""),
            ]},
        )

    def run_scan(self, behaviour, **kwargs):
        calls = []

        def fake(messages, api_key, model, provider, **kw):
            calls.append((provider, model, kw.get("rate_limit_retries")))
            outcome = behaviour(provider, model)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        kwargs.setdefault("keys", {"openrouter": OR_KEY})
        with mock.patch.object(client, "request_chunk", side_effect=fake):
            result = analyzer.analyze(GOOD_TEXT, model="a/first:free", **kwargs)
        return result, calls

    def test_rate_limit_switches_model_without_waiting(self):
        def behaviour(provider, model):
            return RateLimited("slow down", retry_after=30) if model == "a/first:free" else json.dumps(GOOD)

        result, calls = self.run_scan(behaviour)
        self.assertEqual(calls[0], ("openrouter", "a/first:free", 0))  # no sleep-retry
        self.assertEqual(result["model"], "b/second:free")
        self.assertEqual(result["requested_model"], "a/first:free")
        self.assertEqual(result["fallbacks"][0]["reason"], "rate limited")
        self.assertTrue(any("Switched" in w for w in result["warnings"]))
        self.assertGreater(routing.cooldown_remaining("openrouter", "a/first:free"), 0)
        self.assertNotIn("c/guard:free", [m for _, m, _ in calls])

    def test_cooldown_skips_model_on_next_scan(self):
        routing.start_cooldown("openrouter", "a/first:free", 60)
        result, calls = self.run_scan(lambda p, m: json.dumps(GOOD))
        self.assertEqual(calls[0][1], "b/second:free")
        self.assertTrue(any("cooling down" in w for w in result["warnings"]))

    def test_cross_provider_fallback(self):
        def behaviour(provider, model):
            return RateLimited("busy") if provider == "openrouter" else json.dumps(GOOD)

        result, _ = self.run_scan(behaviour, keys={"openrouter": OR_KEY, "groq": GROQ_KEY})
        self.assertEqual(result["provider"], "groq")
        self.assertEqual(result["model"], "openai/gpt-oss-120b")

    def test_last_model_retries_then_fails(self):
        calls = []

        def fake(messages, api_key, model, provider, **kw):
            calls.append((model, kw["rate_limit_retries"]))
            raise RateLimited("busy")

        with mock.patch.object(client, "request_chunk", side_effect=fake), \
             self.assertRaises(RateLimited):
            analyzer.analyze(GOOD_TEXT, model="a/first:free", keys={"openrouter": OR_KEY})
        # Every model but the last fails fast; the last one gets backoff retries.
        self.assertEqual(calls, [("a/first:free", 0), ("b/second:free", 2)])

    def test_fallback_disabled(self):
        with self.assertRaises(ModelFailure):
            self.run_scan(
                lambda p, m: ModelFailure("gone", status=404),
                options=analyzer.ScanOptions(fallback=False),
            )

    def test_primary_key_error_is_not_masked(self):
        with self.assertRaises(DetectorError) as ctx:
            self.run_scan(lambda p, m: DetectorError("bad key", status=401))
        self.assertEqual(ctx.exception.status, 401)


class ProviderTests(unittest.TestCase):
    def setUp(self):
        reset_state()

    def test_openrouter_catalogue_keeps_only_free_and_ranks_task_fit(self):
        items = [
            {"id": "vendor/paid-model", "supported_parameters": ["response_format"],
             "architecture": {"modality": "text->text"}, "description": "A paid model."},
            {"id": "vendor/plain:free", "context_length": 8000, "supported_parameters": [],
             "architecture": {"modality": "text->text"}, "description": "General chat model."},
            {"id": "vendor/json:free", "context_length": 32000,
             "supported_parameters": ["response_format", "structured_outputs"],
             "architecture": {"modality": "text->text"},
             "description": "Instruction-tuned model for extraction."},
            {"id": "vendor/guard:free", "context_length": 8000, "supported_parameters": [],
             "architecture": {"modality": "text->text"},
             "description": "A safety classifier and moderation guard."},
        ]
        models = catalogue.openrouter_models(items)
        ids = [m["id"] for m in models]
        self.assertNotIn("vendor/paid-model", ids)
        self.assertEqual(ids[0], "vendor/json:free")
        groups = {m["id"]: m["group"] for m in models}
        self.assertEqual(groups["vendor/guard:free"], "specialized")
        self.assertTrue(all(m["free"] for m in models))

    def test_groq_catalogue_intersects_free_plan(self):
        live = [
            {"id": "openai/gpt-oss-120b", "context_window": 131072},
            {"id": "llama-3.3-70b-versatile", "context_window": 131072},
            {"id": "whisper-large-v3", "context_window": 0},
        ]
        models = catalogue.groq_entries(live)
        self.assertEqual([m["id"] for m in models], ["openai/gpt-oss-120b"])

    def test_groq_fallback_without_key(self):
        data = catalogue.list_models("groq", "")
        self.assertEqual(data["source"], "fallback")
        ids = {m["id"] for m in data["models"]}
        self.assertIn("openai/gpt-oss-120b", ids)
        self.assertNotIn("llama-3.3-70b-versatile", ids)

    def test_free_model_validation(self):
        validate_model("openrouter", "vendor/anything:free")
        validate_model("groq", "openai/gpt-oss-120b")
        for provider, model in [
            ("openrouter", "vendor/anything"),
            ("groq", "llama-3.3-70b-versatile"),
            ("groq", "meta-llama/llama-prompt-guard-2-86m"),
        ]:
            with self.assertRaises(DetectorError):
                validate_model(provider, model)

    def test_openrouter_listing_uses_public_catalogue(self):
        payload = {"data": [
            {"id": "vendor/free:free", "context_length": 16000,
             "supported_parameters": ["response_format"],
             "architecture": {"modality": "text->text"}},
            {"id": "vendor/paid", "context_length": 16000},
        ]}
        response = fake_response(200, payload)
        response.raise_for_status = lambda: None
        with mock.patch.object(client._session, "get", return_value=response):
            data = catalogue.list_models("openrouter", force=True)
        self.assertEqual(data["source"], "live")
        self.assertEqual([m["id"] for m in data["models"]], ["vendor/free:free"])
        self.assertEqual(data["default"], "vendor/free:free")

    def test_fallback_models_exclude_classifiers(self):
        self.assertNotIn("openai/gpt-oss-safeguard-20b", catalogue.fallback_models("groq"))


class ErrorMappingTests(unittest.TestCase):
    def post(self, response, **kwargs):
        with mock.patch.object(client._session, "post", return_value=response), \
             mock.patch.object(client.time, "sleep"):
            client.post_chat([], OR_KEY, "test/model:free", **kwargs)

    def test_401_maps_to_key_error(self):
        with self.assertRaises(DetectorError) as ctx:
            self.post(fake_response(401, {"error": {"message": "Invalid key"}}))
        self.assertEqual(ctx.exception.status, 401)
        self.assertNotIsInstance(ctx.exception, ModelFailure)
        self.assertIn("API key", ctx.exception.message)

    def test_429_after_retries(self):
        response = fake_response(429, {"error": {"message": "Rate limit exceeded"}}, {"Retry-After": "7"})
        with self.assertRaises(RateLimited) as ctx:
            self.post(response)
        self.assertEqual(ctx.exception.status, 429)
        self.assertEqual(ctx.exception.retry_after, 7)

    def test_429_without_retries_is_immediate(self):
        response = fake_response(429, {"error": {"message": "Rate limit exceeded"}})
        with mock.patch.object(client._session, "post", return_value=response) as post:
            with self.assertRaises(RateLimited):
                client.post_chat([], OR_KEY, "test/model:free", rate_limit_retries=0)
        self.assertEqual(post.call_count, 1)

    def test_404_maps_to_model_failure(self):
        with self.assertRaises(ModelFailure) as ctx:
            self.post(fake_response(404, {"error": {"message": "No such model"}}))
        self.assertEqual(ctx.exception.status, 404)

    def test_deadline_stops_waiting(self):
        response = fake_response(429, {"error": {"message": "busy"}}, {"Retry-After": "10"})
        with self.assertRaises(DetectorError) as ctx:
            self.post(response, deadline=client.time.monotonic() + 8)
        self.assertEqual(ctx.exception.status, 504)


class ConfigTests(unittest.TestCase):
    def test_dotenv_parsing_does_not_override_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text(
                "# comment\nGLITCH_T_A=plain\nexport GLITCH_T_B=\"quoted value\"\n"
                "GLITCH_T_C=value # trailing\nGLITCH_T_D=from-file\n"
            )
            with mock.patch.dict(os.environ, {"GLITCH_T_D": "from-env"}):
                config.load_dotenv(path)
                self.assertEqual(os.environ["GLITCH_T_A"], "plain")
                self.assertEqual(os.environ["GLITCH_T_B"], "quoted value")
                self.assertEqual(os.environ["GLITCH_T_C"], "value")
                self.assertEqual(os.environ["GLITCH_T_D"], "from-env")
        for name in ("GLITCH_T_A", "GLITCH_T_B", "GLITCH_T_C"):
            os.environ.pop(name, None)

    def test_user_key_overrides_server_key(self):
        with mock.patch.dict(os.environ, {"GROQ_KEY": "gsk-from-env-0000"}):
            self.assertEqual(keys.resolve_request_keys({})["groq"], "gsk-from-env-0000")
            self.assertEqual(keys.resolve_request_keys({"groq": " gsk-user "})["groq"], "gsk-user")


class WebTests(unittest.TestCase):
    def setUp(self):
        reset_state()
        self.client = make_client()

    def test_env_key_flag_never_leaks_key(self):
        with mock.patch.dict(os.environ, {"GROQ_API_KEY": "gsk-secret-value-123"}):
            body = self.client.get("/api/health").text
            page = self.client.get("/").text
        self.assertNotIn("gsk-secret-value-123", body + page)
        providers = {p["id"]: p for p in json.loads(body)["providers"]}
        self.assertTrue(providers["groq"]["ready"])
        self.assertFalse(providers["gemini"]["ready"])

    def test_non_local_host_rejected(self):
        response = self.client.get("/api/health", headers={"Host": "evil.example:8501"})
        self.assertEqual(response.status_code, 403)
        self.assertEqual(self.client.get("/api/health").status_code, 200)

    def test_analyze_uses_env_key_when_none_supplied(self):
        seen = {}

        def fake(messages, api_key, model, provider, **kw):
            seen["key"] = api_key
            return json.dumps(GOOD)

        with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-env-key-9999"}), \
             mock.patch.object(client, "request_chunk", side_effect=fake):
            response = self.client.post("/api/analyze", json={
                "text": GOOD_TEXT, "model": "test/model:free", "fallback": False,
            })
        self.assertEqual(response.status_code, 200, response.json())
        self.assertEqual(seen["key"], "sk-or-env-key-9999")

    def test_missing_key_points_to_settings(self):
        with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}):
            response = self.client.post("/api/analyze", json={"text": GOOD_TEXT})
        self.assertEqual(response.status_code, 400)
        self.assertIn("Settings", response.json()["error"]["message"])

    def test_word_limit(self):
        long_text = "word " * (config.MAX_WORDS + 1)
        for route in ("/api/analyze", "/api/humanize"):
            with self.subTest(route=route):
                response = self.client.post(route, json={"text": long_text})
                self.assertEqual(response.status_code, 413)
                self.assertIn("2,500 words", response.json()["error"]["message"])

    def test_providers_endpoint_lists_readiness_without_keys(self):
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "AIza" + "x" * 35}):
            data = self.client.get("/api/providers").json()
        providers = {p["id"]: p for p in data["providers"]}
        self.assertTrue(providers["gemini"]["ready"])
        self.assertNotIn("AIza", json.dumps(data))


class CompactModeTests(unittest.TestCase):
    def setUp(self):
        reset_state()

    def test_label_formats(self):
        cases = [
            '{"labels": ["AI", "HUMAN", "REFINED"]}',
            '["ai", "human", "polished"]',
            '```json\n{"labels":["AI","HUMAN","REFINED",]}\n```',
            'Sure: {"1": "AI", "2": "HUMAN", "3": "REFINED"}',
            json.dumps(GOOD),  # classic-shaped replies are understood too
        ]
        for raw in cases:
            with self.subTest(raw=raw):
                labels, _ = parse_labels(raw, 3)
                self.assertEqual(labels, ["AI", "HUMAN", "REFINED"])

    def test_truncated_reply_recovers_labels(self):
        labels, warnings = parse_labels('{"labels": ["AI", "HUMAN", "REFI', 2)
        self.assertEqual(labels, ["AI", "HUMAN"])
        self.assertTrue(any("fallback parser" in w for w in warnings))

    def test_label_count_mismatch(self):
        labels, warnings = parse_labels('{"labels": ["AI", "AI", "AI", "AI"]}', 3)
        self.assertEqual(labels, ["AI"] * 3)
        labels, warnings = parse_labels('{"labels": ' + json.dumps(["AI"] * 19) + "}", 20)
        self.assertEqual(labels[-1], "HUMAN")
        self.assertTrue(any("unlabelled" in w for w in warnings))
        with self.assertRaises(ParseError):
            parse_labels('{"labels": ["AI"]}', 20)

    def test_compact_scan_sends_numbered_sentences_and_small_budget(self):
        seen = {}

        def fake(messages, api_key, model, provider, **kw):
            seen.update(messages=messages, **kw)
            return '{"labels": ["AI", "HUMAN", "REFINED"]}'

        with mock.patch.object(client, "request_chunk", side_effect=fake):
            result = analyzer.analyze(
                GOOD_TEXT, model="test/model:free", keys={"openrouter": OR_KEY},
                options=analyzer.ScanOptions(fallback=False),
            )
        self.assertEqual(seen["messages"][0]["content"], COMPACT_SYSTEM_PROMPT)
        self.assertIn("1. The system is robust and scalable.", seen["messages"][1]["content"])
        self.assertIn("3. Ultimately,", seen["messages"][1]["content"])
        self.assertLessEqual(seen["max_tokens"], 128)
        self.assertEqual(result["mode"], "compact")
        self.assertEqual(result["sentences"][1], {
            "text": "I rewrote it twice before it worked.", "classification": "HUMAN",
        })
        self.assertEqual(result["score"], 67)

    def test_classic_mode_uses_original_prompt(self):
        seen = {}

        def fake(messages, api_key, model, provider, **kw):
            seen["system"] = messages[0]["content"]
            return json.dumps(GOOD)

        with mock.patch.object(client, "request_chunk", side_effect=fake):
            result = analyzer.analyze(
                GOOD_TEXT, model="test/model:free", keys={"openrouter": OR_KEY},
                options=analyzer.ScanOptions(fallback=False, compact=False),
            )
        self.assertEqual(seen["system"], SYSTEM_PROMPT)
        self.assertEqual(result["mode"], "classic")


class TokenBudgetTests(unittest.TestCase):
    def setUp(self):
        reset_state()

    def test_reasoning_models_get_headroom_and_low_effort(self):
        plain = analyzer.Candidate("openrouter", "test/model:free")
        thinker = analyzer.Candidate("groq", "openai/gpt-oss-120b")
        self.assertLess(analyzer.output_budget(plain, "x", 10, True), 256)
        self.assertGreater(analyzer.output_budget(thinker, "x", 10, True), 768)
        self.assertEqual(catalogue.request_options("groq", "openai/gpt-oss-120b"),
                         {"reasoning_effort": "low"})
        self.assertEqual(catalogue.request_options("openrouter", "test/model:free"), {})

    def test_rejected_options_are_dropped_and_retried(self):
        bodies = []

        def fake_post(url, headers=None, json=None, timeout=None):
            bodies.append(json)
            if "reasoning_effort" in json:
                return fake_response(400, {"error": {"message": "unknown parameter"}})
            return chat_ok()

        with mock.patch.object(client._session, "post", side_effect=fake_post):
            client.post_chat([], GROQ_KEY, "openai/gpt-oss-120b", "groq",
                             options={"reasoning_effort": "low"})
        self.assertEqual(len(bodies), 2)
        self.assertNotIn("reasoning_effort", bodies[1])

    def test_truncated_reply_is_retried_with_bigger_budget(self):
        budgets = []

        def fake_post(url, headers=None, json=None, timeout=None):
            budgets.append(json["max_tokens"])
            finish = "length" if len(budgets) == 1 else "stop"
            return fake_response(200, {
                "choices": [{"message": {"content": "{}"}, "finish_reason": finish}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10},
            })

        usage = {}
        with mock.patch.object(client._session, "post", side_effect=fake_post):
            client.request_chunk([], OR_KEY, "test/model:free", max_tokens=200, usage=usage)
        self.assertEqual(budgets[0], 200)
        self.assertGreater(budgets[1], 200)
        self.assertEqual(usage, {"prompt_tokens": 200, "completion_tokens": 20})


class CacheTests(unittest.TestCase):
    def setUp(self):
        reset_state()
        self.sent = []

    def fake(self, messages, api_key, model, provider, **kw):
        body = messages[1]["content"].split("===DATA START===\n")[1].split("\n===DATA END===")[0]
        lines = [line.split(". ", 1)[1] for line in body.splitlines()]
        self.sent.append(lines)
        return json.dumps({"labels": ["AI"] * len(lines)})

    def scan(self, text, model="test/model:free", **opts):
        with mock.patch.object(client, "request_chunk", side_effect=self.fake):
            return analyzer.analyze(
                text, model=model, keys={"openrouter": OR_KEY},
                options=analyzer.ScanOptions(fallback=False, **opts),
            )

    def test_recheck_sends_nothing(self):
        first = self.scan(GOOD_TEXT)
        second = self.scan(GOOD_TEXT)
        self.assertEqual(len(self.sent), 1)
        self.assertEqual(second["chunks"], 0)
        self.assertEqual(second["reused_sentences"], 3)
        self.assertEqual(second["usage"]["total_tokens"], 0)
        self.assertEqual(second["counts"], first["counts"])
        self.assertIn("hl ai", second["html"])

    def test_edit_sends_only_changed_sentences(self):
        self.scan(GOOD_TEXT)
        edited = GOOD_TEXT.replace("I rewrote it twice", "I rewrote it three times")
        result = self.scan("New opening line here. " + edited)
        self.assertEqual(self.sent[1], [
            "New opening line here.",
            "I rewrote it three times before it worked.",
        ])
        self.assertEqual(result["reused_sentences"], 2)
        self.assertEqual(result["total"], 4)
        self.assertEqual(result["sentences"][0]["text"], "New opening line here.")

    def test_duplicate_sentences_sent_once(self):
        result = self.scan("Same line repeats here. Same line repeats here. Another one here.")
        self.assertEqual(self.sent[0], ["Same line repeats here.", "Another one here."])
        self.assertEqual(result["total"], 3)

    def test_cache_is_per_selected_model(self):
        self.scan(GOOD_TEXT)
        self.scan(GOOD_TEXT, model="other/model:free")
        self.assertEqual(len(self.sent), 2)

    def test_finished_groups_survive_a_failed_scan(self):
        text = "\n\n".join(f"Paragraph number {i} " + "word " * 130 + "end." for i in range(3))
        calls = []

        def flaky(messages, api_key, model, provider, **kw):
            calls.append(1)
            if len(calls) == 2:
                raise RateLimited("busy")
            return self.fake(messages, api_key, model, provider, **kw)

        with mock.patch.object(client, "request_chunk", side_effect=flaky), \
             self.assertRaises(RateLimited):
            analyzer.analyze(text, model="test/model:free", keys={"openrouter": OR_KEY},
                             options=analyzer.ScanOptions(chunk_words=150, fallback=False,
                                                          concurrency=1))
        result = self.scan(text, chunk_words=150, concurrency=1)
        self.assertEqual(result["reused_sentences"], 1)
        self.assertEqual(result["chunks"], 2)

    def test_classic_mode_caches_whole_sections(self):
        with mock.patch.object(client, "request_chunk", return_value=json.dumps(GOOD)) as call:
            for _ in range(2):
                result = analyzer.analyze(
                    GOOD_TEXT, model="test/model:free", keys={"openrouter": OR_KEY},
                    options=analyzer.ScanOptions(fallback=False, compact=False),
                )
        self.assertEqual(call.call_count, 1)
        self.assertEqual(result["cached_chunks"], 1)


class BalancingTests(unittest.TestCase):
    def setUp(self):
        reset_state()

    def scan(self, balance):
        text = "\n\n".join(f"Section {i} " + "word " * 130 + "end." for i in range(3))
        seen = []

        def fake(messages, api_key, model, provider, **kw):
            seen.append(model)
            return '{"labels": ["HUMAN"]}'

        with mock.patch.object(client, "request_chunk", side_effect=fake):
            result = analyzer.analyze(
                text, provider="groq", model="openai/gpt-oss-120b", keys={"groq": GROQ_KEY},
                options=analyzer.ScanOptions(chunk_words=150, balance=balance, concurrency=1),
            )
        return result, seen

    def test_sections_spread_across_free_models(self):
        result, seen = self.scan(balance=True)
        self.assertEqual(len(set(seen)), 3)
        self.assertEqual(seen[0], "openai/gpt-oss-120b")
        self.assertEqual(result["fallbacks"], [])

    def test_balancing_off_keeps_one_model(self):
        _, seen = self.scan(balance=False)
        self.assertEqual(set(seen), {"openai/gpt-oss-120b"})

    def test_fallback_off_never_switches(self):
        calls = []

        def fake(messages, api_key, model, provider, **kw):
            calls.append(model)
            raise RateLimited("busy")

        text = "\n\n".join(f"Section {i} " + "word " * 130 + "end." for i in range(3))
        with mock.patch.object(client, "request_chunk", side_effect=fake), \
             self.assertRaises(RateLimited):
            analyzer.analyze(
                text, provider="groq", model="openai/gpt-oss-120b", keys={"groq": GROQ_KEY},
                options=analyzer.ScanOptions(chunk_words=150, fallback=False, concurrency=1),
            )
        self.assertEqual(calls, ["openai/gpt-oss-120b"])


SAMPLE_KEYS = {
    "openrouter": "sk-or-v1-" + "a" * 64,
    "groq": "gsk_" + "B" * 52,
    "gemini": "AIza" + "C" * 35,
    "cerebras": "csk-" + "d" * 48,
    "mistral": "E" * 32,
    "openai": "sk-proj-" + "F" * 100,
    "deepseek": "sk-" + "0123456789abcdef" * 2,
    "xai": "xai-" + "G" * 80,
    "together": "tgp_v1_" + "H" * 40,
}


class KeyResolutionTests(unittest.TestCase):
    def test_each_provider_is_recognised_by_format(self):
        for provider, key in SAMPLE_KEYS.items():
            with self.subTest(provider=provider):
                self.assertEqual(keys.identify(key), [provider])

    def test_resolve_confirms_with_the_matching_provider_only(self):
        probed = []

        def fake_probe(key, provider):
            probed.append(provider)
            return True

        with mock.patch.object(keys, "probe", side_effect=fake_probe):
            match = keys.resolve(SAMPLE_KEYS["groq"])
        self.assertEqual((match.provider, match.verified), ("groq", True))
        self.assertEqual(probed, ["groq"])  # never sent anywhere else

    def test_unreachable_provider_trusts_unambiguous_format(self):
        with mock.patch.object(keys, "probe", return_value=None):
            match = keys.resolve(SAMPLE_KEYS["gemini"])
        self.assertEqual((match.provider, match.verified), ("gemini", False))

    def test_rejected_and_unknown_keys(self):
        with mock.patch.object(keys, "probe", return_value=False), \
             self.assertRaises(DetectorError) as ctx:
            keys.resolve(SAMPLE_KEYS["groq"])
        self.assertEqual(ctx.exception.status, 401)
        with self.assertRaises(DetectorError) as ctx:
            keys.resolve("not-a-known-key-format!!")
        self.assertIn("recognise", ctx.exception.message)

    def test_probe_uses_models_endpoint_of_that_provider(self):
        seen = {}

        def fake_get(url, headers=None, timeout=None):
            seen.update(url=url, auth=(headers or {}).get("Authorization"))
            return fake_response(200, {"data": []})

        with mock.patch.object(client._session, "get", side_effect=fake_get):
            self.assertTrue(keys.probe(SAMPLE_KEYS["gemini"], "gemini"))
        self.assertTrue(seen["url"].startswith("https://generativelanguage.googleapis.com/"))
        self.assertTrue(seen["url"].endswith("/models"))
        self.assertEqual(seen["auth"], "Bearer " + SAMPLE_KEYS["gemini"])


class KeyStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "keys.json"
        self.store = keys.KeyStore(self.path)
        self.addCleanup(self.tmp.cleanup)

    def test_add_list_remove(self):
        with mock.patch.object(keys, "probe", return_value=True):
            entry = self.store.add(SAMPLE_KEYS["groq"], "team key")
            self.store.add(SAMPLE_KEYS["groq"])  # duplicate ignored
        self.assertEqual(entry["provider"], "groq")
        self.assertNotIn(SAMPLE_KEYS["groq"], json.dumps(self.store.list()))
        self.assertEqual(self.store.key_for("groq"), SAMPLE_KEYS["groq"])
        self.assertEqual(oct(self.path.stat().st_mode & 0o777), "0o600")
        reloaded = keys.KeyStore(self.path)
        self.assertEqual(reloaded.key_for("groq"), SAMPLE_KEYS["groq"])
        self.assertTrue(reloaded.remove(entry["id"]))
        self.assertEqual(reloaded.key_for("groq"), "")

    def test_environment_keys_including_generic_list(self):
        env = {"MISTRAL_API_KEY": SAMPLE_KEYS["mistral"],
               keys.GENERIC_ENV_VAR: f"{SAMPLE_KEYS['xai']}, junk-key-value-here"}
        with mock.patch.dict(os.environ, env):
            store = keys.KeyStore(self.path)
            self.assertEqual(store.ready_providers(), {"mistral", "xai"})
            sources = {e["provider"]: e["source"] for e in store.list()}
        self.assertEqual(sources, {"mistral": "env", "xai": "env"})


class AdminApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patches = [
            mock.patch.object(keys, "store", keys.KeyStore(Path(self.tmp.name) / "k.json")),
            mock.patch.object(keys, "probe", return_value=True),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.client = make_client()

    def test_disabled_without_token(self):
        with mock.patch.object(config, "ADMIN_TOKEN", ""):
            self.assertEqual(self.client.get("/api/admin/keys").status_code, 404)

    def test_add_list_delete_with_token(self):
        auth = {"Authorization": "Bearer s3cret-admin"}
        with mock.patch.object(config, "ADMIN_TOKEN", "s3cret-admin"):
            self.assertEqual(self.client.get("/api/admin/keys").status_code, 401)
            self.assertEqual(
                self.client.get("/api/admin/keys", headers={"Authorization": "Bearer nope"}).status_code,
                401,
            )
            added = self.client.post("/api/admin/keys", json={"key": SAMPLE_KEYS["gemini"]}, headers=auth)
            self.assertEqual(added.status_code, 201)
            entry = added.json()["key"]
            self.assertEqual(entry["provider"], "gemini")  # recognised, not specified
            listed = self.client.get("/api/admin/keys", headers=auth).json()
            self.assertNotIn(SAMPLE_KEYS["gemini"], json.dumps(listed))
            ready = {p["id"] for p in self.client.get("/api/providers").json()["providers"] if p["ready"]}
            self.assertIn("gemini", ready)
            gone = self.client.delete(f"/api/admin/keys/{entry['id']}", headers=auth)
            self.assertEqual(gone.status_code, 200)

    def test_identify_endpoint_does_not_store(self):
        response = self.client.post("/api/keys/identify", json={"key": SAMPLE_KEYS["cerebras"]})
        self.assertEqual(response.json()["provider"], "cerebras")
        self.assertEqual(keys.store.ready_providers(), set())


class GenericProviderTests(unittest.TestCase):
    def setUp(self):
        reset_state()

    def test_generic_models_filter_and_rank(self):
        items = [
            {"id": "models/text-embedding-004"},
            {"id": "models/gemini-2.5-pro"},
            {"id": "models/gemini-2.5-flash"},
            {"id": "models/imagen-4"},
            {"id": "models/gemini-2.5-flash-lite"},
        ]
        models = catalogue.generic_models("gemini", items)
        ids = [m["id"] for m in models]
        self.assertNotIn("text-embedding-004", ids)
        self.assertNotIn("imagen-4", ids)
        self.assertEqual(ids[-1], "gemini-2.5-pro")  # "pro" ranks after fast models
        self.assertTrue(models[0]["reasoning"])

    def test_request_options_per_provider(self):
        self.assertEqual(catalogue.request_options("gemini", "gemini-2.5-flash"), {"reasoning_effort": "low"})
        self.assertEqual(catalogue.request_options("openai", "gpt-4.1-mini"), {})
        self.assertEqual(catalogue.request_options("openai", "o4-mini"), {"reasoning_effort": "low"})

    def test_fallback_never_moves_onto_a_paid_provider(self):
        for provider, models in (("openai", ["gpt-x"]), ("gemini", ["gemini-flash"])):
            catalogue._MODEL_CACHE[provider].update(at=1e12, payload={"models": [
                catalogue._entry(m, m, 0, "recommended", "") for m in models
            ]})
        cands = routing.build_candidates(
            "groq", "openai/gpt-oss-120b",
            {"groq": GROQ_KEY, "openai": SAMPLE_KEYS["openai"], "gemini": SAMPLE_KEYS["gemini"]},
            routing.ScanOptions(),
        )
        providers = {c.provider for c in cands}
        self.assertIn("gemini", providers)
        self.assertNotIn("openai", providers)


class HumanizeTests(unittest.TestCase):
    TEXT = (
        "The system is robust and scalable. I rewrote it twice before it worked.\n\n"
        "Ultimately, the results underscore its importance."
    )
    LABELS = [
        {"text": "The system is robust and scalable.", "classification": "AI"},
        {"text": "I rewrote it twice before it worked.", "classification": "HUMAN"},
        {"text": "Ultimately, the results underscore its importance.", "classification": "REFINED"},
    ]

    def setUp(self):
        reset_state()
        self.requests = []
        self.rewrite = lambda passage, body: f"Plainly put, {passage[0].lower()}{passage[1:]}"

    @staticmethod
    def data(messages):
        return messages[1]["content"].split("===DATA START===\n")[1].split("\n===DATA END===")[0]

    def fake(self, messages, api_key, model, provider, **kw):
        import re
        system, body = messages[0]["content"], self.data(messages)
        self.requests.append({"system": system, "body": body, **kw})
        if '"rewrites"' in system:
            passages = re.findall(r"^\[(\d+)\] (.*)$", body, re.MULTILINE)
            return json.dumps({"rewrites": {n: self.rewrite(t, body) for n, t in passages}})
        lines = [line.split(". ", 1)[1] for line in body.splitlines()]
        return json.dumps({"labels": ["HUMAN" if l.startswith("Plainly") else "AI" for l in lines]})

    def rewrites_sent(self):
        return [r for r in self.requests if '"rewrites"' in r["system"]]

    def run_humanize(self, text=None, **kw):
        kw.setdefault("labels", self.LABELS if text is None else None)
        with mock.patch.object(client, "request_chunk", side_effect=self.fake):
            return humanize_mod.humanize(
                text or self.TEXT, provider="openrouter", model="test/model:free",
                keys={"openrouter": OR_KEY}, options=routing.ScanOptions(fallback=False), **kw,
            )

    def test_flagged_passages_sent_with_context_and_specific_notes(self):
        result = self.run_humanize()
        body = self.rewrites_sent()[0]["body"]
        self.assertIn("[1] The system is robust and scalable.", body)
        self.assertIn("[1 notes]", body)
        self.assertIn("robust", body.split("[1 notes]")[1].split("\n")[0])
        self.assertIn("[context] I rewrote it twice", body)
        self.assertIn("[2] Ultimately", body)
        self.assertIn("summing-up opener", body)
        self.assertEqual(self.rewrites_sent()[0]["temperature"], humanize_mod.REWRITE_TEMPERATURE)
        self.assertIn("I rewrote it twice before it worked.", result["text"])  # human, untouched
        self.assertIn("\n\nPlainly put, ultimately", result["text"])        # paragraphs kept
        self.assertIn('<span class="hw">Plainly put, the system', result["html"])
        self.assertEqual((result["passages"], result["rewritten"], result["targets"]), (2, 2, 2))
        self.assertEqual((result["score_before"], result["score_after"]), (67, 0))
        self.assertIn("rhythm", result["style"]["before"])

    def test_consecutive_flagged_sentences_become_one_passage(self):
        text = ("Moreover, it is robust. Furthermore, it is seamless. Additionally, it is holistic.\n\n"
                "My cat sat on the keyboard at 3am and deleted the draft.")
        self.rewrite = lambda passage, body: "Plainly put, it works. Mostly."
        labels = [{"text": s, "classification": "AI"} for s in
                  ("Moreover, it is robust.", "Furthermore, it is seamless.", "Additionally, it is holistic.")]
        labels.append({"text": "My cat sat on the keyboard at 3am and deleted the draft.", "classification": "HUMAN"})
        result = self.run_humanize(text, labels=labels)
        body = self.rewrites_sent()[0]["body"]
        self.assertIn("[1] Moreover, it is robust. Furthermore, it is seamless. Additionally, it is holistic.", body)
        self.assertNotIn("[2]", body)
        self.assertEqual((result["passages"], result["rewritten"]), (1, 3))
        self.assertTrue(result["text"].startswith("Plainly put, it works. Mostly.\n\nMy cat"))

    def test_rewrite_that_keeps_ai_patterns_is_retried_with_feedback(self):
        def rewrite(passage, body):
            return "It plays a crucial role here." if "last attempt" not in body else "Plainly put, it matters."
        self.rewrite = rewrite
        result = self.run_humanize()
        retry = self.rewrites_sent()[1]["body"]
        self.assertIn("Your last attempt needs fixing", retry)
        self.assertIn("crucial", retry)
        self.assertNotIn("crucial", result["text"])

    def test_lost_numbers_keep_the_original(self):
        text = "Revenue grew 42% in 2023 at Acme. It was robust growth.\n\nI was there for it, honestly."
        labels = [
            {"text": "Revenue grew 42% in 2023 at Acme.", "classification": "AI"},
            {"text": "It was robust growth.", "classification": "AI"},
            {"text": "I was there for it, honestly.", "classification": "HUMAN"},
        ]
        self.rewrite = lambda passage, body: "Plainly put, revenue went up a lot."
        result = self.run_humanize(text, labels=labels)
        self.assertEqual(len(self.rewrites_sent()), 2)  # one retry, naming what was lost
        self.assertIn("42%", self.rewrites_sent()[1]["body"].split("last attempt")[1])
        self.assertEqual(result["passages"], 0)
        self.assertEqual(result["text"], text)
        self.assertTrue(any("Kept the original wording" in w for w in result["warnings"]))

    def test_repeat_uses_cache_and_fresh_skips_it(self):
        self.run_humanize()
        count = len(self.rewrites_sent())
        second = self.run_humanize()
        self.assertEqual(len(self.rewrites_sent()), count)
        self.assertEqual(second["reused"], 2)
        self.run_humanize(fresh=True)
        self.assertGreater(len(self.rewrites_sent()), count)

    def test_second_pass_shows_the_model_its_last_attempt(self):
        def rewrite(passage, body):
            return "Plainly put, it is fine." if "still reads your last version" in body else "Robotic version of it."
        self.rewrite = rewrite
        result = self.run_humanize(second_pass=True)
        last = self.rewrites_sent()[-1]["body"]
        self.assertIn("still reads your last version", last)
        self.assertIn("Robotic version of it.", last)
        self.assertIn("[1] The system is robust", last)  # re-done from the original
        self.assertEqual(result["score_after"], 0)

    def test_labels_are_fetched_when_not_supplied(self):
        result = self.run_humanize(self.TEXT, labels=None)
        self.assertTrue(any('"labels"' in r["system"] for r in self.requests))
        self.assertEqual(result["targets"], 3)

    def test_nothing_to_rewrite(self):
        labels = [dict(item, classification="HUMAN") for item in self.LABELS]
        result = self.run_humanize(labels=labels)
        self.assertEqual(result["rewritten"], 0)
        self.assertEqual(self.requests, [])

    def test_endpoint(self):
        with mock.patch.object(client, "request_chunk", side_effect=self.fake):
            response = make_client().post("/api/humanize", json={
                "text": self.TEXT, "model": "test/model:free", "api_keys": {"openrouter": OR_KEY},
                "labels": self.LABELS, "tone": "casual", "fallback": False, "fresh": True,
            })
        self.assertEqual(response.status_code, 200, response.json())
        result = response.json()["result"]
        self.assertEqual(result["tone"], "casual")
        self.assertIn("tips", result)


class StyleTests(unittest.TestCase):
    AI = ("In today's rapidly evolving digital landscape, artificial intelligence plays a crucial role in shaping "
          "the future of human communication. By leveraging advanced algorithms, organizations can foster "
          "innovation, streamline workflows, and unlock unprecedented levels of efficiency. Moreover, these "
          "systems demonstrate a strong commitment to adaptability in an increasingly complex world.")
    HUMAN = ("I tried the beta on a Tuesday and it ate four hours of my afternoon. The export button silently "
             "dropped every table with a merged cell, which I only noticed because a colleague asked why the "
             "revenue column was empty. It has been two years.")

    def test_ai_and_human_profiles_differ(self):
        ai, human = style.analyze_style(self.AI), style.analyze_style(self.HUMAN)
        self.assertGreater(ai.phrase_count, 8)
        self.assertEqual(human.phrase_count, 0)
        self.assertGreater(human.rhythm, ai.rhythm)

    def test_rewrite_notes_are_specific(self):
        notes = " ".join(style.rewrite_notes(self.AI))
        for expected in ("crucial", "leveraging", "three-part list", "scene-setting opener", "Moreover"):
            self.assertIn(expected, notes)

    def test_check_rewrite(self):
        lost, problems = style.check_rewrite(
            "Sales at Acme grew 42% in 2023.", "The company grew a lot, fostering growth."
        )
        self.assertTrue(any("42%" in l and "2023" in l for l in lost))
        self.assertTrue(any("Acme" in l for l in lost))
        self.assertTrue(any("fostering" in p or "foster" in p for p in problems))
        self.assertEqual(style.check_rewrite("Acme grew 42%.", "Acme grew by 42%. Fast."), ([], []))

    def test_tricolons_skip_plain_lists(self):
        self.assertEqual(style.tricolons("I bought eggs, milk and bread."), [])
        self.assertTrue(style.tricolons("We value innovation, collaboration, and resilience."))
        self.assertTrue(style.tricolons("They cut costs, cut down repetitive tasks, and ship faster."))

    def test_tricolon_regex_is_linear_on_long_input(self):
        import time
        started = time.monotonic()
        style.analyze_style(" ".join(["word"] * 3000) + ", x")
        self.assertLess(time.monotonic() - started, 1.0)

    def test_voice_markers_and_personal_validation(self):
        voiced = "AI is changing how we talk. Fast. I'm less sure the gains are real."
        self.assertGreaterEqual(len(style.voice_markers(voiced)), 3)
        flat = "Artificial intelligence changes communication. Organizations adopt new tools quickly."
        _, problems = style.check_rewrite(flat, flat, voice="personal")
        self.assertTrue(any("voiceless" in p for p in problems))
        _, problems = style.check_rewrite(flat, flat, voice="faithful")
        self.assertFalse(any("voiceless" in p for p in problems))

    def test_grounding_tips_only_for_unanchored_paragraphs(self):
        tips = style.grounding_tips(self.AI + "\n\n" + self.HUMAN)
        self.assertEqual([t["paragraph"] for t in tips], [1])


class RoutingCostTests(unittest.TestCase):
    def setUp(self):
        reset_state()

    def test_unavailable_model_is_skipped_across_runs(self):
        router = routing.ModelRouter([routing.Candidate("openrouter", "gone/model:free")])
        router.retire(routing.Candidate("openrouter", "gone/model:free"), ModelFailure("gone", status=404))
        self.assertGreater(routing.cooldown_remaining("openrouter", "gone/model:free"), 3000)

    def test_fallbacks_prefer_models_that_do_not_reason(self):
        catalogue._MODEL_CACHE["openrouter"].update(at=1e12, payload={"models": [
            catalogue._entry("a/thinker:free", "", 0, "recommended", "", reasoning=True),
            catalogue._entry("b/direct:free", "", 0, "recommended", ""),
        ]})
        self.assertEqual(catalogue.fallback_models("openrouter"), ["b/direct:free", "a/thinker:free"])


class ApiLayerTests(unittest.TestCase):
    def setUp(self):
        self.client = make_client()

    def test_config_for_static_frontend(self):
        data = self.client.get("/api/config").json()
        self.assertEqual(data["max_words"], config.MAX_WORDS)
        self.assertIn("natural", data["tones"])
        self.assertIn("personal", data["voices"])

    def test_writes_must_be_json(self):
        response = self.client.post("/api/analyze", content="text=hi",
                                    headers={"Content-Type": "application/x-www-form-urlencoded"})
        self.assertEqual(response.status_code, 415)
        self.assertFalse(response.json()["ok"])

    def test_malformed_and_missing_bodies_use_the_error_envelope(self):
        bad = self.client.post("/api/analyze", content="{not json", headers={"Content-Type": "application/json"})
        self.assertEqual(bad.status_code, 400)
        self.assertIn("message", bad.json()["error"])
        empty = self.client.post("/api/analyze", headers={"Content-Type": "application/json"})
        self.assertEqual(empty.status_code, 400)
        self.assertIn("80+ characters", empty.json()["error"]["message"])

    def test_lenient_fields(self):
        with mock.patch.object(client, "request_chunk", return_value=json.dumps(GOOD)):
            response = self.client.post("/api/analyze", json={
                "text": GOOD_TEXT, "model": "test/model:free", "api_keys": {"openrouter": OR_KEY},
                "chunk_words": "lots", "concurrency": None, "fallback": False, "extra": 1,
            })
        self.assertEqual(response.status_code, 200, response.json())

    def test_pages_and_static_files(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertIn("text/html", self.client.get("/").headers["content-type"])
        self.assertEqual(self.client.get("/static/app.js").status_code, 200)
        self.assertEqual(self.client.get("/admin").status_code, 404)  # until frontend/admin.html exists
        self.assertEqual(self.client.get("/..%2Fbackend%2Fmain").status_code, 404)
        self.assertEqual(self.client.get("/api/nope").json()["error"]["message"], "Not found.")

    def test_security_headers_and_docs_policy(self):
        app_csp = self.client.get("/").headers["content-security-policy"]
        self.assertIn("script-src 'self';", app_csp)
        self.assertNotIn("unsafe-inline", app_csp.split("style-src")[0])
        docs = self.client.get("/docs")
        self.assertEqual(docs.status_code, 200)
        self.assertIn("cdn.jsdelivr.net", docs.headers["content-security-policy"])
        self.assertEqual(self.client.get("/api/health").headers["x-frame-options"], "DENY")


if __name__ == "__main__":
    unittest.main(verbosity=2)
