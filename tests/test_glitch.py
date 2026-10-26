import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from glitch import analyzer, catalogue, client, config, create_app  # noqa: E402
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


# Never let the developer's real .env keys, or any request, leave a test.
# Individual tests patch these again with fakes where they need a response.
_env_patches = [
    mock.patch.object(config, "ENV_KEY_ALIASES", {}),
    mock.patch.dict(os.environ, {name: "" for name in config.ENV_KEY_NAMES.values()}),
    mock.patch.object(client._session, "get", side_effect=_offline),
    mock.patch.object(client._session, "post", side_effect=_offline),
]


def setUpModule():
    for patch in _env_patches:
        patch.start()


def tearDownModule():
    for patch in reversed(_env_patches):
        patch.stop()


def reset_state():
    analyzer._cooldowns.clear()
    analyzer.clear_cache()
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
        self.assertGreater(analyzer.cooldown_remaining("openrouter", "a/first:free"), 0)
        self.assertNotIn("c/guard:free", [m for _, m, _ in calls])

    def test_cooldown_skips_model_on_next_scan(self):
        analyzer._start_cooldown("openrouter", "a/first:free", 60)
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

    def test_user_key_overrides_env_key(self):
        with mock.patch.dict(os.environ, {"GROQ_API_KEY": "gsk-from-env-0000"}):
            self.assertEqual(config.resolve_key("groq", ""), "gsk-from-env-0000")
            self.assertEqual(config.resolve_key("groq", " gsk-user "), "gsk-user")


class WebTests(unittest.TestCase):
    def setUp(self):
        reset_state()
        self.app = create_app()
        self.client = self.app.test_client()

    def test_env_key_flag_never_leaks_key(self):
        with mock.patch.dict(os.environ, {"GROQ_API_KEY": "gsk-secret-value-123"}):
            body = self.client.get("/api/health").get_data(as_text=True)
            page = self.client.get("/").get_data(as_text=True)
        self.assertNotIn("gsk-secret-value-123", body + page)
        self.assertTrue(json.loads(body)["providers"]["groq"]["env_key"])

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
        self.assertEqual(response.status_code, 200, response.get_json())
        self.assertEqual(seen["key"], "sk-or-env-key-9999")

    def test_missing_key_points_to_settings(self):
        with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}):
            response = self.client.post("/api/analyze", json={"text": GOOD_TEXT})
        self.assertEqual(response.status_code, 400)
        self.assertIn("Settings", response.get_json()["error"]["message"])

    def test_humanize_placeholder(self):
        response = self.client.post("/api/humanize", json={"text": "x"})
        self.assertEqual(response.status_code, 501)


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

    def test_recheck_is_served_from_cache(self):
        reply = '{"labels": ["AI", "HUMAN", "REFINED"]}'
        options = analyzer.ScanOptions(fallback=False)
        with mock.patch.object(client, "request_chunk", return_value=reply) as call:
            first = analyzer.analyze(GOOD_TEXT, model="test/model:free",
                                     keys={"openrouter": OR_KEY}, options=options)
            second = analyzer.analyze(GOOD_TEXT, model="test/model:free",
                                      keys={"openrouter": OR_KEY}, options=options)
            third = analyzer.analyze(GOOD_TEXT, model="other/model:free",
                                     keys={"openrouter": OR_KEY}, options=options)
        self.assertEqual(call.call_count, 2)  # first scan + different model
        self.assertEqual(first["cached_chunks"], 0)
        self.assertEqual(second["cached_chunks"], 1)
        self.assertEqual(second["counts"], first["counts"])
        self.assertEqual(third["cached_chunks"], 0)


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
