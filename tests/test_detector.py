import json
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import detector  # noqa: E402


GOOD = {
    "sentences": [
        {"text": "The system is robust and scalable.", "classification": "AI"},
        {"text": "I rewrote it twice before it worked.", "classification": "HUMAN"},
        {"text": "Ultimately, the results underscore its importance.", "classification": "REFINED"},
    ]
}


class ParsingTests(unittest.TestCase):
    def test_plain_json(self):
        sentences, warnings = detector.parse_analysis(json.dumps(GOOD))
        self.assertEqual(len(sentences), 3)
        self.assertEqual([s.classification for s in sentences], ["AI", "HUMAN", "REFINED"])
        self.assertEqual(warnings, [])

    def test_fenced_json(self):
        raw = "```json\n" + json.dumps(GOOD) + "\n```"
        sentences, _ = detector.parse_analysis(raw)
        self.assertEqual(len(sentences), 3)

    def test_prose_wrapper(self):
        raw = "Sure! Here is the analysis:\n" + json.dumps(GOOD) + "\nLet me know if you need more."
        sentences, _ = detector.parse_analysis(raw)
        self.assertEqual(len(sentences), 3)

    def test_trailing_commas(self):
        raw = '{"sentences": [{"text": "One.", "classification": "AI"},]}'
        sentences, warnings = detector.parse_analysis(raw)
        self.assertEqual(sentences[0].classification, "AI")
        self.assertTrue(any("Repaired" in w for w in warnings))

    def test_regex_fallback_reverse_order(self):
        # Truncated JSON (missing the closing array/object) cannot be repaired
        # by the trailing-comma pass, so the structural regex must recover it.
        raw = '{"sentences": [{"classification": "HUMAN", "text": "I walked home at dusk."}'
        sentences, warnings = detector.parse_analysis(raw)
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
        sentences, _ = detector.parse_analysis(raw)
        self.assertEqual([s.classification for s in sentences], ["AI", "REFINED", "HUMAN"])

    def test_unknown_label_defaults_human(self):
        raw = json.dumps({"sentences": [{"text": "A.", "classification": "???"}]})
        sentences, warnings = detector.parse_analysis(raw)
        self.assertEqual(sentences[0].classification, "HUMAN")
        self.assertTrue(any("unrecognised" in w for w in warnings))

    def test_empty_response_raises(self):
        with self.assertRaises(detector.ParseError):
            detector.parse_analysis("")

    def test_no_sentences_raises(self):
        with self.assertRaises(detector.ParseError):
            detector.parse_analysis("I cannot help with that.")

    def test_content_parts(self):
        self.assertEqual(
            detector._content_to_text([{"type": "text", "text": "a"}, {"text": "b"}]),
            "ab",
        )


class ChunkTests(unittest.TestCase):
    def test_short_text_single_chunk(self):
        self.assertEqual(detector.chunk_text("One two three."), ["One two three."])

    def test_chunks_respect_limit(self):
        text = "\n\n".join(["word " * 60 for _ in range(10)])
        chunks = detector.chunk_text(text, max_words=120)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk.split()), 130)

    def test_long_paragraph_splits_on_sentences(self):
        paragraph = " ".join(["This is a sentence about systems."] * 60)
        chunks = detector.chunk_text(paragraph, max_words=100)
        self.assertGreater(len(chunks), 1)


class HtmlTests(unittest.TestCase):
    def test_escapes_untrusted_text(self):
        sentences = [
            detector.Sentence('<script>alert("x")</script>', "AI"),
            detector.Sentence("Normal sentence.", "HUMAN"),
        ]
        html = detector.build_highlight_html(sentences, "<script>alert('x')</script> Normal sentence.")
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)
        self.assertIn('class="hl ai"', html)
        self.assertIn('class="hl human"', html)

    def test_paragraph_reconstruction(self):
        sentences = [
            detector.Sentence("First para one.", "HUMAN"),
            detector.Sentence("First para two.", "AI"),
            detector.Sentence("Second para.", "REFINED"),
        ]
        source = "First para one. First para two.\n\nSecond para."
        html = detector.build_highlight_html(sentences, source)
        self.assertEqual(html.count("doc-para"), 2)


class PromptTests(unittest.TestCase):
    def test_prompt_is_verbatim(self):
        self.assertTrue(detector.SYSTEM_PROMPT.startswith(
            "You are a highly advanced computational linguistics model specialized in deep text forensics"
        ))
        self.assertIn("1. \"AI\" - Clearly machine-generated text", detector.SYSTEM_PROMPT)
        self.assertIn("2. \"REFINED\" - Text that shows heavy AI optimization", detector.SYSTEM_PROMPT)
        self.assertIn("3. \"HUMAN\" - Authentic human text", detector.SYSTEM_PROMPT)
        self.assertIn("You must respond ONLY with a raw, valid JSON object.", detector.SYSTEM_PROMPT)
        self.assertNotIn("```json\n{", detector.SYSTEM_PROMPT.split("format must look exactly like this:")[0])

    def test_payload_shape(self):
        captured = {}

        class FakeResponse:
            status_code = 200

            @staticmethod
            def json():
                return {"choices": [{"message": {"content": json.dumps(GOOD)}}]}

        def fake_post(url, headers=None, json=None, timeout=None):
            captured.update(url=url, headers=headers, json=json, timeout=timeout)
            return FakeResponse()

        with mock.patch.object(detector.requests, "post", side_effect=fake_post):
            out = detector.request_chunk("Some text.", "sk-test-key-1234", "test/model:free", 1, 1)

        self.assertEqual(captured["url"], "https://openrouter.ai/api/v1/chat/completions")
        self.assertEqual(captured["json"]["messages"][0]["role"], "system")
        self.assertEqual(captured["json"]["messages"][0]["content"], detector.SYSTEM_PROMPT)
        self.assertEqual(captured["json"]["messages"][1]["role"], "user")
        self.assertEqual(captured["json"]["temperature"], 0.0)
        self.assertIn("Bearer sk-test-key-1234", captured["headers"]["Authorization"])
        self.assertIn("HTTP-Referer", captured["headers"])
        self.assertEqual(out, json.dumps(GOOD))

    def test_groq_payload_shape(self):
        captured = {}

        class FakeResponse:
            status_code = 200

            @staticmethod
            def json():
                return {"choices": [{"message": {"content": json.dumps(GOOD)}}]}

        def fake_post(url, headers=None, json=None, timeout=None):
            captured.update(url=url, headers=headers, json=json, timeout=timeout)
            return FakeResponse()

        with mock.patch.object(detector.requests, "post", side_effect=fake_post):
            detector.request_chunk(
                "Some text.", "gsk-test-key-1234", "openai/gpt-oss-120b", 1, 1, provider="groq"
            )

        self.assertEqual(captured["url"], "https://api.groq.com/openai/v1/chat/completions")
        self.assertNotIn("HTTP-Referer", captured["headers"])
        self.assertEqual(captured["json"]["messages"][0]["content"], detector.SYSTEM_PROMPT)
        self.assertEqual(captured["json"]["temperature"], 0.0)


class ScoreTests(unittest.TestCase):
    def test_score_math_and_html(self):
        raw = json.dumps(GOOD)

        def fake_chunk(chunk, api_key, model, index, total, provider="openrouter"):
            return raw

        text = (
            "The system is robust and scalable. "
            "I rewrote it twice before it worked. "
            "Ultimately, the results underscore its importance."
        )
        with mock.patch.object(detector, "request_chunk", side_effect=fake_chunk):
            result = detector.analyze(text, "sk-test-key-1234", "test/model:free")

        self.assertEqual(result["counts"], {"AI": 1, "REFINED": 1, "HUMAN": 1})
        self.assertEqual(result["total"], 3)
        self.assertEqual(result["score"], 67)  # (1 + 1) / 3 -> 66.67 rounds to 67
        self.assertEqual(result["chunks"], 1)
        self.assertIn("hl ai", result["html"])

    def test_validation(self):
        with self.assertRaises(detector.DetectorError):
            detector.analyze("", "sk-test-key-1234")
        with self.assertRaises(detector.DetectorError):
            detector.analyze("A sentence long enough to pass.", "short")
        with self.assertRaises(detector.DetectorError):
            detector.analyze("A sentence long enough to pass.", "sk-test-key-1234", "bad-model")
        # Paid OpenRouter models are refused even if a caller sends the ID.
        with self.assertRaises(detector.DetectorError):
            detector.analyze(
                "A sentence long enough to pass.",
                "sk-test-key-1234",
                "google/gemma-4-31b-it",
            )
        # Paid / enterprise Groq models are refused.
        with self.assertRaises(detector.DetectorError):
            detector.analyze(
                "A sentence long enough to pass.",
                "gsk-test-key-1234",
                "llama-3.3-70b-versatile",
                provider="groq",
            )


class ProviderTests(unittest.TestCase):
    def test_openrouter_catalogue_keeps_only_free_and_ranks_task_fit(self):
        items = [
            {
                "id": "vendor/paid-model",
                "name": "Paid",
                "context_length": 128000,
                "supported_parameters": ["response_format"],
                "architecture": {"modality": "text->text"},
                "description": "A paid model.",
            },
            {
                "id": "vendor/plain:free",
                "name": "Plain",
                "context_length": 8000,
                "supported_parameters": [],
                "architecture": {"modality": "text->text"},
                "description": "General chat model.",
            },
            {
                "id": "vendor/json:free",
                "name": "JSON",
                "context_length": 32000,
                "supported_parameters": ["response_format", "structured_outputs"],
                "architecture": {"modality": "text->text"},
                "description": "Instruction-tuned model for extraction.",
            },
            {
                "id": "vendor/guard:free",
                "name": "Guard",
                "context_length": 8000,
                "supported_parameters": [],
                "architecture": {"modality": "text->text"},
                "description": "A safety classifier and moderation guard.",
            },
        ]
        models = detector._openrouter_models(items)
        ids = [m["id"] for m in models]
        self.assertNotIn("vendor/paid-model", ids)
        self.assertEqual(ids[0], "vendor/json:free")
        self.assertEqual(models[0]["group"], "recommended")
        groups = {m["id"]: m["group"] for m in models}
        self.assertEqual(groups["vendor/guard:free"], "specialized")
        self.assertTrue(all(m["free"] for m in models))

    def test_groq_catalogue_intersects_free_plan(self):
        live = [
            {"id": "openai/gpt-oss-120b", "context_window": 131072},
            {"id": "llama-3.3-70b-versatile", "context_window": 131072},
            {"id": "whisper-large-v3", "context_window": 0},
        ]
        models = detector._groq_entries(live)
        ids = [m["id"] for m in models]
        self.assertEqual(ids, ["openai/gpt-oss-120b"])
        self.assertEqual(models[0]["group"], "recommended")

    def test_groq_fallback_without_key(self):
        data = detector.list_models("groq", "")
        self.assertEqual(data["source"], "fallback")
        ids = {m["id"] for m in data["models"]}
        self.assertIn("openai/gpt-oss-120b", ids)
        self.assertNotIn("llama-3.3-70b-versatile", ids)
        self.assertNotIn("whisper-large-v3", ids)

    def test_free_model_validation(self):
        detector.validate_model("openrouter", "vendor/anything:free")
        detector.validate_model("groq", "openai/gpt-oss-120b")
        for provider, model in [
            ("openrouter", "vendor/anything"),
            ("groq", "llama-3.3-70b-versatile"),
            ("groq", "meta-llama/llama-prompt-guard-2-86m"),
        ]:
            with self.assertRaises(detector.DetectorError):
                detector.validate_model(provider, model)

    def test_openrouter_listing_uses_public_catalogue(self):
        payload = {
            "data": [
                {
                    "id": "vendor/free:free",
                    "name": "Free",
                    "context_length": 16000,
                    "supported_parameters": ["response_format"],
                    "architecture": {"modality": "text->text"},
                },
                {"id": "vendor/paid", "name": "Paid", "context_length": 16000},
            ]
        }

        class Response:
            status_code = 200

            @staticmethod
            def raise_for_status():
                return None

            @staticmethod
            def json():
                return payload

        detector._MODEL_CACHE["openrouter"]["payload"] = None
        with mock.patch.object(detector.requests, "get", return_value=Response()):
            data = detector.list_models("openrouter", force=True)

        self.assertEqual(data["source"], "live")
        self.assertEqual([m["id"] for m in data["models"]], ["vendor/free:free"])
        self.assertEqual(data["default"], "vendor/free:free")


class ErrorMappingTests(unittest.TestCase):
    def _response(self, status, payload):
        class R:
            status_code = status
            headers = {}

            def __init__(self):
                self.text = json.dumps(payload)

            @staticmethod
            def json():
                return payload

        return R()

    def test_401_maps_to_key_error(self):
        with mock.patch.object(
            detector.requests, "post",
            return_value=self._response(401, {"error": {"message": "Invalid key"}}),
        ):
            with self.assertRaises(detector.DetectorError) as ctx:
                detector._post([], "sk-test-key-1234", "test/model:free")
        self.assertEqual(ctx.exception.status, 401)
        self.assertIn("API key", ctx.exception.message)

    def test_429_after_retries(self):
        response = self._response(429, {"error": {"message": "Rate limit exceeded"}})
        response.headers = {"Retry-After": "0"}
        with mock.patch.object(detector.requests, "post", return_value=response), \
             mock.patch.object(detector.time, "sleep"):
            with self.assertRaises(detector.DetectorError) as ctx:
                detector._post([], "sk-test-key-1234", "test/model:free")
        self.assertEqual(ctx.exception.status, 429)
        self.assertIn("rate limit", ctx.exception.message.lower())

    def test_404_maps_to_model_error(self):
        with mock.patch.object(
            detector.requests, "post",
            return_value=self._response(404, {"error": {"message": "No such model"}}),
        ):
            with self.assertRaises(detector.DetectorError) as ctx:
                detector._post([], "sk-test-key-1234", "test/model:free")
        self.assertEqual(ctx.exception.status, 404)


if __name__ == "__main__":
    unittest.main(verbosity=2)
