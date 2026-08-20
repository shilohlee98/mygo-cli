import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

import mygo_search
from mygo_search import (
    MyGoEngine,
    OnnxSentenceEncoder,
    fuzzy_search,
    lexical_similarity,
    normalize_semantic_query,
    resolve_onnx_model_file_name,
    resolve_writable_dir,
)
from mygo_app import acquire_instance_lock, result_grid_metrics


IMAGES = [
    {"alt": "大家冷靜", "episode": 1, "popularity": 5, "url": "https://example.com/1.jpg"},
    {"alt": "冷靜不了", "episode": 2, "popularity": 8, "url": "https://example.com/2.jpg"},
]


class FakeModel:
    def encode(self, _queries, normalize_embeddings=True, is_query=False):
        self.queries = list(_queries)
        self.normalize_embeddings = normalize_embeddings
        self.is_query = is_query
        return np.array([[1.0, 0.0]])


class SearchTests(unittest.TestCase):
    def test_short_english_greetings_are_normalized_for_semantic_search(self):
        self.assertEqual(normalize_semantic_query("hi"), "妳好")
        self.assertEqual(normalize_semantic_query(" Hello "), "妳好")
        self.assertEqual(normalize_semantic_query("high"), "high")
        self.assertEqual(normalize_semantic_query("say hi"), "say hi")

    def test_short_english_query_does_not_match_inside_an_english_word(self):
        self.assertEqual(lexical_similarity("hi", "祥在CRYCHIC時"), 0.0)
        self.assertGreaterEqual(lexical_similarity("hi", "她說 hi"), 0.85)
        self.assertEqual(lexical_similarity("crying", "祥在CRYCHIC時"), 0.0)

    def test_hybrid_uses_normalized_semantics_but_keeps_original_query(self):
        model = FakeModel()
        images = [
            {"alt": "妳好", "popularity": 0},
            {"alt": "祥在CRYCHIC時", "popularity": 0},
        ]
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])

        results = mygo_search.hybrid_search("hi", model, embeddings, images, top_n=2)

        self.assertEqual(model.queries, ["妳好"])
        self.assertEqual(results[0][1]["alt"], "妳好")

    def test_result_grid_reflows_and_keeps_images_widescreen(self):
        narrow = result_grid_metrics(578, 12)
        normal = result_grid_metrics(738, 12)
        wide = result_grid_metrics(1358, 12)

        self.assertEqual(narrow[0], 1)
        self.assertEqual(normal[0], 2)
        self.assertEqual(wide[0], 3)
        self.assertAlmostEqual(normal[4] / normal[3], 9 / 16)
        self.assertGreater(normal[4], 190)

    def test_onnx_encoder_limits_large_jobs_to_batches(self):
        encoder = OnnxSentenceEncoder.__new__(OnnxSentenceEncoder)
        with patch.object(
            encoder,
            "_encode_batch",
            side_effect=lambda texts, _normalize: np.ones((len(texts), 384)),
        ) as encode_batch:
            result = encoder.encode([str(index) for index in range(65)], batch_size=32)

        self.assertEqual(result.shape, (65, 384))
        self.assertEqual(encode_batch.call_count, 3)

    def test_unwritable_cache_uses_fallback_directory(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            not_a_directory = root / "blocked"
            not_a_directory.write_text("file", encoding="utf-8")
            fallback = root / "fallback"

            self.assertEqual(resolve_writable_dir(not_a_directory, fallback), fallback)

    def test_e5_uses_single_file_int8_onnx_by_default(self):
        with patch.object(mygo_search, "ONNX_QUANTIZATION", "auto"):
            self.assertEqual(resolve_onnx_model_file_name(), "model_int8.onnx")
        with patch.object(mygo_search, "ONNX_QUANTIZATION", "off"):
            self.assertEqual(resolve_onnx_model_file_name(), "model.onnx")

    def test_onnx_failure_has_actionable_error(self):
        with (
            patch.object(mygo_search, "MODEL_BACKEND", "onnx"),
            patch.object(
                mygo_search,
                "load_onnx_model",
                side_effect=PermissionError("cache denied"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "Could not load the ONNX model"):
                mygo_search.load_model(status=None)

    def test_instance_lock_allows_only_one_process(self):
        with TemporaryDirectory() as directory:
            data_dir = Path(directory)
            first = acquire_instance_lock(data_dir)
            self.assertIsNotNone(first)
            self.assertIsNone(acquire_instance_lock(data_dir))
            first.close()
            second = acquire_instance_lock(data_dir)
            self.assertIsNotNone(second)
            second.close()

    def test_fuzzy_search_prefers_exact_match(self):
        results = fuzzy_search("大家冷靜", IMAGES)
        self.assertEqual(results[0][1]["alt"], "大家冷靜")

    def test_long_lived_engine_supports_semantic_and_fuzzy_queries(self):
        engine = MyGoEngine(status=None)
        engine.images = IMAGES
        engine.model = FakeModel()
        engine.embeddings = np.array([[0.9, 0.1], [0.2, 0.8]])

        self.assertEqual(engine.search("冷靜", top_n=1)[0][1]["alt"], "大家冷靜")
        self.assertEqual(engine.search("/s 冷靜", top_n=1)[0][1]["alt"], "大家冷靜")
        self.assertEqual(engine.search("/f 大家冷靜", top_n=1)[0][1]["alt"], "大家冷靜")

    def test_hybrid_exact_match_beats_semantic_only_top_result(self):
        engine = MyGoEngine(status=None)
        engine.images = IMAGES
        engine.model = FakeModel()
        engine.embeddings = np.array([[0.0, 1.0], [1.0, 0.0]])

        self.assertEqual(engine.search("大家冷靜", top_n=1)[0][1]["alt"], "大家冷靜")
        self.assertEqual(engine.search("/s 大家冷靜", top_n=1)[0][1]["alt"], "冷靜不了")

    def test_empty_query_returns_no_results(self):
        engine = MyGoEngine(status=None)
        engine.images = IMAGES
        engine.model = FakeModel()
        engine.embeddings = np.array([[0.9, 0.1], [0.2, 0.8]])

        self.assertEqual(engine.search("   "), [])


if __name__ == "__main__":
    unittest.main()
