"""Lightweight import smoke tests (no GPU, no large data)."""

from __future__ import annotations

import re
import unittest


class TestImports(unittest.TestCase):
    def test_cambridge_exp_modules(self):
        from cambridge_exp import config, data_pipeline, model, tfidf_baseline_cli

        self.assertIsNotNone(config.TrainConfig)
        self.assertIsNotNone(model.MultiTaskRoberta)
        self.assertIsNotNone(data_pipeline.build_eval_dataset)
        self.assertIsNotNone(tfidf_baseline_cli.make_tfidf_pipeline)

    def test_parse_heads(self):
        from cambridge_exp.train_cli import parse_heads

        self.assertEqual(parse_heads("cefr,l1,nat"), ("cefr", "l1", "nat"))
        self.assertEqual(parse_heads("l1"), ("l1",))
        with self.assertRaises(ValueError):
            parse_heads("cefr,foo")

    def test_fastapi_app_factory(self):
        import api.main

        self.assertIsNotNone(api.main.app)
        self.assertEqual(api.main.app.title, "EF-CamDAT L2 Profiler")

    def test_ollama_narrate_prompt(self):
        from api.ollama_narrate import (
            OLLAMA_NUM_PREDICT,
            SUMMARY_MODE,
            build_ollama_prompt,
            _validate_summary,
            _deterministic_fallback,
            _relative_pct,
        )

        system, user = build_ollama_prompt(
            "L1",
            "Italian",
            "Mario Bianchi from Modena spoke.",
            100,
            ["Mario", "Bianchi", "Modena"],
            "Emilia Romagna region.",
            46,
        )
        self.assertIn("Do NOT use the words might, could, may", system)
        self.assertIn("Max 30 words.", system)
        self.assertIn("Head: L1, Prediction: Italian", user)
        self.assertIn('Top sentence (100%): "Mario Bianchi from Modena spoke."', user)
        self.assertIn("Key words: Mario, Bianchi, Modena", user)
        self.assertIn('Second sentence (46%): "Emilia Romagna region."', user)
        self.assertIn("'The model predicted Italian mainly because of'", user)
        self.assertNotIn("Rules:", user)
        self.assertEqual(OLLAMA_NUM_PREDICT, 60)
        self.assertEqual(SUMMARY_MODE, "openrouter_v1")

        second = {"sentence": "The economy grew slowly.", "toward_mass": 0.4}
        self.assertEqual(_relative_pct(second, 0.8), 50)

        good = _validate_summary(
            "The model predicted Italian mainly because of 'Modena' and 'Bianchi' in the top sentence.",
            "Italian",
            ["Modena", "Bianchi", "Mario"],
        )
        self.assertIsNotNone(good)
        self.assertLessEqual(len(good.split()), 30)

        bad = _validate_summary(
            "The model might predict Italian because of uncertain patterns.",
            "Italian",
            ["Modena", "Bianchi"],
        )
        self.assertIsNone(bad)

        bad2 = _validate_summary(
            "The model predicted Italian mainly because of Modena which contributed to the signal.",
            "Italian",
            ["Modena", "Bianchi"],
        )
        self.assertIsNone(bad2)

        cefr_hallucination = _validate_summary(
            "The model predicted B2 mainly because of a high prevalence of elevated cholesterol levels.",
            "B2",
            ["HIGH", "LEVEL", "OF"],
        )
        self.assertIsNone(cefr_hallucination)

        cefr_good = _validate_summary(
            "The model predicted B2 mainly because of 'HIGH', 'LEVEL', and 'OF' in the top sentence.",
            "B2",
            ["HIGH", "LEVEL", "OF"],
        )
        self.assertIsNotNone(cefr_good)

        from api.ollama_narrate import _clean_token, _key_words

        self.assertEqual(_clean_token("government,"), "government")
        row = {
            "tokens": [
                {"text": "government,", "attribution": 0.5},
                {"text": "the", "attribution": 0.1},
            ]
        }
        self.assertEqual(_key_words(row, 1), ["government"])

        from api.sentence_shap import is_flat_attribution, template_narrative

        flat_rows = [
            {"sentence": "a", "toward_mass": 1.0, "direction": "positive", "attribution": 0.2},
            {"sentence": "b", "toward_mass": 0.99, "direction": "positive", "attribution": 0.2},
            {"sentence": "c", "toward_mass": 0.98, "direction": "positive", "attribution": 0.2},
        ]
        self.assertTrue(is_flat_attribution(flat_rows))
        flat_narr = template_narrative("nat", "de", flat_rows)
        self.assertIn("spread evenly", flat_narr)

        spread_rows = [
            {"sentence": "a", "toward_mass": 1.0, "direction": "positive", "attribution": 0.5, "tokens": []},
            {"sentence": "b", "toward_mass": 0.5, "direction": "positive", "attribution": 0.3, "tokens": []},
        ]
        self.assertFalse(is_flat_attribution(spread_rows))

        top = {
            "sentence": "Japanese fleets returned to Japan.",
            "attribution": 0.46,
            "tokens": [
                {"text": "Japanese", "attribution": 0.4},
                {"text": "Japan", "attribution": 0.3},
                {"text": "Leite", "attribution": 0.2},
            ],
        }
        fallback = _deterministic_fallback("Japanese", top, None)
        self.assertIn("Japanese", fallback)
        self.assertLessEqual(len(fallback.split()), 30)


if __name__ == "__main__":
    unittest.main()
