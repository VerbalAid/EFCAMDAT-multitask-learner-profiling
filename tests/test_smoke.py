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

        second = {"sentence": "The economy grew slowly.", "signed_mass": 0.4}
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
            "The model predicted B2 mainly because of complex vocabulary such as ignados and the discussion of future events.",
            "B2",
            ["political", "including", "the"],
        )
        self.assertIsNone(cefr_hallucination)

        stopword_only = _validate_summary(
            "The model predicted B2 mainly because of the discussion of future events.",
            "B2",
            ["political", "including"],
        )
        self.assertIsNone(stopword_only)

        empty_keywords = _validate_summary(
            "The model predicted B2 mainly because of the way of living by then.",
            "B2",
            [],
        )
        self.assertIsNone(empty_keywords)

        cefr_good = _validate_summary(
            "The model predicted B2 mainly because of 'HIGH', 'LEVEL', and 'OF' in the top sentence.",
            "B2",
            ["HIGH", "LEVEL", "OF"],
        )
        self.assertIsNotNone(cefr_good)

        nat_hallucination = _validate_summary(
            "The model predicted mx mainly because of all recent political events, "
            "including the Arab Spring and Occupy movements, which are part of a global trend.",
            "mx",
            ["First", "all", "recent"],
        )
        self.assertIsNone(nat_hallucination)

        unquoted_match = _validate_summary(
            "The model predicted mx mainly because of all recent political events in the top sentence.",
            "mx",
            ["all", "recent"],
        )
        self.assertIsNone(unquoted_match)

        too_long = _validate_summary(
            "The model predicted Italian mainly because of 'Modena' and 'Bianchi' in the top sentence "
            "and also many other long phrases that make this summary far too verbose for the UI "
            "because it keeps going on and on without stopping.",
            "Italian",
            ["Modena", "Bianchi"],
        )
        self.assertIsNone(too_long)

        from api.ollama_narrate import _clean_token, _key_words

        self.assertEqual(_clean_token("government,"), "government")
        row = {
            "tokens": [
                {"text": "government,", "attribution": 0.5},
                {"text": "the", "attribution": 0.1},
            ]
        }
        self.assertEqual(_key_words(row, 1), ["government"])

        empty_tokens_row = {"sentence": "This is different from what I expected.", "tokens": []}
        self.assertIn("different", _key_words(empty_tokens_row, 2))

        from api.gradient_attr import _calibrate_attribution_sign
        import numpy as np

        attr = np.array([-1.0, -2.0, 0.5])
        flipped = _calibrate_attribution_sign(attr, [0, 1], logit_input=5.0, logit_baseline=0.0)
        self.assertGreater(float(flipped[0]), 0.0)
        self.assertGreater(float(flipped[1]), 0.0)

        from api.sentence_shap import is_flat_attribution, template_narrative, aggregate_to_sentences

        flat_rows = [
            {"sentence": "a", "signed_mass": 1.0, "direction": "positive", "attribution": 0.2},
            {"sentence": "b", "signed_mass": 0.99, "direction": "positive", "attribution": 0.2},
            {"sentence": "c", "signed_mass": 0.98, "direction": "positive", "attribution": 0.2},
        ]
        self.assertTrue(is_flat_attribution(flat_rows))
        flat_narr = template_narrative("nat", "de", flat_rows)
        self.assertIn("overall essay organisation", flat_narr)
        self.assertNotIn("spread evenly", flat_narr)

        from api.sentence_shap import decompose_head_attribution

        per_head = {
            "cefr": [{"sentence": "Shared sent.", "signed_mass": 1.0, "tokens": []}],
            "l1": [{"sentence": "Shared sent.", "signed_mass": 1.0, "tokens": []}],
            "nat": [
                {"sentence": "Shared sent.", "signed_mass": 1.0, "tokens": []},
                {"sentence": "Nat only.", "signed_mass": 0.8, "tokens": []},
            ],
        }
        dec = decompose_head_attribution(per_head, ["cefr", "l1", "nat"])
        self.assertTrue(dec["shared_signals"])
        self.assertIn("Shared sent.", dec["shared_signals"][0]["sentence"])

        from api.main import _confidence_warning, ProbItem

        weak = _confidence_warning([ProbItem(label="jp", prob=0.41), ProbItem(label="de", prob=0.35)])
        self.assertIn("Weak signal", weak)
        strong = _confidence_warning([ProbItem(label="B2", prob=0.95), ProbItem(label="B1", prob=0.03)])
        self.assertIsNone(strong)

        from api.evidence import build_head_evidence

        cefr_ev = build_head_evidence("cefr", "B2", "However, learning is important. In conclusion, it helps.")
        features = [e["feature"] for e in cefr_ev]
        self.assertTrue(any("discourse" in f for f in features))

        spread_rows = [
            {"sentence": "a", "signed_mass": 1.0, "direction": "positive", "attribution": 0.5, "tokens": []},
            {"sentence": "b", "signed_mass": 0.5, "direction": "positive", "attribution": 0.3, "tokens": []},
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

        mixed = aggregate_to_sentences(
            "First of all, politics matter. Second sentence here.",
            [
                {"token": "politics", "shap": 0.8, "start": 13, "end": 21},
                {"token": "matter", "shap": -0.9, "start": 22, "end": 28},
                {"token": "Second", "shap": 0.5, "start": 30, "end": 36},
            ],
        )
        self.assertNotIn("First of all, politics matter.", [r["sentence"] for r in mixed])
        self.assertTrue(all(r["signed_mass"] > 0 for r in mixed))
        self.assertTrue(all(r["direction"] == "positive" for r in mixed))
        net_top = aggregate_to_sentences(
            "Net positive wins. Net negative loses.",
            [
                {"token": "positive", "shap": 0.6, "start": 4, "end": 12},
                {"token": "wins", "shap": 0.1, "start": 13, "end": 17},
                {"token": "negative", "shap": 0.9, "start": 22, "end": 30},
                {"token": "loses", "shap": -1.2, "start": 31, "end": 36},
            ],
        )
        self.assertTrue(all(r["signed_mass"] > 0 for r in net_top))
        self.assertNotIn("Net negative loses.", [r["sentence"] for r in net_top])


if __name__ == "__main__":
    unittest.main()
