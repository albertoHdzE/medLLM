"""Check the computed benchmark ranking against Table 1 as published.

Table 1 was never emitted by code -- the script built the dataframe and then
evaluated np.round(...) as a bare expression, a no-op outside a notebook -- so
the published table had never been verified against the data until now.

The expected values below are transcribed from doc/nature-paper.pdf, Table 1.
If one of these fails, either the data changed or the metric changed. Neither
should happen quietly.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from superarc import table1  # noqa: E402

# Model -> (rho1, rho2, rho3, rho4, delta1, delta2, delta3, phi), as printed.
PUBLISHED = {
    "AIXI/BDM/CTM":     (1.000, 0.00, 0.0, 0.000, 1.000, 0.000, 0.000, 1.000),
    "ChatGPT-4.5":      (0.000, 1.00, 0.0, 0.000, 0.000, 0.419, 0.000, 0.042),
    "o1-Mini":          (0.000, 0.64, 0.0, 0.360, 0.000, 0.537, 0.000, 0.034),
    "Claude-3.7":       (0.000, 0.81, 0.0, 0.190, 0.000, 0.407, 0.000, 0.033),
    "Claude-3.5":       (0.060, 0.14, 0.0, 0.800, 0.449, 0.428, 0.000, 0.033),
    "o1-Preview":       (0.000, 0.29, 0.0, 0.710, 0.000, 0.423, 0.000, 0.012),
    "Gemini":           (0.000, 0.00, 1.0, 0.000, 0.000, 0.000, 0.762, 0.008),
    "Cursor-Small":     (0.000, 0.00, 1.0, 0.000, 0.000, 0.000, 0.762, 0.008),
    "ChatGPT-4o-Mini":  (0.000, 0.00, 1.0, 0.000, 0.000, 0.000, 0.762, 0.008),
    "Mistral":          (0.000, 0.00, 1.0, 0.000, 0.000, 0.000, 0.710, 0.007),
    "Qwen":             (0.000, 0.00, 1.0, 0.000, 0.000, 0.000, 0.710, 0.007),
    "DeepSeek":         (0.000, 0.00, 1.0, 0.000, 0.000, 0.000, 0.710, 0.007),
    "Llama-4-Scout":    (0.010, 0.00, 0.0, 0.990, 0.450, 0.000, 0.000, 0.004),
    "Grok-3":           (0.000, 0.02, 0.0, 0.980, 0.000, 0.318, 0.000, 0.001),
}

# Every remaining model scored zero on every column except rho4 = 1.
PUBLISHED_ALL_ZERO = {
    "Mistral-Large-3", "Meta", "Gemini-3-Pro", "Claude-4.5", "ChatGPT-5.2",
    "Grok-4.1", "ChatGPT-4o", "Grok-4", "Claude-Sonnet-4", "Gemini-2.5-Pro",
    "Mistral-Large-2405", "Claude-Opus-4", "DeepSeek-R1-0528", "Qwen-3",
    "ChatGPT-5",
}

COLUMNS = ("rho1", "rho2", "rho3", "rho4", "delta1", "delta2", "delta3", "phi")


class TestTable1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table = table1.compute()
        cls.rows = {r["Model"]: r for _, r in cls.table.iterrows()}

    def test_every_published_model_is_present(self):
        expected = set(PUBLISHED) | PUBLISHED_ALL_ZERO
        self.assertEqual(set(self.rows), expected)

    def test_values_match_the_published_table(self):
        for model, values in PUBLISHED.items():
            row = self.rows[model]
            for column, expected in zip(COLUMNS, values):
                self.assertAlmostEqual(
                    row[column], expected, places=2,
                    msg=f"{model}.{column}: published {expected}, computed {row[column]:.4f}",
                )

    def test_models_that_scored_zero_still_score_zero(self):
        for model in PUBLISHED_ALL_ZERO:
            row = self.rows[model]
            self.assertAlmostEqual(row["rho4"], 1.0, places=3, msg=model)
            self.assertAlmostEqual(row["phi"], 0.0, places=3, msg=model)

    def test_rho_is_a_distribution(self):
        for model, row in self.rows.items():
            total = sum(row[c] for c in ("rho1", "rho2", "rho3", "rho4"))
            self.assertAlmostEqual(total, 1.0, places=6, msg=f"{model}: rho sums to {total}")

    def test_ranking_is_ordered_by_phi(self):
        phis = list(self.table["phi"])
        self.assertEqual(phis, sorted(phis, reverse=True))

    def test_the_reference_row_tops_the_ranking(self):
        self.assertEqual(self.table.iloc[0]["Model"], "AIXI/BDM/CTM")
        self.assertAlmostEqual(self.table.iloc[0]["phi"], 1.0, places=6)

    def test_best_llm_is_two_orders_of_magnitude_below_the_reference(self):
        # The paper's central claim, as a number: no model approaches optimal
        # abstraction and prediction.
        best_llm = self.table[self.table["Model"] != "AIXI/BDM/CTM"]["phi"].max()
        self.assertLess(best_llm, 0.05)


class TestPhiMetric(unittest.TestCase):
    def test_perfect_compressor_scores_one(self):
        import numpy as np
        self.assertAlmostEqual(
            table1.phi(np.array([1.0, 0, 0, 0]), np.array([1.0, 0, 0])), 1.0
        )

    def test_prints_are_worth_a_hundredth_of_real_answers(self):
        import numpy as np
        real = table1.phi(np.array([1.0, 0, 0, 0]), np.array([1.0, 0, 0]))
        prints = table1.phi(np.array([0, 0, 1.0, 0]), np.array([0, 0, 1.0]))
        self.assertAlmostEqual(real / prints, 100.0)

    def test_harmonic_mean_of_nothing_is_zero_not_an_error(self):
        import numpy as np
        self.assertEqual(table1.harmonic_mean(np.array([])), 0.0)

    def test_harmonic_mean_is_dominated_by_the_worst_case(self):
        import numpy as np
        values = np.array([1.0, 1.0, 1.0, 0.01])
        self.assertLess(table1.harmonic_mean(values), np.mean(values) / 2)


if __name__ == "__main__":
    unittest.main()
