"""Specification for the formulae pipeline, pending its extraction.

This suite predates the current refactor and targets a ``scripts.`` package that
was started and lost -- it has never run in this repository. It is kept, and
skipped, because it is a useful statement of what the formulae pipeline is
supposed to expose: the function names below are the API that
``superarc.pipelines.formulae`` should present when 30-1_multiFormula_experiment.py
is moved into the package (Phase 3).

Remove the skip and repoint the import when that lands.
"""

import unittest
from pathlib import Path
import subprocess
import sys

import pytest

import pandas as pd

pytest.importorskip(
    "scripts.multiFormula_analysis",
    reason="pending Phase 3: formulae pipeline not yet extracted into superarc.pipelines",
)

from scripts.multiFormula_analysis import (  # noqa: E402
    get_model_display_name,
    compare_sequences,
    load_data,
    test_model_equivalence,
    compute_equiv_df,
    test_model_accuracy,
    compute_accuracy_by_complexity,
    compute_number_formulas,
    compute_total_formulas_by_complexity,
)


class TestMultiFormulaAnalysis(unittest.TestCase):
    def setUp(self) -> None:
        self.csv_path = Path("multi-formula-time-series.csv")
        self.assertTrue(self.csv_path.exists())
        self.df = load_data(self.csv_path)
        self.models = [
            "chatgpt_4.5",
            "o1_mini",
            "claude_3.7",
            "claude_3.5",
            "o1_preview",
            "gemini",
            "cursor_small",
            "gpt_4o_mini",
            "mistral",
            "qwen",
            "deepseek",
            "llama_4_scout",
            "grok_3",
            "qwen3",
            "chatgpt_5",
            "grok4",
            "deepseek_r1_0528",
            "opus_4",
            "mistral_large2405",
            "gemini_2.5_pro",
            "claude_sonnet_4",
            "meta",
            "gpt_4o",
        ]

    def test_display_name_mapping(self):
        self.assertEqual(get_model_display_name("gpt_4o"), "ChatGPT-4o")
        self.assertEqual(get_model_display_name("chatgpt_5"), "ChatGPT-5")
        self.assertEqual(get_model_display_name("new_model_x"), "New-Model-X")

    def test_compare_sequences(self):
        self.assertTrue(compare_sequences("1, 2, 3", "1,2,3,4,5"))
        self.assertFalse(compare_sequences("1, 2, 4", "1,2,3,4,5"))
        self.assertFalse(compare_sequences("1, 2, 3", "*not found"))

    def test_equivalence_and_accuracy(self):
        import logging
        logger = logging.getLogger("test")
        logger.setLevel(logging.CRITICAL)
        test_model_equivalence(self.df, self.models, logger)
        equiv_df = compute_equiv_df(self.df, self.models)
        self.assertFalse(equiv_df.empty)
        test_model_accuracy(self.df, self.models, logger)
        acc_by_comp, acc_cols = compute_accuracy_by_complexity(self.df)
        self.assertTrue(len(acc_cols) > 0)
        expected_acc = {
            "accuracy-percentage-qwen": [100.00, 46.67, 0.0],
            "accuracy-percentage-grok_3": [96.67, 45.00, 0.0],
            "accuracy-percentage-chatgpt_4.5": [78.89, 16.67, 0.0],
        }
        for col, vals in expected_acc.items():
            if col in acc_cols:
                got = acc_by_comp[col].round(2).tolist()
                self.assertEqual(got, vals)

    def test_formula_counts(self):
        import logging
        logger = logging.getLogger("test2")
        logger.setLevel(logging.CRITICAL)
        formula_cols = compute_number_formulas(self.df, self.models, logger)
        self.assertTrue(len(formula_cols) > 0)
        totals = compute_total_formulas_by_complexity(self.df, formula_cols)
        self.assertIn(1, totals.index)
        self.assertIn(2, totals.index)
        self.assertIn(3, totals.index)

    def test_cli_argument_combinations(self):
        out_dir = Path("out_plots_tests")
        if out_dir.exists():
            for p in out_dir.glob("**/*"):
                p.unlink() if p.is_file() else None
            try:
                out_dir.rmdir()
            except Exception:
                pass
        cmd = [sys.executable, "scripts/multiFormula_analysis.py", "--path", str(out_dir)]
        subprocess.run(cmd, check=True)
        self.assertTrue(out_dir.exists())
        self.assertTrue((out_dir / "equivalence_percentage.png").exists())
        self.assertTrue((out_dir / "accuracy_by_complexity.png").exists())
        self.assertTrue((out_dir / "total_number_formulas.png").exists())
        cmd_v = [sys.executable, "scripts/multiFormula_analysis.py", "--v", "--s", "--path", str(out_dir)]
        subprocess.run(cmd_v, check=True)
        self.assertTrue((out_dir / "logs" / "multiFormula_analysis.log").exists())
        self.assertTrue((out_dir / "equivalence_by_complexity.csv").exists())
        self.assertTrue((out_dir / "accuracy_by_complexity.csv").exists())
        self.assertTrue((out_dir / "total_formulas_by_complexity.csv").exists())


if __name__ == "__main__":
    unittest.main()

