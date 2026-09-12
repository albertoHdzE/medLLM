"""Check the model registry against the real datasets.

The registry claims that a given column in a given file is a given model. That
claim is only worth something if it is checked against the files, so these tests
open them.
"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from superarc import REPO_ROOT  # noqa: E402
from superarc.registry import (  # noqa: E402
    A_FORMULA,
    B_SCRIPT,
    C_SERIES,
    MODELS,
    RETIRED_COLUMNS,
    UnknownColumnError,
    by_column,
    columns_for,
    display_name,
    models_for,
    sorted_by_family,
    validate_columns,
)

STRUCTURAL = {"sequence", "Complexity", "complexity"}


def load(name: str, encoding: str) -> pd.DataFrame:
    return pd.read_csv(REPO_ROOT / name, encoding=encoding, low_memory=False)


class TestRegistryShape(unittest.TestCase):
    def test_names_are_unique(self):
        names = [m.name for m in MODELS]
        self.assertEqual(len(names), len(set(names)))

    def test_no_two_models_share_a_column_in_a_dataset(self):
        for dataset in (A_FORMULA, B_SCRIPT, C_SERIES):
            cols = columns_for(dataset)
            self.assertEqual(len(cols), len(set(cols)), f"duplicate column in {dataset}")

    def test_family_order_is_dense_and_zero_based(self):
        by_family: dict[str, list[int]] = {}
        for m in MODELS:
            by_family.setdefault(m.family, []).append(m.order)
        for family, orders in by_family.items():
            self.assertEqual(sorted(orders), list(range(len(orders))), family)

    def test_every_model_is_sortable_by_family(self):
        self.assertEqual(len(sorted_by_family()), len(MODELS))


class TestAgainstFormulaDataset(unittest.TestCase):
    """multi-formula-time-series.csv -- published Figures 3 and 4."""

    @classmethod
    def setUpClass(cls):
        cls.df = load("multi-formula-time-series.csv", "latin1")

    def test_every_registered_column_exists(self):
        missing = [c for c in columns_for(A_FORMULA) if c not in self.df.columns]
        self.assertEqual(missing, [], f"registered but absent: {missing}")

    def test_no_unexplained_model_columns(self):
        # Derived columns look like <alias>_<n> or <alias>_<n>_eval.
        aliases = set(columns_for(A_FORMULA)) | set(RETIRED_COLUMNS.get(A_FORMULA, {}))
        unexplained = []
        for col in self.df.columns:
            if col in STRUCTURAL or col in aliases:
                continue
            m = re.match(r"^(.*?)_\d+(_eval)?$", col)
            if m and m.group(1) in aliases:
                continue
            unexplained.append(col)
        self.assertEqual(unexplained, [], f"columns matching no registered model: {unexplained}")

    def test_gemini_columns_are_distinct_models(self):
        self.assertEqual(by_column(A_FORMULA, "gemini").name, "Gemini")
        self.assertEqual(by_column(A_FORMULA, "gemini_2.5_pro").name, "Gemini-2.5-Pro")


class TestAgainstScriptDataset(unittest.TestCase):
    """multi-python-script-time-series.csv -- published Figures 5 and 6.

    This is the dataset where the two conflicting maps lived.
    """

    @classmethod
    def setUpClass(cls):
        cls.df = load("multi-python-script-time-series.csv", "utf-8")

    def test_validate_columns_accepts_the_real_file(self):
        model_cols = [c for c in self.df.columns if c not in STRUCTURAL]
        validate_columns(B_SCRIPT, model_cols)  # must not raise

    def test_gemini_25_pro_is_the_gemini_25_pro_column(self):
        # The ruling: Figure 5 was right, Figure 6 was wrong.
        self.assertEqual(by_column(B_SCRIPT, "gemini-2.5-pro").name, "Gemini-2.5-Pro")
        self.assertEqual(by_column(B_SCRIPT, "gemini-thinking").name, "Gemini")

    def test_the_superseded_gemini_column_is_declared_retired(self):
        self.assertIn("gemini", RETIRED_COLUMNS[B_SCRIPT])
        self.assertIsNone(by_column(B_SCRIPT, "gemini"))
        # ...and it really is still in the file, so the declaration is load-bearing.
        self.assertIn("gemini", self.df.columns)

    def test_grok_columns_lag_the_model_versions_by_one(self):
        self.assertEqual(by_column(B_SCRIPT, "grok").name, "Grok-3")
        self.assertEqual(by_column(B_SCRIPT, "grok-3").name, "Grok-4")

    def test_unknown_column_fails_loudly(self):
        cols = [c for c in self.df.columns if c not in STRUCTURAL] + ["gpt-6-turbo"]
        with self.assertRaises(UnknownColumnError) as ctx:
            validate_columns(B_SCRIPT, cols)
        self.assertIn("gpt-6-turbo", str(ctx.exception))

    def test_missing_registered_column_fails_loudly(self):
        cols = [c for c in self.df.columns if c not in STRUCTURAL and c != "qwen3"]
        with self.assertRaises(UnknownColumnError):
            validate_columns(B_SCRIPT, cols)


class TestAgainstSeriesDataset(unittest.TestCase):
    """seriesWithLLMs_ext_Dic2025.csv -- Table 1 and Figures 7-10."""

    @classmethod
    def setUpClass(cls):
        cls.df = load("seriesWithLLMs_ext_Dic2025.csv", "latin-1")

    def test_every_model_has_the_four_columns_phi_needs(self):
        needed = ("-formula", "-formula-correctness", "-formula-ordinal", "-formula-copy_seq")
        for model in models_for(C_SERIES):
            col = model.column(C_SERIES)
            for suffix in needed:
                self.assertIn(f"{col}{suffix}", self.df.columns, f"{model.name}{suffix}")

    def test_model_count_matches_published_table_1(self):
        # Table 1 ranks 28 models plus the AIXI/BDM/CTM reference row.
        self.assertEqual(len(models_for(C_SERIES)), 28)


class TestPublishedLabels(unittest.TestCase):
    def test_deepseek_r1_published_under_two_labels(self):
        # Table 1 and Figures 7-10 say 0528; Figures 3 and 4 said 0525.
        self.assertEqual(display_name(C_SERIES, "deepseek_r1_0528"), "DeepSeek-R1-0528")
        self.assertEqual(
            display_name(A_FORMULA, "deepseek_r1_0528", published=True),
            "DeepSeek-R1-0525",
        )

    def test_canonical_name_is_used_when_not_asking_for_published(self):
        self.assertEqual(display_name(A_FORMULA, "deepseek_r1_0528"), "DeepSeek-R1-0528")


if __name__ == "__main__":
    unittest.main()
