"""Complexity of the formulae the models produced -- Supplementary Figure 1.

Titled "Metrics Comparison Across Models and Complexities for Formulae
generation". Six panels, each a different complexity measure of the formulae
themselves, plotted against the complexity of the sequence they were meant to
reproduce. The point is compression: a model that understood a sequence should
answer with something *shorter* than the sequence, and the panels show that the
answers instead grow with the target.

This lived in ``26_programming_codes.ipynb``, which begins
``from processing_answers import *`` -- a module that instantiates MOMENT-1-large
at import time and so needs ``torch``, none of which this figure uses. The four
functions it actually needs already live in
:mod:`superarc.complexity_measures`, and the 30 kB cell that built its inputs was
25 models x 3 complexities of the same one-line filter, written out by hand. It
is a dict here.

The placeholder, and why the figure is deterministic now
-------------------------------------------------------
28.1% of the plotted cells -- 550 of 1955 -- are ``'***'``, meaning the model
produced no formula at all. (No cell is empty; the original's ``fillna`` was
dead code.)

The original replaced every ``'***'`` with a random 45-character alphanumeric
string. **That is right, and it cannot be skipped.** Left literal, ``'***'`` is
three highly compressible characters, so "no answer" would score as the *best*
answer in the dataset -- and the mismatch is large: leaving it literal lands
15.6% of pixels from the published figure, against 3.3% for a random draw, and
every panel's maximum then falls well short of the published axis (Avg BDM
713 vs ~980, Avg ZIP 60 vs ~72, Avg BDM (LZW) 2425 vs ~2570).

The defect is only that the draw was never seeded, so the published image came
from one arbitrary string nobody recorded. Two different seeds land ~3% of
pixels apart, which is the whole of the gap.

**Resolution (author's ruling, 2026-09-12): average over the draw.** Each panel
is a *mean* of per-answer values, and with one shared string every placeholder
cell carries the same value, so each panel is linear in a single random
quantity: the measure of one 45-character string. Averaging the figure over
infinitely many draws is therefore exactly substituting that measure's
expectation, which is :data:`PLACEHOLDER_EXPECTATION`. Same method, Monte Carlo
error removed rather than frozen at an arbitrary seed, and no seed appears in
the result.

Verified, not assumed:

* the decomposition into per-answer means is exact -- 0.0 difference over 108
  values against the original grouped computation;
* the expectation reproduces the mean of 40 seeded draws to within 0.2% of each
  panel's range, which is the 40-draw average's own error, not the
  expectation's (its standard errors are all under 0.05% of range);
* two runs agree to 0.0 over all 414 values.

The regenerated figure sits **4.6% of pixels from the published one** -- further
than the 3.3% a single lucky seed gives. That is the expected outcome and not a
regression: the published image is one draw from the distribution, and the
expectation is its centre, not that draw. Individual draws sit 3-6% from each
other.

Per-cell independent draws were tried first and rejected on measurement: they
cut the spread only from 3.3% to 1.8% on Avg BDM, because each model and
complexity has just ~10 missing answers, and they made Avg ZIP *worse*
(0.0% -> 0.2%), since a shared string gives every placeholder an identical
compressed length.

The legend says ``grok_4``
--------------------------
The display map in the notebook has a key ``'grok4'`` where the model is
``'grok_4'``, so the lookup falls through to its default and prints the raw key.
It is printed that way in the Supplementary Information, so it is reproduced
that way here. Fixing it is an author's call.
"""

from __future__ import annotations

import random
import string
from pathlib import Path

import numpy as np
import pandas as pd

from . import REPO_ROOT, plots_dir
from .complexity_measures import (
    ascii_to_binary_list,
    average_length_of_strings,
    compress_text,
    list_of_strings_to_binary_lists,
    list_of_strings_to_compressed,
    process_binary_sequences,
)

_BDM = None


def _bdm():
    """One BDM instance, built on first use.

    Constructing it loads the CTM lookup tables, which is far too expensive to
    repeat per string -- and per_string_measures is called tens of thousands of
    times when estimating the placeholder expectation.
    """
    global _BDM
    if _BDM is None:
        from pybdm import BDM
        _BDM = BDM(ndim=1)
    return _BDM

DATASET = REPO_ROOT / "formulas_found_by_llm.csv"

# Stand-in for "no formula produced". Length and alphabet are what the measures
# see; see the module docstring.
PLACEHOLDER_LENGTH = 45
PLACEHOLDER_SEED = 42

COMPLEXITIES = (1, 2, 3)

# Internal key -> column in formulas_found_by_llm.csv. This dataset uses its own
# naming, agreeing with neither the formula nor the script dataset -- which is
# why superarc.registry exists and why this map is kept explicit rather than
# guessed at.
FORMULA_COLUMNS = {
    'chatgpt_4o':          'chatgpt-4o',
    'gpt_4o_mini':         'gpt-4o-mini',
    'gemini_2_5_pro':      'gemini_2.5_pro',
    'gemini_thinking':     'gemini-thinking',
    'claude_3_5_sonnet':   'claude-3.5-sonnet',
    'chatgpt_o1':          'chatgpt-o1',
    'mistral':             'mistral',
    'meta':                'meta',
    'cursor_small':        'cursor_small',
    'o1_mini':             'o1-mini',
    'grok_3':              'grok-3',
    'grok_4':              'grok_4',
    'qwen':                'qwen',
    'deepseek':            'deepseek',
    'chatgpt_4_5':         'chatgpt-4.5',
    'claude_3_7':          'claude_3.7',
    'deepseek_r1_0525':    'deepseek_r1_0525',
    'llama_4_scout':       'llama_4_scout',
    'qwen3':               'qwen3',
    'chatgpt_5':           'chatgpt_5',
    'opus_4':              'opus_4',
    'mistral_large2405':   'mistral_large2405',
    'claude_sonnet_4':     'claude_sonnet_4',
    'grok':                'grok',
    'gemini_1_5_advanced': 'gemini-1.5-advanced',
}

# Plotted, in legend order. `grok` and `gemini_1_5_advanced` are computed but not
# drawn: the notebook comments them out of the model list.
PLOTTED_MODELS = (
    'chatgpt_4o',
    'gpt_4o_mini',
    'gemini_2_5_pro',
    'gemini_thinking',
    'claude_3_5_sonnet',
    'chatgpt_o1',
    'mistral',
    'meta',
    'cursor_small',
    'o1_mini',
    'grok_3',
    'grok_4',
    'qwen',
    'deepseek',
    'chatgpt_4_5',
    'claude_3_7',
    'deepseek_r1_0525',
    'llama_4_scout',
    'qwen3',
    'chatgpt_5',
    'opus_4',
    'mistral_large2405',
    'claude_sonnet_4',
)

DISPLAY_NAMES = {
    'chatgpt_4o':        'ChatGPT-4o',
    'gpt_4o_mini':       'ChatGPT-4o-Mini',
    'gemini_2_5_pro':    'Gemini-2.5-Pro',
    'gemini_thinking':   'Gemini',
    'claude_3_5_sonnet': 'Claude-3.5',
    'chatgpt_o1':        'o1-Preview',
    'mistral':           'Mistral',
    'meta':              'Meta',
    'cursor_small':      'Cursor-Small',
    'o1_mini':           'o1-Mini',
    'grok_3':            'Grok-3',
    'grok4':             'Grok-4',
    'qwen':              'Qwen',
    'deepseek':          'DeepSeek',
    'chatgpt_4_5':       'ChatGPT-4.5',
    'claude_3_7':        'Claude-3.7',
    'deepseek_r1_0525':  'DeepSeek-R1-0528',
    'llama_4_scout':     'Llama-4-Scout',
    'qwen3':             'Qwen-3',
    'chatgpt_5':         'ChatGPT-5',
    'opus_4':            'Claude-Opus-4',
    'mistral_large2405': 'Mistral-Large-2405',
    'claude_sonnet_4':   'Claude-Sonnet-4',
}

# Panel order, left to right then down.
METRICS = (
    ("avg_bdm", "Avg BDM"),
    ("avg_shannon", "Avg Shannon"),
    ("avg_lzw", "Avg LZW"),
    ("avg_zip", "Avg ZIP"),
    ("avg_bdm_lzw", "Avg BDM (LZW)"),
    ("avg_bdm_zip", "Avg BDM (ZIP)"),
)

MARKERS = ("o", "s", "^", "D", "x", "v", "*", "p", "h", "+", "1", "2", "3", "4")


def display_name(key: str) -> str:
    """Legend label. Falls through to the raw key, as published."""
    return DISPLAY_NAMES.get(key, key)


def load_formulas(seed: int | None = PLACEHOLDER_SEED) -> pd.DataFrame:
    """Load the formulae, substituting the placeholder for missing answers."""
    rng = random.Random(seed)

    def placeholder() -> str:
        alphabet = string.ascii_letters + string.digits
        return "".join(rng.choice(alphabet) for _ in range(PLACEHOLDER_LENGTH))

    frame = pd.read_csv(DATASET)
    # One draw for all '***', a second for all empty -- as published, not one
    # per cell.
    frame = frame.replace("***", placeholder())
    return frame.fillna(placeholder())


def metrics_for(sequences: list[str]) -> dict[str, float]:
    """The six measures over one model's formulae at one complexity."""
    _, avg_bdm, avg_shannon = process_binary_sequences(
        [list_of_strings_to_binary_lists(sequences)], normalize=False
    )
    lzw_compressed, zip_compressed = list_of_strings_to_compressed(sequences)
    _, avg_bdm_lzw, _ = process_binary_sequences(
        [list_of_strings_to_binary_lists(lzw_compressed)], normalize=False
    )
    _, avg_bdm_zip, _ = process_binary_sequences(
        [list_of_strings_to_binary_lists(zip_compressed)], normalize=False
    )
    return {
        "avg_bdm": float(avg_bdm[0]),
        "avg_shannon": float(avg_shannon[0]),
        "avg_lzw": average_length_of_strings(lzw_compressed),
        "avg_zip": average_length_of_strings(zip_compressed),
        "avg_bdm_lzw": float(avg_bdm_lzw[0]),
        "avg_bdm_zip": float(avg_bdm_zip[0]),
    }


PLACEHOLDER = "***"


def placeholder_census(frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """How many answers are missing, per model and complexity.

    This is the quantity that decides how much the placeholder matters, so it is
    worth being able to look at rather than take on trust.
    """
    frame = pd.read_csv(DATASET) if frame is None else frame
    rows = []
    for model in PLOTTED_MODELS:
        column = FORMULA_COLUMNS[model]
        for complexity in COMPLEXITIES:
            values = frame.loc[frame["complexity"] == complexity, column]
            missing = (values.astype(str).str.strip() == PLACEHOLDER).sum()
            rows.append({
                "model": model,
                "complexity": complexity,
                "answers": len(values),
                "missing": int(missing),
            })
    census = pd.DataFrame(rows)
    census["missing %"] = (100 * census["missing"] / census["answers"]).round(1)
    return census


def per_string_measures(text: str) -> dict[str, float]:
    """The six measures for a single answer.

    Each panel is a *mean* of these over a model's answers, which is what makes
    the placeholder tractable -- see :data:`PLACEHOLDER_EXPECTATION`.
    """
    bits = np.array(ascii_to_binary_list(text))
    lzw_text, zip_text = compress_text(text)
    bdm = _bdm()
    return {
        "avg_bdm": bdm.bdm(bits),
        "avg_shannon": bdm.ent(bits),
        "avg_lzw": float(len(lzw_text)),
        "avg_zip": float(len(zip_text)),
        "avg_bdm_lzw": bdm.bdm(np.array(ascii_to_binary_list(lzw_text))),
        "avg_bdm_zip": bdm.bdm(np.array(ascii_to_binary_list(zip_text))),
    }


def estimate_placeholder_expectation(draws: int = 20_000, seed: int = 0) -> pd.DataFrame:
    """Expected measures of a random placeholder string, with standard errors.

    This is the function that produced :data:`PLACEHOLDER_EXPECTATION`. Kept so
    the constant is reproducible rather than asserted, and so the tests can
    check it still holds.
    """
    rng = random.Random(seed)
    alphabet = string.ascii_letters + string.digits
    table = pd.DataFrame([
        per_string_measures("".join(rng.choice(alphabet) for _ in range(PLACEHOLDER_LENGTH)))
        for _ in range(draws)
    ])
    return pd.DataFrame({
        "expectation": table.mean(),
        "sd": table.std(ddof=1),
        "standard error": table.std(ddof=1) / np.sqrt(draws),
    })


# Expected measures of one random 45-character alphanumeric placeholder.
#
# Produced by estimate_placeholder_expectation(draws=20_000, seed=0). Standard
# errors were 0.18, 0.00035, 0.0, 0.0013, 0.43 and 0.30 -- under 0.05% of the
# range its panel spans in every case, so the residual Monte Carlo error is far
# below one line width.
#
# avg_lzw carries no error at all: LZMA of any 45-character alphanumeric string
# base64-encodes to exactly 140 characters. Measured sd was exactly 0 over
# 20,000 draws.
PLACEHOLDER_EXPECTATION = {
    "avg_bdm": 958.5531,
    "avg_shannon": 4.8681,
    "avg_lzw": 140.0,
    "avg_zip": 71.9910,
    "avg_bdm_lzw": 2556.5311,
    "avg_bdm_zip": 1511.0446,
}


def metrics_with_placeholders(answers: list[str]) -> dict[str, float]:
    """The six measures, with '***' answers held at their expected value.

    '***' means the model produced no formula. The original replaced it with a
    random 45-character string so that "no answer" scores as *incompressible*
    rather than as a very short, highly compressible one. That is correct, and
    is why the placeholder cannot simply be dropped -- leaving '***' literal
    makes a missing answer the best-compressing answer in the dataset.

    But the draw was never seeded, so the published image is unrecoverable.
    Every panel is a mean of per-answer values, so it is linear in the
    placeholder's measure, and averaging the figure over infinitely many draws
    is exactly substituting that measure's expectation. That is what happens
    here: the same method, with its Monte Carlo error removed rather than frozen
    at one arbitrary seed.
    """
    if not answers:
        return dict.fromkeys(PLACEHOLDER_EXPECTATION, 0.0)

    real = [a for a in answers if str(a).strip() != PLACEHOLDER]
    missing = len(answers) - len(real)

    totals = {k: v * missing for k, v in PLACEHOLDER_EXPECTATION.items()}
    for answer in real:
        for key, value in per_string_measures(str(answer)).items():
            totals[key] += value
    return {key: total / len(answers) for key, total in totals.items()}


def compute(frame: pd.DataFrame | None = None, models=PLOTTED_MODELS) -> pd.DataFrame:
    """Long-format table: one row per model, complexity and measure.

    Reads the dataset as it stands, placeholders and all, and holds them at
    their expectation. Pass a frame from :func:`load_formulas` to reproduce a
    single seeded draw instead -- that frame has no '***' left in it, so the
    per-answer path is used.
    """
    frame = pd.read_csv(DATASET) if frame is None else frame
    already_drawn = not (frame == PLACEHOLDER).any().any()
    records = []
    for model in models:
        column = FORMULA_COLUMNS[model]
        for complexity in COMPLEXITIES:
            answers = frame.loc[frame["complexity"] == complexity, column].tolist()
            measures = metrics_for(answers) if already_drawn else metrics_with_placeholders(answers)
            for measure, value in measures.items():
                records.append({
                    "model": model,
                    "complexity": complexity,
                    "measure": measure,
                    "value": value,
                })
    return pd.DataFrame.from_records(records)


def placeholder_sensitivity(seeds=(0, 1, 42, 1234, 99999)) -> pd.DataFrame:
    """How much the unseeded placeholder draw could have moved each measure.

    Returns, per measure, the spread across seeds as a fraction of the range the
    panel spans. The published figure was drawn with an unknown seed, so this is
    the honest statement of how close any regeneration can get.
    """
    runs = [compute(load_formulas(seed)).assign(seed=seed) for seed in seeds]
    combined = pd.concat(runs, ignore_index=True)
    grouped = combined.groupby("measure")["value"]
    spread = (
        combined.groupby(["measure", "model", "complexity"])["value"]
        .agg(lambda values: values.max() - values.min())
        .groupby("measure")
        .max()
    )
    return pd.DataFrame({
        "max spread across seeds": spread,
        "panel range": grouped.max() - grouped.min(),
        "as % of panel": 100 * spread / (grouped.max() - grouped.min()),
    })


def output_dir() -> Path:
    """Where figures are written. See ``superarc.plots_dir``."""
    return plots_dir()


def plot(table: pd.DataFrame | None = None, out_dir: Path | None = None) -> list[Path]:
    import matplotlib.pyplot as plt
    import seaborn as sns

    table = compute() if table is None else table
    out_dir = out_dir or output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="darkgrid")
    sns.set_palette("husl")

    fig, axes = plt.subplots(3, 2, figsize=(20, 20))
    fig.suptitle(
        "Metrics Comparison Across Models and Complexities for Formulae generation",
        fontsize=35,
    )

    for index, (measure, label) in enumerate(METRICS):
        ax = axes[index // 2, index % 2]
        ax.set_facecolor("none")
        ax.grid(True, color="gray")
        for position, model in enumerate(PLOTTED_MODELS):
            series = table[(table["model"] == model) & (table["measure"] == measure)]
            series = series.sort_values("complexity")
            ax.plot(
                series["complexity"],
                series["value"],
                label=display_name(model),
                marker=MARKERS[position % len(MARKERS)],
                markersize=15,
                linestyle="-",
                linewidth=3,
            )
        ax.set_ylabel(label, fontsize=27)
        ax.set_title(label, fontsize=36)
        ax.set_xlabel("Complexity", fontsize=27)
        ax.set_xticks(list(COMPLEXITIES))
        ax.set_xticklabels(list(COMPLEXITIES), fontsize=22)
        ax.tick_params(axis="y", labelsize=22)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=20, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    fig.subplots_adjust(top=0.93, right=0.85)

    written = []
    for suffix in (".pdf", ".svg", ".png"):
        path = out_dir / ("figure10" + suffix)
        fig.savefig(path, bbox_inches="tight")
        written.append(path)
    plt.close(fig)
    return written


def main() -> int:
    table = compute()
    for path in plot(table):
        if path.suffix == ".png":
            print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
