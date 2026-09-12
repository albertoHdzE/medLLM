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

Two things preserved deliberately
---------------------------------
**The '***' and NaN placeholder.** Where a model produced no formula the cell is
``'***'`` or empty, and the original replaced both with a random 45-character
alphanumeric string -- one string for all ``'***'`` cells, a second for all empty
ones -- so that "no answer" would score as incompressible rather than as zero.
The draw was unseeded, so the published figure cannot be recovered exactly. It is
seeded now.

:func:`placeholder_sensitivity` measures what that draw is worth, and it is not
negligible. Across five seeds, as a fraction of the range each panel spans:

    Avg LZW          0.0%      exactly reproducible
    Avg ZIP          0.0%      exactly reproducible
    Avg Shannon      3.3%
    Avg BDM (ZIP)    3.7%
    Avg BDM          7.4%
    Avg BDM (LZW)   10.6%

The two compressed-length panels do not move at all, because the compressed
length of a random 45-character string does not depend on which characters were
drawn. The BDM panels do, because BDM is sensitive to the exact bit pattern --
which is precisely why it is the measure the paper is built on.

So four of the six panels are reproducible only up to the draw. Regenerating with
seed 42 lands 3.3% of pixels away from the published figure, and two different
seeds land 3.1% from each other, so the draw accounts for the whole gap and
nothing else has changed. The trend every panel shows -- complexity of the
answers rising with complexity of the target -- is far larger than this wobble
and is unaffected. Choosing a reference seed is an author's decision, recorded
in ``superarc.parity``.

**The legend says ``grok_4``.** The display map in the notebook has a key
``'grok4'`` where the model is ``'grok_4'``, so the lookup falls through to its
default and prints the raw key. It is printed that way in the Supplementary
Information, so it is reproduced that way here. Fixing it is an author's call.
"""

from __future__ import annotations

import random
import string
from pathlib import Path

import pandas as pd

from . import REPO_ROOT, plots_dir
from .complexity_measures import (
    average_length_of_strings,
    list_of_strings_to_binary_lists,
    list_of_strings_to_compressed,
    process_binary_sequences,
)

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


def compute(frame: pd.DataFrame | None = None, models=PLOTTED_MODELS) -> pd.DataFrame:
    """Long-format table: one row per model, complexity and measure."""
    frame = load_formulas() if frame is None else frame
    records = []
    for model in models:
        column = FORMULA_COLUMNS[model]
        for complexity in COMPLEXITIES:
            answers = frame.loc[frame["complexity"] == complexity, column].tolist()
            for measure, value in metrics_for(answers).items():
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
