"""SuperARC-seq: the benchmark score, its ranking, and the figures built on it.

    Ranking by SuperARC-seq                          -> figure05
    Percentages by output types: p1..p4              -> figure06
    Test scores by type of sequence                  -> figure07
    Bootstrap over 25/50/75/100 sequences            -> figure08

This is the part of the paper that does not ask whether a model got the answer
right. It asks whether the answer was a *model* of the sequence or a restatement
of it. An answer that reprints the sequence is correct and worth almost nothing;
an answer that compresses it is worth a hundred times as much. That ratio is the
weighting ``(1, 0.1, 0.01)`` in :data:`superarc.table1.CLASS_WEIGHTS`.

The metric itself lives in :mod:`superarc.table1`, which owns it. This module
owns the four figures and nothing else.

Why this module exists
----------------------
``34-2_S-ARC_ext.py`` inlined the same score **four times** -- once for Table 1
and the ranking, once for the p-only version behind the probabilities figure,
once for the phi-by-sequence-type figure, and once inside the bootstrap's inner
loop. Four copies of a benchmark's definition means a change to the benchmark had
to be made in four places or in none. They are now one call to
:func:`superarc.table1.score`, which was checked against all four before they
were removed.

The script also carried its own 28-entry display map. Checked elementwise against
the registry before deletion: 28 of 28 agreed, so it was pure duplication.

``MODEL_ORDER`` stays here, for the same reason it does in
:mod:`superarc.formulae` and :mod:`superarc.scripts`: it is not model identity.
Here it is load-bearing in a way that is easy to miss -- the probabilities and
phi figures reshape a flat array by model count and then *assign* this list as
the index, so reordering it would silently relabel every point.

Two things preserved as published
---------------------------------
The reference row is labelled ``ASI`` in the ranking figure and
``AIXI/BDM/CTM`` in Table 1. Same row, two labels in print; both kept.

``rho2`` and ``rho3`` are not exclusive -- an answer that is both ordinal and a
verbatim copy is counted in both -- so the four class counts can exceed the
number of sequences, and ``rho`` is normalised by their total rather than by the
row count. See :func:`superarc.table1.class_masks`.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import altair as alt  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from . import plots_dir  # noqa: E402
from .registry import C_SERIES, display_name  # noqa: E402
from .table1 import DATASET, score  # noqa: E402

N_JOBS = min(8, os.cpu_count() or 1)

# Row ranges of the dataset, in the order the file stores them. The first 100 are
# the binary sequences Table 1 and the ranking are computed over; the rest are
# three families of integer sequences that only the by-type figures use.
SEQUENCE_SETS: dict[str, slice] = {
    "Binary": slice(0, 100),
    "Integers Type 1": slice(100, 130),
    "Integers Type 2": slice(130, 160),
    "Integers Type 3": slice(160, None),
}

# Legend, colour and -- critically -- reshape order. See the module docstring.
MODEL_ORDER = [
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
    "grok_4.1",
    "gpt-5.2",
    "claude-4.5",
    "gemini-3-pro",
    "mistral-large-3",
]

PROB_LABELS = ("p\u2081", "p\u2082", "p\u2083", "p\u2084")
PROB_COLOURS = ('#96ceb4', '#ffcc5c', '#ff6f69', '#da680f')
PHI_LABEL = "\U0001d711"

ALT_AXIS_LABEL_FONT = 16
ALT_AXIS_TITLE_FONT = 18
ALT_LEGEND_LABEL_FONT = 16
ALT_LEGEND_TITLE_FONT = 18

FIG08_TITLE_FONT = 20
FIG08_AXIS_LABEL_FONT = 17
FIG08_TICK_FONT = 15
FIG08_LEGEND_FONT = 15
FIG08_ANNOT_FONT = 14

# The reference row of the ranking figure: an optimal compressor answers every
# sequence correctly, never by copying or index-mapping, and compresses fully.
# Table 1 prints the same row as "AIXI/BDM/CTM"; the figure printed "ASI".
REFERENCE_ROW = {
    "Model": "ASI", "p1": 1, "p2": 0, "p3": 0, "p4": 0,
    "r1": 1, "r2": 0, "r3": 1, "tst": 1,
}


def get_model_display_name(model_key: str) -> str:
    """Canonical label for a column of the series dataset.

    Delegates to the registry. The script's own 28-entry map agreed with it on
    all 28 before being removed.
    """
    return display_name(C_SERIES, model_key)


def load(path: Path | None = None) -> pd.DataFrame:
    """The series dataset: 100 binary sequences followed by three integer sets."""
    return pd.read_csv(path or DATASET, encoding="latin-1", low_memory=False)


def subset(df: pd.DataFrame, name: str) -> pd.DataFrame:
    return df.iloc[SEQUENCE_SETS[name], :]


# ---- the score, per model and per sequence set -------------------------------


def ranking(df: pd.DataFrame, models=None, reference: bool = True) -> pd.DataFrame:
    """Every model's score over the binary sequences, best first.

    The same numbers as :func:`superarc.table1.compute`, in this figure's column
    names and with this figure's reference-row label.
    """
    model_names = MODEL_ORDER if models is None else models
    binary = subset(df, "Binary")

    rows = []
    for column in model_names:
        rho, delta, value = score(binary, column)
        rows.append([get_model_display_name(column), *rho, *delta, value])

    frame = pd.DataFrame(rows, columns=["Model", "p1", "p2", "p3", "p4",
                                        "r1", "r2", "r3", "tst"])
    if reference:
        frame = pd.concat([frame, pd.DataFrame([REFERENCE_ROW])], ignore_index=True)
    return frame.sort_values(by="tst", ascending=False)


def proportions_by_type(df: pd.DataFrame, models=None) -> pd.DataFrame:
    """p1..p4 for every model on every sequence set, long form.

    Only the class proportions; the compression term is not needed here, and the
    published script computed it anyway and threw it away.
    """
    model_names = MODEL_ORDER if models is None else models
    records = []
    for label in SEQUENCE_SETS:
        part = subset(df, label)
        for column in model_names:
            rho, _delta, _value = score(part, column)
            records.append([column, *rho, label])
    df_stack = pd.DataFrame(records, columns=["Model", "p1", "p2", "p3", "p4", "Type"])

    # CORRECTED, author's ruling 2026-09-14. See RESHAPE_ORDER.
    frames = []
    for column, label in zip(["p1", "p2", "p3", "p4"], PROB_LABELS):
        wide = _wide(df_stack, column, model_names)
        frames.append(_prep(wide, label, "Prob"))

    df_final = pd.concat(frames, ignore_index=True)
    df_final['c1'] = df_final['c1'].apply(get_model_display_name)
    return df_final


# The one-character defect, and why it is worth this much prose.
#
# Both by-type figures flatten a column of `len(SEQUENCE_SETS) * len(models)`
# values, built set-major by the loop above, reshape it, transpose it, and then
# *assign* MODEL_ORDER as the index. The reshape is what decides which model each
# row belongs to, and nothing downstream can detect a mistake in it: the figure
# plots either way, and every number in it is a real number that some model
# really scored.
#
# The published phi figure reshaped in C order, which is right. The published
# p1-p4 figure reshaped with order='F', which fills column by column, so the
# point drawn for model j and sequence set i was flat[i + 4*j] -- four
# CONSECUTIVE entries of a set-major array, which is four different models'
# values for one set, not one model's values for four sets.
#
# Confirmed on all 448 points before correcting. 173 of them showed a number
# belonging to a different model or a different sequence set; the rest coincided
# because many class proportions are zero or repeat across models. ChatGPT-4o's
# row, for instance, read p1 = 0.000 on every sequence set, when on Integers
# Type 1 its true p1 is 1.000 -- every answer correct.
#
# Nothing else in this module used that reshape: Table 1, the ranking figure and
# the bootstrap each compute from `score` directly.
RESHAPE_ORDER = "C"


def _wide(df_stack, column, model_names):
    """One column of the set-major stack as a model-by-sequence-set frame."""
    flat = df_stack.loc[:, column].to_numpy()
    wide = pd.DataFrame(flat.reshape(len(flat) // len(model_names), len(model_names),
                                     order=RESHAPE_ORDER)).T
    wide.index = list(model_names)
    wide.columns = list(SEQUENCE_SETS)
    return wide


def scores_by_type(df: pd.DataFrame, models=None) -> pd.DataFrame:
    """phi for every model on every sequence set, long form."""
    model_names = MODEL_ORDER if models is None else models
    records = []
    for label in SEQUENCE_SETS:
        part = subset(df, label)
        for column in model_names:
            _rho, delta, value = score(part, column)
            records.append([column, *delta, value, label])
    df_stack_2 = pd.DataFrame(records, columns=["Model", "r1", "r2", "r3", "tst", "Type"])

    # Same reshape as the probabilities figure, which is the point: the published
    # script wrote it twice and got it right here and wrong there.
    wide = _wide(df_stack_2, "tst", model_names)
    df_final_2 = pd.concat([_prep(wide, PHI_LABEL, "Metric")], ignore_index=True)
    df_final_2['c1'] = df_final_2['c1'].apply(get_model_display_name)
    return df_final_2


def _prep(df, name, column):
    """Wide model-by-type frame to long form.

    The script defined this twice, as ``prep_df``, the second shadowing the
    first; they differed only in the name of the label column.
    """
    df = df.stack().reset_index()
    df.columns = ['c1', 'c2', 'values']
    df[column] = name
    return df


# ---- figures -----------------------------------------------------------------


def _out(out_dir) -> Path:
    path = Path(out_dir) if out_dir else plots_dir()
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_altair_chart(chart, file_base, out_dir=None):
    import importlib.util

    if importlib.util.find_spec("vl_convert") is None:
        raise ValueError(
            "Saving Altair charts as PDF/PNG requires vl-convert-python. "
            "Install with: python -m pip install vl-convert-python"
        )
    out = _out(out_dir)
    alt.data_transformers.enable("default")
    chart.save(str(out / f"{file_base}.pdf"))
    chart.save(str(out / f"{file_base}.png"), scale_factor=3)


def plot_ranking(df_ranking2, out_dir=None):
    """Ranking by SuperARC-seq.

    The script drew this figure twice: once on a log y-axis, which it never
    saved, and then again on a linear axis, which it did. Only the saved one is
    here; the log version was dead code that leaked an open figure.
    """
    out = _out(out_dir)
    plt.rcdefaults()
    plt.figure(figsize=(10, 6))
    # Sort values in descending order
    sorted_df = df_ranking2.sort_values('tst', ascending=False)
    # Create bar plot using Model column with default color cycle
    bars = plt.bar(range(len(sorted_df)), sorted_df['tst'])
    # Set different colors for each bar using default color cycle
    for i, bar in enumerate(bars):
        bar.set_color(f'C{i}')
    plt.grid(True)
    plt.title('Ranking by SuperARC-seq')
    plt.xlabel('Model')
    plt.ylabel('SuperARC-seq')

    # Set x-tick labels with model names
    plt.xticks(range(len(sorted_df)), sorted_df['Model'], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(out / "figure05.pdf", bbox_inches="tight")
    plt.savefig(out / "figure05.png", bbox_inches="tight", dpi=600)
    plt.close()


def plot_probabilities(df_final, out_dir=None):
    """Percentages by output types: p1..p4, faceted by sequence set."""
    prob_colors = alt.Scale(domain=list(PROB_LABELS), range=list(PROB_COLOURS))

    prob_colors = alt.Scale(domain=["p₁", "p₂", "p₃", "p₄"], range=['#96ceb4', '#ffcc5c', '#ff6f69', '#da680f'])

    plot06 = alt.Chart(df_final).mark_circle(size=80, opacity=0.8).encode(
        y=alt.Y('c1:N', title=None, sort=alt.EncodingSortField(field='values', op='sum', order='descending')),
        x=alt.X('values:Q', title="Probability", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color('Prob:N', scale=prob_colors, title="Probability Type"),
        column=alt.Column('c2:N', title="", header=alt.Header(labelFontSize=ALT_AXIS_LABEL_FONT, titleFontSize=ALT_AXIS_TITLE_FONT))
    ).properties(
        width=220, height=620
    ).configure_axis(
        labelFontSize=ALT_AXIS_LABEL_FONT, titleFontSize=ALT_AXIS_TITLE_FONT, grid=True
    ).configure_legend(
        labelFontSize=ALT_LEGEND_LABEL_FONT, titleFontSize=ALT_LEGEND_TITLE_FONT, orient='top'
    )

    save_altair_chart(plot06, "figure06")
    save_altair_chart(plot06, "figure06", out_dir=out_dir)
    return plot06


def plot_phi_by_type(df_final_2, df_final, out_dir=None):
    """Test scores when different types of sequences are considered.

    Shares the probabilities figure's model ordering, which is why it takes that
    figure's frame as well as its own.
    """
    labels_study = list(SEQUENCE_SETS)
    seq_colors = alt.Scale(domain=labels_study, range=['#96ceb4', '#ffcc5c', '#ff6f69', '#da680f'])

    # Get the model order from Figure 06 (sorted by sum of values descending)
    model_order_fig06 = df_final.groupby('c1')['values'].sum().sort_values(ascending=False).index.tolist()

    plot07 = alt.Chart(df_final_2).mark_circle(size=120, opacity=0.85).encode(
        y=alt.Y('c1:N', title=None, sort=model_order_fig06),
        x=alt.X('values:Q', title="𝜑 (Harmonic Mean Ratio)"),
        color=alt.Color('c2:N', title="Sequence Type", scale=seq_colors),
        tooltip=['c1', 'c2', 'values']
    ).properties(
        width=700, height=620
    ).configure_axis(
        labelFontSize=ALT_AXIS_LABEL_FONT, titleFontSize=ALT_AXIS_TITLE_FONT, grid=True
    ).configure_legend(
        labelFontSize=ALT_LEGEND_LABEL_FONT, titleFontSize=ALT_LEGEND_TITLE_FONT, orient='top'
    )

    save_altair_chart(plot07, "figure07")
    save_altair_chart(plot07, "figure07", out_dir=out_dir)
    return plot07


# ---- the bootstrap -----------------------------------------------------------


def inner_loop_function(models, bin_seq_df):
    """One bootstrap resample: every model's score over the drawn rows.

    Top level so joblib can pickle it. The body was four inlined copies of the
    metric in the published script; it is one call now.
    """
    warnings.filterwarnings(
        "ignore",
        message=r"pkg_resources is deprecated as an API\..*",
        category=UserWarning,
    )
    return {mdl: [score(bin_seq_df, mdl)[2]] for mdl in models}


def bootstrap(df, models=None, seed: int = 42, sizes=(25, 50, 75, 100),
              repeats: int = 100) -> pd.DataFrame:
    """Resample the binary sequences and rescore, to show the ranking is stable.

    ``np.random.seed()`` seeds the LEGACY global RNG; ``np.random.default_rng()``
    with no argument builds a fresh Generator from OS entropy and ignores it
    entirely, so the published bootstrap was never reproducible. Seeding the
    Generator itself fixes that. Consequence: o1-Mini's displayed mean becomes
    0.034 rather than the published 0.035. Its exact, non-bootstrap score is
    0.034380, so 0.034 is the correctly rounded value and the published figure
    showed the noisier draw.
    """
    from joblib import Parallel, delayed

    model_names = MODEL_ORDER if models is None else models
    bin_seqs = subset(df, "Binary")
    rng = np.random.default_rng(seed)

    tst_bootstrap = []
    for sz in sizes:
        inpts_bts = []
        for _rep_id in range(repeats):
            idx_v = rng.integers(low=0, high=len(bin_seqs), size=sz, dtype=int)
            inpts_bts.append([model_names, bin_seqs.iloc[idx_v]])
        calcs_lst = Parallel(n_jobs=N_JOBS)(
            delayed(inner_loop_function)(*x) for x in inpts_bts)
        dict_tst = {mdl: [] for mdl in model_names}
        for dict_indiv in calcs_lst:
            for ddd in dict_indiv:
                dict_tst[ddd] = dict_tst[ddd] + dict_indiv[ddd]
        tst_bootstrap.append(dict_tst)

    lst_dfs=[]
    for pp,sz in enumerate(sizes):  # the script hardcoded [25,50,75,100] here
        df_interm = pd.DataFrame(tst_bootstrap[pp]).T.reset_index()
        df_interm = df_interm.melt(id_vars=['index'])
        df_interm.loc[:,'variable']=sz
        lst_dfs.append(df_interm)

    df_bts = pd.concat(lst_dfs)
    df_bts.columns = ["Model","# of Sequences","Test Score"]
    # Map model names to display names
    df_bts['Model'] = df_bts['Model'].apply(get_model_display_name)
    return df_bts


def assign_tiers(df_bts):
    """Split the models into High / Medium / Low by their mean bootstrap score."""
    model_means = df_bts.groupby('Model')['Test Score'].mean().sort_values(ascending=False)

    def assign_broad_tier(model_name):
        mean_score = model_means[model_name]
        if mean_score >= 0.030:
            return 'High'
        elif mean_score >= 0.007:
            return 'Medium'
        else:
            return 'Low'

    df_bts = df_bts.copy()
    df_bts['Broad_Tier'] = df_bts['Model'].apply(assign_broad_tier)
    return df_bts, model_means


def plot_bootstrap(df_bts, out_dir=None):
    """Bootstrap procedure over 25, 50, 75 and 100 sequences, by tier."""
    out = _out(out_dir)
    plt.rcdefaults()
    df_bts, model_means = assign_tiers(df_bts)
    # Create the plot with 3 main tiers, but use offset plotting within each tier
    FIG08_TITLE_FONT = 20
    FIG08_AXIS_LABEL_FONT = 17
    FIG08_TICK_FONT = 15
    FIG08_LEGEND_FONT = 15
    FIG08_ANNOT_FONT = 14

    fig, axes = plt.subplots(3, 1, figsize=(18, 16), sharex=True)

    tiers = ['High', 'Medium', 'Low']
    tier_titles = {
        'High': 'High Performance (≥0.030)',
        'Medium': 'Medium Performance (0.007-0.030)',
        'Low': 'Low Performance (<0.007)'
    }

    # Create color palette
    all_models = sorted(df_bts['Model'].unique())
    colors = sns.color_palette("husl", len(all_models))
    model_color_map = dict(zip(all_models, colors))

    # Different markers
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'X', 'P', 'h', '<', '>', '8']

    for idx, tier in enumerate(tiers):
        tier_data = df_bts[df_bts['Broad_Tier'] == tier]
        tier_models = sorted(tier_data['Model'].unique(), 
                            key=lambda x: model_means[x], 
                            reverse=True)
    
        n_models = len(tier_models)
    
        if n_models == 0:
            axes[idx].text(0.5, 0.5, 'No models in this tier', 
                          ha='center', va='center', transform=axes[idx].transAxes)
            continue
    
        # Calculate vertical offset to separate overlapping lines
        # Models will be slightly offset vertically for visibility
        for model_id, model in enumerate(tier_models):
            model_data = tier_data[tier_data['Model'] == model]
        
            # Calculate statistics
            mean_scores = model_data.groupby('# of Sequences')['Test Score'].mean()
            ci_lower = model_data.groupby('# of Sequences')['Test Score'].quantile(0.25)
            ci_upper = model_data.groupby('# of Sequences')['Test Score'].quantile(0.75)
        
            # Apply small vertical offset based on position in sorted list
            # This spreads overlapping lines slightly
            offset_factor = 0.0001 if tier == 'High' else (0.00005 if tier == 'Medium' else 0.000002)
            vertical_offset = (model_id - n_models/2) * offset_factor
        
            # CHANGE 4: Create label with average in parentheses
            model_avg = model_means[model]
            label_with_avg = f"{model} ({model_avg:.3f})"
        
            # CHANGE 2: Increased markersize from 7 to 9.1 (30% bigger: 7 * 1.3 = 9.1)
            axes[idx].plot(
                mean_scores.index,
                mean_scores.values + vertical_offset,
                marker=markers[model_id % len(markers)],
                markersize=9.1,
                linewidth=2,
                label=label_with_avg,
                color=model_color_map[model],
                markeredgewidth=0.7,
                markeredgecolor='white',
                alpha=0.85
            )
        
            # Add very subtle confidence band (without offset for accuracy)
            axes[idx].fill_between(
                mean_scores.index,
                ci_lower.values,
                ci_upper.values,
                alpha=0.08,
                color=model_color_map[model]
            )
    
        # Formatting
        axes[idx].set_title(tier_titles[tier], fontsize=FIG08_TITLE_FONT, fontweight='bold', pad=12)
        axes[idx].set_ylabel('Test Score', fontsize=FIG08_AXIS_LABEL_FONT)
        axes[idx].tick_params(axis='both', which='major', labelsize=FIG08_TICK_FONT)
        axes[idx].grid(True, alpha=0.25, linestyle='--', linewidth=0.7)
    
        # CHANGE 1: Always place High tier legend outside, adjust other tiers accordingly
        # CHANGE 3: Increased legend fontsize by 25% (9 * 1.25 = 11.25, 8 * 1.25 = 10)
        if tier == 'High':
            # Always place High performance legend outside the plot area
            axes[idx].legend(bbox_to_anchor=(1.02, 1), loc='upper left', 
                            fontsize=FIG08_LEGEND_FONT, framealpha=0.95, ncol=1)
        elif n_models <= 6:
            axes[idx].legend(loc='best', fontsize=FIG08_LEGEND_FONT, framealpha=0.95, ncol=1)
        else:
            axes[idx].legend(bbox_to_anchor=(1.02, 1), loc='upper left', 
                            fontsize=FIG08_LEGEND_FONT, framealpha=0.95, ncol=1)
    
        # Add model count annotation
        axes[idx].text(0.02, 0.02, f'{n_models} model{"s" if n_models != 1 else ""}',
                      transform=axes[idx].transAxes,
                      fontsize=FIG08_ANNOT_FONT, verticalalignment='bottom',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4))

    axes[-1].set_xlabel('# of Sequences', fontsize=FIG08_AXIS_LABEL_FONT)
    axes[-1].set_xticks([25, 50, 75, 100])

    plt.tight_layout(rect=[0, 0, 0.72, 1])
    plt.savefig(out / "figure08.pdf", bbox_inches="tight")
    plt.savefig(out / "figure08.png", bbox_inches="tight", dpi=600)
    plt.close(fig)
    return fig


def print_tier_summary(df_bts):
    """The console report the script printed under the bootstrap figure."""
    df_bts, model_means = assign_tiers(df_bts)
    tiers = ['High', 'Medium', 'Low']
    tier_titles = {
        'High': 'High Performance (>=0.030)',
        'Medium': 'Medium Performance (0.007-0.030)',
        'Low': 'Low Performance (<0.007)',
    }
    print("\nModel Performance Summary by Tier:")
    print("=" * 80)
    for tier in tiers:
        tier_models = df_bts[df_bts['Broad_Tier'] == tier]['Model'].unique()
        if len(tier_models) > 0:
            print(f"\n{tier_titles[tier]} ({len(tier_models)} models):")
            for model in sorted(tier_models, key=lambda x: model_means[x], reverse=True):
                mean = model_means[model]
                std = df_bts[df_bts['Model'] == model]['Test Score'].std()
                print(f"  • {model:30s}  Mean: {mean:.6f}  Std: {std:.6f}")


def main(out_dir=None, models=None, quiet: bool = False) -> None:
    """Reproduce all four figures, in published order."""
    df = load()
    plot_ranking(ranking(df, models), out_dir=out_dir)

    df_final = proportions_by_type(df, models)
    plot_probabilities(df_final, out_dir=out_dir)
    plot_phi_by_type(scores_by_type(df, models), df_final, out_dir=out_dir)

    df_bts = bootstrap(df, models)
    plot_bootstrap(df_bts, out_dir=out_dir)
    if not quiet:
        print_tier_summary(df_bts)


if __name__ == "__main__":
    main()
