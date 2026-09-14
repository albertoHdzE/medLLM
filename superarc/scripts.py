"""Script generation: the two figures built from the Python the models wrote.

    Script Equivalence Analysis + Model Accuracy Performance
    Integrated Script Analysis: Type Distribution and Volume by Model Complexity

The formulae experiment asks a model to *describe* a sequence; this one asks it
to write a program that prints the sequence, and then runs the program. The
measurement is therefore not an opinion about the answer -- the program either
printed the sequence or it did not.

**Equivalence** compares a model's own programs against each other: three
programs that print three different things mean the model did not know the
sequence. **Accuracy** compares each program's output against the target.

Why this module exists
----------------------
The analysis lived in ``31-1_multiScript_experiment.py``, a 1063-line top-level
script whose name cannot be imported -- it begins with a digit and contains a
hyphen -- so nothing could show the numbers behind Figures 5 and 6 without
re-implementing them, and ``31_multiScript_experiment.ipynb`` duly carried its
own diverged copies. The script is now a forwarder onto this module.

Three defects this consolidation had to resolve
-----------------------------------------------
1. The script defined ``get_model_display_name`` **twice**, once before each
   figure, and the two disagreed about which column is Gemini-2.5-Pro. Figure 5
   used ``gemini-2.5-pro``; Figure 6 used ``gemini``. One label, two different
   models, in two figures of the same paper. Identity now comes from
   :mod:`superarc.registry`, which owns it, and the ``gemini`` column is declared
   retired there with the reason.

2. The accuracy panel of Figure 5 selected its columns with
   ``model.lower() in col.lower()`` -- a substring test. Four model names are
   substrings of a longer model's column name in this dataset, so Grok-3's
   accuracy included Grok-4's and Grok-4.1's, Mistral's included
   Mistral-Large-2405's and Mistral-Large-3's, Qwen's included Qwen-3's, and
   DeepSeek's included DeepSeek-R1-0528's. This is the same defect corrected in
   the formulae figures, reached by a different spelling. Resolution is now by
   whole name through :func:`superarc.registry.columns_of`. Note that the
   equivalence panel, the volume counts and the summary emitter all used the
   safer ``f'{model}_script' in col`` form, so Figure 5's two panels disagreed
   with each other about which programs belong to Grok-3.

3. The printed equivalence summary pivoted on one model's value column, so every
   other row came out NaN and "highest average equivalence" always named
   ChatGPT-4o. Console output only; the figure was correct.

``MODEL_ORDER`` stays here rather than in the registry for the reason given in
:mod:`superarc.formulae`: it is this figure's legend and marker order, not model
identity.
"""

from __future__ import annotations

import ast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import REPO_ROOT, plots_dir
from .registry import B_SCRIPT, by_column, columns_of, validate_columns
from .sandbox import Executor

DATASET = REPO_ROOT / "multi-python-script-time-series.csv"

# A model produced no usable answer.
NOT_FOUND = "*not found"

COMPLEXITIES = (1, 2, 3)
COMPLEXITY_LABELS = ("Low", "Medium", "High")

# Four classes here against the formulae figure's three: a program can also be a
# bare `print` of the sequence, which is copying rather than modelling, and
# separating those two is what the benchmark exists to do.
CLASS_LABELS = ("Known sequence", "Not found", "Pure math", "Print")
CLASS_COLOURS = ("#E74C3C", "#3498DB", "#2ECC71", "#F39C12")  # Red, Blue, Green, Orange

# Legend and marker order for the figures -- see the module docstring. Kept in
# the order the published script listed it, which is also the order markers cycle
# in, so reordering it would remark every line of Figure 5.
MODEL_ORDER = [
    "chatgpt-4o",
    "gpt-4o-mini",
    "gemini-2.5-pro",
    "gemini",
    "gemini-thinking",
    "claude-3-5-sonnet",
    "chatgpt-o1",
    "mistral",
    "meta",
    "cursor_small",
    "o1_mini",
    "grok",
    "grok-3",
    "qwen",
    "deepseek",
    "chatgpt-4.5",
    "claude-3.7",
    "deepseek_r1_0525",
    "llama_4_scout",
    "qwen3",
    "chatgpt_5",
    "opus_4",
    "mistral_large2405",
    "claude_sonnet_4",
    "grok_4.1",
    "gpt-5.2",
    "claude-4.5",
    "gemini-3-pro",
    "mistral-large-3",
]

# Markers, in the order MODEL_ORDER consumes them.
MARKERS = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', '8']


def get_model_display_name(model_key: str) -> str:
    """Canonical label for a dataset column.

    One definition, replacing the two the script carried. An unregistered key
    comes back unchanged, which is what :func:`registered` filters on -- but a
    column can now only be unregistered by being declared retired in the
    registry, so that filter no longer hides anything.
    """
    model = by_column(B_SCRIPT, model_key)
    return model.name if model is not None else model_key


def registered(models=None) -> list[str]:
    """The subset of ``MODEL_ORDER`` the registry knows, in figure order."""
    return [m for m in (MODEL_ORDER if models is None else models)
            if get_model_display_name(m) != m]


# ---- loading -----------------------------------------------------------------


def split_model_scripts(df, model_name) -> pd.DataFrame:
    """One model's stringified list of programs, as one column per program.

    Two mechanical changes from the published version, both verified to produce
    identical values on all 2495 populated cells:

    * ``ast.literal_eval`` instead of ``eval``. It parses every cell of this
      dataset to the same object, and this module will be fed new model answers,
      which ``eval`` would *execute* rather than read.
    * returns the new columns instead of inserting them one at a time, so the
      caller can add them in a single concat. Inserting ~300 columns one by one
      is what made pandas warn about a fragmented frame 600 times per run.

    What is *not* changed is the padding behaviour, including its defect: a cell
    holding a quoted string rather than a list is indexed character by character,
    so a ``"*not found*"`` answer becomes eleven one-character programs. See the
    module docstring.
    """
    # Convert string representation of list to actual list and get max length
    def safe_eval(x):
        if pd.isna(x) or x == "" or x == "*not found":
            return ["*not found"]  # Return "*not found" for empty/nan/not found cases
        return ast.literal_eval(x)

    parsed = df[model_name].apply(safe_eval)
    max_scripts = parsed.str.len().max()

    # Create new columns for each script
    return pd.DataFrame({
        f"{model_name}_script_{i+1}":
            parsed.apply(lambda x, i=i: x[i] if i < len(x) else "*not found")
        for i in range(max_scripts)
    }, index=df.index)


def load(path: Path | None = None, models=None) -> pd.DataFrame:
    """The script dataset, with each model's list of programs split into columns."""
    model_names = MODEL_ORDER if models is None else models
    raw = pd.read_csv(path or DATASET)

    # Fail loudly on a model column the registry does not know. Previously an
    # unmapped column was dropped from the figures without a word, which is how
    # a Gemini column silently vanished from each of Figures 5 and 6.
    validate_columns(B_SCRIPT, list(model_names))

    pieces = [raw.drop(columns=list(model_names))]
    pieces += [split_model_scripts(raw, model) for model in model_names]
    return pd.concat(pieces, axis=1)


# ---- per-answer measurement --------------------------------------------------


def clean_result(result):
    # Check for error messages
    if 'Error' in result:
        return result
        
    try:
        # Remove brackets, extra spaces and split on newlines or spaces
        cleaned = result.replace('[','').replace(']','').strip()
        # Split on newlines and/or commas
        numbers = [num.strip() for num in cleaned.replace('\n',',').split(',')]
        # Filter out empty strings and join with commas
        formatted = ', '.join(num for num in numbers if num)
        return formatted
    except:  # noqa: E722 -- verbatim; narrowing it could change results
        return result


def compare_sequences(seq1, seq2):
    # Remove any whitespace and split on commas
    seq1_clean = [x.strip() for x in seq1.split(',')]
    seq2_clean = [x.strip() for x in seq2.split(',')]
    return seq1_clean == seq2_clean


def classify_script(script):
    if pd.isna(script) or not script or 'Error' in str(script) or '*not found' in str(script):
        return 'Not found'
    
    script = str(script).lower()
    
    # Check if it's just a print of sequence
    if script.count('print') == 1 and all(x not in script for x in ['range', '+', '-', '*', '/', 'for', 'while']):
        return 'Print'
    
    # Check for known sequence keywords
    known_sequences = ['fibonacci', 'fib', 'prime', 'lucas', 'arithmetic', 'geometric']
    if any(seq in script for seq in known_sequences):
        return 'Known sequence'
    
    # If contains math operators or programming constructs
    if any(op in script for op in ['+', '-', '*', '/', 'for', 'while', 'range', 'if', '=', '>']):
        return 'Pure math'
        
    return 'Not found'


def _script_cols(df, model, suffix: str = "") -> list[str]:
    """This model's script columns bearing ``suffix``, and no other model's.

    The replacement for ``model in col``. See the module docstring: four names in
    this dataset are substrings of a longer name, so a substring test scored four
    models partly on a later model's programs. The derived-column grammar lives
    in :data:`superarc.registry.COLUMN_SUFFIXES`.
    """
    out = []
    for col in columns_of(B_SCRIPT, model, list(df.columns)):
        rest = col[len(model):]
        if not rest.startswith("_script_"):
            continue
        tail = rest.removeprefix("_script_")
        if suffix:
            if tail.endswith(suffix) and tail[: -len(suffix)].isdigit():
                out.append(col)
        elif tail.isdigit():
            out.append(col)
    return out


def add_measures(df, models=None, executor=None) -> pd.DataFrame:
    """Every derived column the two figures need.

    Returns a new frame -- ``df`` is not modified, so callers must use the return
    value. One concat per stage rather than one insert per column, because ~12,000
    individual inserts is what made pandas warn about a fragmented frame 600
    times per run.

    Order matters twice over. The programs are run first, their output is
    formatted, accuracy compares that output against the target, and equivalence
    compares the outputs against each other -- and the resulting *column* order
    matters too, because :func:`process_model_data` pairs each model's
    classification and accuracy columns by their position in ``df.columns``.
    """
    model_names = MODEL_ORDER if models is None else models
    execute_code_safely = Executor() if executor is None else executor
    frame = df

    def stage(make):
        nonlocal frame
        new = dict(make())
        if new:
            frame = pd.concat([frame, pd.DataFrame(new, index=frame.index)], axis=1)

    stage(lambda: (
        (f"{col}_result", frame[col].apply(execute_code_safely))
        for model in model_names for col in _script_cols(frame, model)))

    stage(lambda: (
        (f"{col}_formatted", frame[col].apply(clean_result))
        for model in model_names for col in _script_cols(frame, model, "_result")))

    stage(lambda: (
        (col.replace("_result_formatted", "_accuracy"),
         frame.apply(lambda row, c=col: compare_sequences(row[c], row["sequence"]), axis=1))
        for model in model_names
        for col in _script_cols(frame, model, "_result_formatted")))

    stage(lambda: (
        (f"{col}_classification", frame[col].apply(classify_script))
        for model in model_names for col in _script_cols(frame, model)))

    stage(lambda: (
        (f"{model}_equivalence_percentage", equivalence_series(frame, model))
        for model in model_names))

    # The published script ran `df.loc[:, ~df.columns.duplicated()]` between its
    # two figures. Resolving by whole name cannot produce a duplicate, so that
    # line now has nothing to remove; asserted rather than silently relied on.
    assert not frame.columns.duplicated().any(), "duplicate derived column"
    return frame


def equivalence_series(df, model):
    """Pairwise agreement among one model's own programs, per row."""
    result_cols = _script_cols(df, model, "_result_formatted")

    def calculate_equivalence(row):
        # Get valid results (not empty, not *not found, no Error)
        valid_results = []
        for col in result_cols:
            result = row[col]
            if pd.notna(result) and '*not found' not in str(result) and 'Error' not in str(result) and result != '':
                valid_results.append(result)
        
        if len(valid_results) < 2:  # Need at least 2 valid results to compare
            return 0
            
        # Compare each pair of valid results
        total_comparisons = 0
        matching_comparisons = 0
        
        for i in range(len(valid_results)):
            for j in range(i+1, len(valid_results)):
                total_comparisons += 1
                if valid_results[i] == valid_results[j]:
                    matching_comparisons += 1
                    
        return (matching_comparisons / total_comparisons * 100) if total_comparisons > 0 else 0

    return df.apply(calculate_equivalence, axis=1)


# ---- aggregation -------------------------------------------------------------


def equivalence_table(df, models=None) -> pd.DataFrame:
    """Mean equivalence per model and complexity -- the top panel of Figure 5.

    Tidy, with a single ``Equivalence %`` column. The original concatenated one
    frame per model, each carrying its own value column, which the figure read
    correctly and the printed summary did not (see the module docstring).
    """
    model_names = MODEL_ORDER if models is None else models
    frames = []
    for model in model_names:
        equiv_col = f'{model}_equivalence_percentage'
        if equiv_col in df.columns:
            grouped = (df[df['Complexity'].isin(COMPLEXITIES)]
                       .groupby('Complexity')[equiv_col].mean())
            frames.append(pd.DataFrame({
                'Complexity': grouped.index,
                'Equivalence %': grouped.values,
                'Model': model,
            }))
    return pd.concat(frames)


def accuracy_table(df, models=None) -> pd.DataFrame:
    """Mean accuracy per model and complexity -- the bottom panel of Figure 5."""
    model_names = MODEL_ORDER if models is None else models
    frames = []
    for model in model_names:
        accuracy_cols = _script_cols(df, model, "_accuracy")
        if accuracy_cols:
            grouped = (df[df['Complexity'].isin(COMPLEXITIES)]
                       .groupby('Complexity')[accuracy_cols].mean()
                       .mean(axis=1).reset_index())
            grouped.columns = ['Complexity', 'Accuracy']
            grouped['Accuracy'] = grouped['Accuracy'] * 100
            grouped['Model'] = model
            frames.append(grouped)
    return pd.concat(frames)


# ============ UNIFIED DATA PROCESSING ============
def process_model_data(df, models=None):
    """Volume, total-type and accurate-type counts, per model and complexity."""
    valid_models = registered(models)
    script_counts = []
    total_classifications = []
    accurate_classifications = []
    
    for complexity in [1, 2, 3]:
        complexity_data = df[df['Complexity'] == complexity]
        
        for model in valid_models:
            # 1. SCRIPT COUNTS
            script_cols = [col for col in df.columns if col.startswith(f'{model}_script_') 
                          and not any(x in col for x in ['_result', '_formatted', '_accuracy', '_classification'])]
            
            valid_script_count = sum(1 for col in script_cols 
                                   if not complexity_data[col].isna().any() and 
                                   not (complexity_data[col] == '*not found').any())
            
            script_counts.append({
                'Complexity': complexity,
                'Model': model,
                'Count': valid_script_count
            })
            
            # 2. TOTAL CLASSIFICATION DISTRIBUTION
            classification_cols = [col for col in df.columns if f'{model}_script' in col and col.endswith('_classification')]
            
            all_total_classifications = []
            for col in classification_cols:
                all_total_classifications.extend(complexity_data[col].dropna())
            
            total_class_counts = pd.Series(all_total_classifications).value_counts()
            
            for category in ['Known sequence', 'Not found', 'Pure math', 'Print']:
                total_classifications.append({
                    'Complexity': complexity,
                    'Model': model,
                    'Classification': category,
                    'Count': total_class_counts.get(category, 0)
                })
            
            # 3. ACCURATE CLASSIFICATION DISTRIBUTION
            accuracy_cols = [col for col in df.columns if f'{model}_script' in col and col.endswith('_accuracy')]
            
            all_accurate_classifications = []
            for class_col, acc_col in zip(classification_cols, accuracy_cols):
                if class_col in complexity_data.columns and acc_col in complexity_data.columns:
                    accurate_data = complexity_data[complexity_data[acc_col]][class_col]
                    all_accurate_classifications.extend(accurate_data.dropna())
            
            accurate_class_counts = pd.Series(all_accurate_classifications).value_counts()
            
            for category in ['Known sequence', 'Not found', 'Pure math', 'Print']:
                accurate_classifications.append({
                    'Complexity': complexity,
                    'Model': model,
                    'Classification': category,
                    'Count': accurate_class_counts.get(category, 0)
                })
    
    script_data = pd.DataFrame(script_counts)

    # Typed over a computed value in the published script, under the comment
    # "Fix for ChatGPT-4o-Mini at complexity 3". Kept because it is what the
    # published figure shows, and because removing it would be editing a
    # published result on my own authority. Flagged here so it is visible:
    # the computed count is reported by
    # tests/test_scripts.py::test_the_hand_set_count_is_the_only_one.
    script_data.loc[(script_data['Model'] == 'gpt-4o-mini') &
                    (script_data['Complexity'] == 3), 'Count'] = 1

    return (script_data,
            pd.DataFrame(total_classifications),
            pd.DataFrame(accurate_classifications))


# ---- figures -----------------------------------------------------------------


def plot_equivalence_and_accuracy(df, out_dir=None, models=None):
    """Script Equivalence Analysis + Model Accuracy Performance."""
    model_names = MODEL_ORDER if models is None else models
    equiv_plot_data = equivalence_table(df, model_names)
    acc_plot_data = accuracy_table(df, model_names)
    PLOTS_DIR = Path(out_dir) if out_dir else plots_dir()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Each figure sets its own style rather than leaving it as global state
    # between the two, which in the original meant the pair came out right only
    # when drawn in that one order.
    plt.rcdefaults()
    # Set scientific plotting style
    plt.style.use('seaborn-v0_8-whitegrid')

    # Create figure with subplots stacked vertically
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(21.6, 16))
    plt.rcParams.update({'font.size': 20, 'font.family': 'serif'})

    markers = MARKERS
    # Plot equivalence lines
    for i, model in enumerate(model_names):
        display_name = get_model_display_name(model)
        if display_name != model:  # Only plot if mapped
            model_data = equiv_plot_data[equiv_plot_data['Model'] == model]
            if not model_data.empty:
                ax1.plot(model_data['Complexity'], 
                        model_data['Equivalence %'],
                        marker=markers[i % len(markers)],
                        markersize=8,
                        linewidth=2,
                        label=display_name,
                        alpha=0.8)

    ax1.set_title('Script Equivalence Analysis', fontsize=35, fontweight='bold', pad=20)
    ax1.set_xlabel('Complexity', fontsize=25)
    ax1.set_ylabel('Equivalence Percentage (%)', fontsize=35)
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.set_xticks([1, 2, 3])
    ax1.set_xticklabels(['Low', 'Medium', 'High'])
    ax1.set_ylim(0, 100)
    ax1.xaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax1.yaxis.set_major_locator(plt.MultipleLocator(10))
    ax1.tick_params(axis='both', which='major', labelsize=26)
    # ============ PLOT 2: ACCURACY ============
    # Plot accuracy lines
    for i, model in enumerate(model_names):
        display_name = get_model_display_name(model)
        if display_name != model:  # Only plot if mapped
            model_data = acc_plot_data[acc_plot_data['Model'] == model]
            if not model_data.empty:
                ax2.plot(model_data['Complexity'],
                        model_data['Accuracy'],
                        marker=markers[i % len(markers)],
                        markersize=8,
                        linewidth=2,
                        label=display_name,
                        alpha=0.8)
    ax2.set_title('Model Accuracy Performance', fontsize=35, fontweight='bold', pad=20)
    ax2.set_xlabel('Complexity', fontsize=25)
    ax2.set_ylabel('Accuracy (%)', fontsize=35)
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.set_xticks([1, 2, 3])
    ax2.set_xticklabels(['Low', 'Medium', 'High'])
    ax2.set_ylim(0, 100)
    ax2.xaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax2.yaxis.set_major_locator(plt.MultipleLocator(10))
    ax2.tick_params(axis='both', which='major', labelsize=26)

    # Add shared legend to the right-hand side of the general plot
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels,
              loc='center right',
              bbox_to_anchor=(1.0, 0.5),
              fontsize=30,
              frameon=True,
              fancybox=True,
              shadow=True)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(right=0.70, hspace=0.3)

    fig.savefig(PLOTS_DIR / "figure04-up.pdf", bbox_inches="tight", dpi=600)
    fig.savefig(PLOTS_DIR / "figure04-up.png", bbox_inches="tight")
    # Close it: the inline backend auto-displays every open figure at the end of
    # a notebook cell, embedding a second full-resolution copy.
    plt.close(fig)
    return fig


def print_equivalence_and_accuracy_summary(df, models=None):
    """The console report the script printed under Figure 5."""
    model_names = MODEL_ORDER if models is None else models
    equiv_plot_data = equivalence_table(df, model_names)
    acc_plot_data = accuracy_table(df, model_names)
    accuracy_data = [acc_plot_data]
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY STATISTICS")
    print("="*80)

    # Equivalence summary
    print("\n SCRIPT EQUIVALENCE ANALYSIS (%)")
    print("-" * 60)
    equiv_summary = equiv_plot_data.pivot(index='Model', columns='Complexity',
                                          values='Equivalence %')
    equiv_summary.columns = [f'Complexity {i}' for i in equiv_summary.columns]
    equiv_summary = equiv_summary.round(2)

    # Add display names to index
    equiv_summary.index = [get_model_display_name(model) if get_model_display_name(model) != model else model 
                          for model in equiv_summary.index]

    print(equiv_summary.to_string())

    # Accuracy summary
    if accuracy_data:
        print("\n🎯 MODEL ACCURACY ANALYSIS (%)")
        print("-" * 60)
        acc_summary = acc_plot_data.pivot(index='Model', columns='Complexity', values='Accuracy').round(2)
        acc_summary.columns = [f'Complexity {i}' for i in acc_summary.columns]
    
        # Add display names to index
        acc_summary.index = [get_model_display_name(model) if get_model_display_name(model) != model else model 
                            for model in acc_summary.index]
    
        print(acc_summary.to_string())

    # Statistical insights
    print("\n KEY INSIGHTS")
    print("-" * 40)
    if accuracy_data:
        best_overall_acc = acc_summary.mean(axis=1).idxmax()
        worst_complexity_drop_acc = (acc_summary.iloc[:, 0] - acc_summary.iloc[:, -1]).idxmax()
    
    print(f"• Highest average accuracy: {best_overall_acc}")
    print(f"• Largest complexity-related accuracy drop: {worst_complexity_drop_acc}")

    best_overall_equiv = equiv_summary.mean(axis=1).idxmax()
    most_consistent_equiv = equiv_summary.std(axis=1).idxmin()

    print(f"• Highest average equivalence: {best_overall_equiv}")
    print(f"• Most consistent across complexities: {most_consistent_equiv}")

    print("\n" + "="*80)


def plot_integrated(df, out_dir=None, models=None, counts=None):
    """Integrated Script Analysis: type distribution and volume.

    ``counts`` is the triple :func:`process_model_data` returns; passing it in
    avoids recomputing it when the caller already has it to print.
    """
    valid_models = registered(models)
    script_data, total_class_data, accurate_class_data = (
        process_model_data(df, models) if counts is None else counts)
    PLOTS_DIR = Path(out_dir) if out_dir else plots_dir()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcdefaults()
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'font.family': 'serif', 'font.size': 10})
    # ============ CREATE UNIFIED SCIENTIFIC VISUALIZATION ============
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 20,
        'axes.titlesize': 30,
        'axes.labelsize': 26,
        'xtick.labelsize': 22,
        'ytick.labelsize': 22,
        'legend.fontsize': 18,
    })

    fig, axes = plt.subplots(3, 1, figsize=(24, 22), sharex=True)
    plt.subplots_adjust(top=0.84, bottom=0.12, left=0.13, right=0.83, hspace=0.25)  # Moved 1cm leftwards

    # Define scientific color schemes
    class_colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']  # Red, Blue, Green, Orange
    class_labels = ['Known sequence', 'Not found', 'Pure math', 'Print']

    # ============ INTEGRATED VISUALIZATION: STACKED BARS WITH VOLUME OVERLAY ============
    display_models = [get_model_display_name(model) for model in valid_models]
    x_positions = np.arange(len(valid_models))
    bar_width = 0.4
    volume_axes = []

    for idx, complexity in enumerate([1, 2, 3]):
        ax = axes[idx]
    
        # Get data for this complexity
        total_data = total_class_data[total_class_data['Complexity'] == complexity]
        accurate_data = accurate_class_data[accurate_class_data['Complexity'] == complexity]
        volume_data = script_data[script_data['Complexity'] == complexity]
    
        # Plot total scripts (left bars) - with lighter colors
        bottom_total = np.zeros(len(valid_models))
        for i, classification in enumerate(class_labels):
            class_counts = []
            for model in valid_models:
                count = total_data[(total_data['Model'] == model) & 
                                  (total_data['Classification'] == classification)]['Count']
                class_counts.append(count.iloc[0] if len(count) > 0 else 0)
        
            ax.bar(x_positions - bar_width/2, class_counts, bar_width, bottom=bottom_total,
                   label=f'{classification} (Total)' if idx == 0 else "",
                   color=class_colors[i], alpha=0.4, edgecolor='white', linewidth=0.5)
            bottom_total += np.array(class_counts)
    
        # Plot accurate scripts (right bars) - with full colors
        bottom_accurate = np.zeros(len(valid_models))
        for i, classification in enumerate(class_labels):
            class_counts = []
            for model in valid_models:
                count = accurate_data[(accurate_data['Model'] == model) & 
                                     (accurate_data['Classification'] == classification)]['Count']
                class_counts.append(count.iloc[0] if len(count) > 0 else 0)
        
            ax.bar(x_positions + bar_width/2, class_counts, bar_width, bottom=bottom_accurate,
                   label=f'{classification} (Accurate)' if idx == 0 else "",
                   color=class_colors[i], alpha=0.9, edgecolor='white', linewidth=0.5)
            bottom_accurate += np.array(class_counts)
    
        # ============ OVERLAY: LINE PLOT FOR VALID SCRIPT VOLUME ============
        ax2 = ax.twinx()
        volume_axes.append(ax2)
    
        volume_counts = []
        for model in valid_models:
            count = volume_data[volume_data['Model'] == model]['Count']
            volume_counts.append(count.iloc[0] if len(count) > 0 else 0)
    
        # Plot volume as a line with markers on secondary axis
        ax2.plot(x_positions, volume_counts, 
                        color='black', linewidth=1.2, marker='D', 
                        markersize=4, markerfacecolor='gold', 
                        markeredgecolor='black', markeredgewidth=0.8,
                        label='Valid Scripts Volume' if idx == 0 else "",
                        zorder=10, alpha=0.7)
    
        # Styling for primary axis (Script Types)
        ax.set_title(f'Complexity {complexity}', fontweight='bold', pad=16)
        ax.set_xticks(x_positions)
        ax.set_axisbelow(True)
        ax.grid(True, alpha=1, axis='y', zorder=0)
        ax.grid(True, alpha=0.15, axis='x', zorder=0, linewidth=0.8)
        for x in x_positions:
            ax.axvline(x=x, color='black', alpha=0.10, linewidth=0.8, zorder=0)
        ax.set_ylabel('Script Type Distribution Count', labelpad=12, fontsize=22)
    
        # Styling for secondary axis (Volume)
        ax2.set_ylabel('Valid Scripts Volume', color='darkgoldenrod', fontweight='bold', labelpad=14)
        ax2.tick_params(axis='y', labelcolor='darkgoldenrod')
        ax2.spines['right'].set_color('darkgoldenrod')
        ax2.spines['right'].set_linewidth(2)
        ax2.yaxis.set_label_position("right")
        ax2.yaxis.tick_right()
    
        # Add subtle background shading to distinguish volume line
        ax.axhspan(ax.get_ylim()[0], ax.get_ylim()[1], alpha=0.02, color='yellow', zorder=0)

        # Modify only the left-most subplot (Complexity 1)
        if idx == 0:
            ax2.set_yticks([0, 1, 2, 3])
            ax2.set_ylim(0, 3)

        if idx != len(axes) - 1:
            ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        else:
            ax.set_xticklabels(display_models, rotation=90, ha='center')
            ax.set_xlabel('Model', labelpad=18)

    # ============ ADD UNIFIED LEGEND ============
    # Collect handles and labels from all axes
    handles1, labels1 = axes[0].get_legend_handles_labels()
    handles2, labels2 = volume_axes[0].get_legend_handles_labels()

    # Combine all handles
    all_handles = handles1 + handles2
    all_labels = labels1 + labels2

    # Sort legend to group by classification type
    sorted_indices = sorted(range(len(all_labels)),
                           key=lambda i: (all_labels[i].split(' (')[0] if '(' in all_labels[i] else 'ZZZ',
                                        all_labels[i]))
    sorted_handles = [all_handles[i] for i in sorted_indices]
    sorted_labels = [all_labels[i] for i in sorted_indices]

    # ---- Legend placement explained ----
    # The call below places the legend at a precise, user-defined spot on the figure.
    # How it works:
    #   - fig.legend(...)  ->  legend belongs to the whole figure (not an individual Axes)
    #   - loc='upper center'  ->  anchor point is the *center* of the legend's top edge
    #   - bbox_to_anchor=(0.5, 0.88)  ->  put that anchor at figure-relative coords (x, y)
    #        (0, 0) = bottom-left corner of the figure
    #        (1, 1) = top-right corner of the figure
    #   - ncol=2  ->  two columns of legend entries
    #   - frameon=True, fancybox=True, shadow=True  ->  cosmetic styling
    #
    # Quick reference for other common spots (keep ncol & styling as desired):
    #   Top-right inside:   loc='upper right',  bbox_to_anchor=(0.98, 0.98)
    #   Top-left inside:    loc='upper left',   bbox_to_anchor=(0.02, 0.98)
    #   Bottom-center:      loc='lower center', bbox_to_anchor=(0.5, 0.02)
    #   Right-outside:      loc='center left',  bbox_to_anchor=(1.02, 0.5)  # useful when tight_layout / constrained_layout pad extra space
    #   Absolute center:    loc='center',       bbox_to_anchor=(0.5, 0.5)
    #
    # Fine-tune: change the two numbers in bbox_to_anchor by ±0.01 until the legend
    # sits exactly where you want.  If you use constrained_layout=True or
    # plt.tight_layout(), call that *after* fig.legend(...) so the layout manager
    # reserves space for the legend.

    fig.legend(sorted_handles, sorted_labels,
               loc='upper center', bbox_to_anchor=(0.5, 0.95),
               ncol=4, frameon=True, fancybox=True, shadow=True)

    # Add overall title
    fig.suptitle('Integrated Script Analysis: Type Distribution and Volume by Model Complexity', 
                fontweight='bold', y=0.98, fontsize=35)

    fig.savefig(PLOTS_DIR / "figure04-bottom.pdf", bbox_inches="tight", dpi=600)
    fig.savefig(PLOTS_DIR / "figure04-bottom.png", bbox_inches="tight")
    plt.close(fig)
    return fig


def print_integrated_summary(script_data, total_class_data, accurate_class_data):
    """The console report the script printed under Figure 6."""
    class_labels = list(CLASS_LABELS)
    print("\n" + "="*90)
    print("UNIFIED SCRIPT ANALYSIS RESULTS")
    print("="*90)

    print("\n📊 SCRIPT GENERATION VOLUME")
    print("-" * 50)
    volume_summary = script_data.pivot(index='Model', columns='Complexity', values='Count')
    volume_summary.columns = [f'Complexity {int(i)}' for i in volume_summary.columns]
    volume_summary.index = [get_model_display_name(model) for model in volume_summary.index]
    print(volume_summary.to_string())

    print("\n📝 TOTAL SCRIPT TYPE DISTRIBUTION")
    print("-" * 50)
    for classification in class_labels:
        print(f"\n{classification}:")
        class_summary = total_class_data[total_class_data['Classification'] == classification].pivot(
            index='Model', columns='Complexity', values='Count').fillna(0)
        class_summary.columns = [f'Complexity {int(i)}' for i in class_summary.columns]
        class_summary.index = [get_model_display_name(model) for model in class_summary.index]
        print(class_summary.head(8).to_string())

    print("\n✅ ACCURATE SCRIPT TYPE DISTRIBUTION")
    print("-" * 50)
    for classification in class_labels:
        print(f"\n{classification} (Accurate Only):")
        accurate_summary = accurate_class_data[accurate_class_data['Classification'] == classification].pivot(
            index='Model', columns='Complexity', values='Count').fillna(0)
        accurate_summary.columns = [f'Complexity {int(i)}' for i in accurate_summary.columns]
        accurate_summary.index = [get_model_display_name(model) for model in accurate_summary.index]
        print(accurate_summary.head(8).to_string())

    # ============ KEY INSIGHTS ============
    print("\n🔍 KEY INSIGHTS")
    print("-" * 30)

    # Best performing models by volume
    total_volume = script_data.groupby('Model')['Count'].sum().sort_values(ascending=False)
    total_volume.index = [get_model_display_name(model) for model in total_volume.index]
    print(f"• Highest script generation volume: {total_volume.index[0]} ({total_volume.iloc[0]} scripts)")

    # Most consistent across complexities
    volume_consistency = script_data.pivot(index='Model', columns='Complexity', values='Count').std(axis=1).sort_values()
    volume_consistency.index = [get_model_display_name(model) for model in volume_consistency.index]
    print(f"• Most consistent generation across complexities: {volume_consistency.index[0]}")

    # Accuracy insights
    total_accurate = accurate_class_data.groupby('Model')['Count'].sum().sort_values(ascending=False)
    total_accurate.index = [get_model_display_name(model) for model in total_accurate.index]
    print(f"• Highest accurate script count: {total_accurate.index[0]} ({total_accurate.iloc[0]} accurate scripts)")

    print("\n" + "="*90)


# ---- the measures behind Supplementary Figures 5 and 6 -----------------------


def emit_summary_measures(df, script_data, total_class_data, models=None):
    """Write the measures the family-evolution figures consume.

    See the matching block in :mod:`superarc.formulae`. Accuracy and equivalence
    are recomputed here from the columns the figures plot rather than reused from
    the plotting frames, because those frames carry a per-model value column name
    and are awkward to read back.
    """
    from .model_summary import CASE_SCRIPT as _CASE, write_long as _write_long

    model_names = MODEL_ORDER if models is None else models

    def _canonical(key):
        model = by_column(B_SCRIPT, key)
        return model.name if model is not None else None

    _records = []
    _filtered = df[df['Complexity'].isin([1, 2, 3])]

    for _model in model_names:
        _name = _canonical(_model)
        if _name is None:
            continue  # declared retired in the registry

        # The original selected these with `f'{_model}_script' in c`, which is
        # safe for every name in this dataset but is a second spelling of the
        # same question. One owner: _script_cols. Verified to select exactly the
        # same columns for all 29 models before the switch.
        _acc_cols = _script_cols(df, _model, '_accuracy')
        if _acc_cols:
            for _cx, _val in _filtered.groupby('Complexity')[_acc_cols].mean().mean(axis=1).items():
                _records.append({'model': _name, 'complexity': int(_cx),
                                 'measure': 'Accuracy', 'value': float(_val) * 100})

        _eq = f'{_model}_equivalence_percentage'
        if _eq in df.columns:
            for _cx, _val in _filtered.groupby('Complexity')[_eq].mean().items():
                _records.append({'model': _name, 'complexity': int(_cx),
                                 'measure': 'Equivalence', 'value': float(_val)})

    for _, _r in script_data.iterrows():
        _name = _canonical(_r['Model'])
        if _name is not None:
            _records.append({'model': _name, 'complexity': int(_r['Complexity']),
                             'measure': 'Valid Instances', 'value': float(_r['Count'])})

    for _, _r in total_class_data.iterrows():
        _name = _canonical(_r['Model'])
        if _name is not None:
            _records.append({'model': _name, 'complexity': int(_r['Complexity']),
                             'measure': _r['Classification'], 'value': float(_r['Count'])})
    path = _write_long(_CASE, _records)
    print(f"\nemitted {len(_records)} summary measures to {path}")
    return path


def main(out_dir=None, models=None, quiet: bool = False) -> None:
    """Reproduce both figures and both console reports, in published order."""
    df = add_measures(load(models=models), models=models)
    plot_equivalence_and_accuracy(df, out_dir=out_dir, models=models)
    counts = process_model_data(df, models)
    script_data, total_class_data, _accurate = counts
    plot_integrated(df, out_dir=out_dir, models=models, counts=counts)
    if not quiet:
        print_equivalence_and_accuracy_summary(df, models=models)
        print_integrated_summary(*counts)
    emit_summary_measures(df, script_data, total_class_data, models=models)


if __name__ == "__main__":
    main()
