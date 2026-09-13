"""Formulae generation: the two figures built from what the models wrote.

    Percentage of Equivalence between Formulae + Accuracy by Complexity
    Integrated Formulae Analysis: Type Distribution and Volume by Model Complexity

Models were asked for *at least three different* formulae reproducing a numeric
sequence. Two things are then measured. **Equivalence** asks whether a model's
own answers agree with each other once evaluated -- a model that gives three
formulae which produce three different sequences has not understood the
sequence, whatever else it has done. **Accuracy** asks whether the evaluated
formula reproduces the target at all.

Why this module exists
----------------------
This code was in ``30-1_multiFormula_experiment.py``, a 806-line top-level
script that cannot be imported -- its name begins with a digit and contains a
hyphen -- so a notebook wanting to show the numbers behind the figures had no
way to reach it except by re-implementing them. That is how the repository came
to hold two versions of this analysis: ``30_multiFormula_experiment.ipynb``
carries copies of five of these functions under the same names, and is missing
every correction the script has received.

The script is now a forwarder onto this module. The notebook copy is deleted,
not adapted -- it was the deficient side of the pair.

Model identity comes from :mod:`superarc.registry`, which owns it. The script
carried its own 28-entry label map; it was checked against the registry
elementwise before being removed -- 27 of 28 agreed, and the one that did not is
the known ``DeepSeek-R1-0525``/``-0528`` case, which ``display_name(...,
published=True)`` reproduces, giving 28/28.

``MODEL_ORDER`` stays here rather than in the registry, because it is not model
identity: it is this figure's legend order, and it also fixes each model's colour
through ``tab20``. Reordering it would repaint every line in the figure. The
registry's own ordering is by family and release, which is what the evolution
figures need and what this figure must not use. ``tests/test_formulae.py`` holds
the two to the same *set*, so a model can never be added to one and missed by the
other.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import REPO_ROOT, plots_dir
from .registry import A_FORMULA, columns_of, display_name

DATASET = REPO_ROOT / "multi-formula-time-series.csv"

# A model produced no usable answer. Both spellings occur in the data.
NOT_FOUND = "*not found"

COMPLEXITIES = (1, 2, 3)
COMPLEXITY_LABELS = ("Low", "Medium", "High")

CLASS_LABELS = ("Known sequence", "Pure math", "Not found")
CLASS_COLOURS = ("#3498DB", "#2ECC71", "#E74C3C")

# Legend and colour order for the figures -- see the module docstring.
MODEL_ORDER = [

    "gpt_4o",
    "gpt_4o_mini",
    "gemini_2.5_pro",
    "gemini",
    "claude_3.5",
    "o1_preview",
    "mistral",
    "meta",
    "cursor_small",
    "o1_mini",
    "grok_3",
    "grok4",
    "qwen",
    "deepseek",
    "chatgpt_4.5",
    "claude_3.7",
    "deepseek_r1_0528",
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


def get_model_display_name(model_key: str) -> str:
    """Label as printed in the paper.

    Delegates to the registry, asking for the *published* label so that
    DeepSeek-R1-0528 keeps the name the paper gave it.
    """
    return display_name(A_FORMULA, model_key, published=True)


def load(path: Path | None = None) -> pd.DataFrame:
    """The formulae dataset, with the whitespace the analysis assumes stripped."""
    df = pd.read_csv(path or DATASET, encoding="latin1")
    df = df.fillna(NOT_FOUND)
    for col in [c for c in df.columns if c.endswith("_eval")]:
        df[col] = df[col].astype(str).str.strip()
    return df


# ---- column resolution -------------------------------------------------------
# Columns are `<model-name><suffix>`, and the model name is an atomic keyword:
# it must be matched WHOLE. Matching by prefix (the original behaviour) also
# caught longer model names, so ChatGPT-4o absorbed ChatGPT-4o-Mini's answers,
# Gemini absorbed Gemini-2.5-Pro, Mistral absorbed Mistral-Large-2405, DeepSeek
# absorbed DeepSeek-R1-0528 and Qwen absorbed Qwen-3.
def _eval_cols(df, model):
    """The `_eval` columns belonging to this model and no other."""
    return [c for c in columns_of(A_FORMULA, model, list(df.columns)) if c.endswith("_eval")]


# Lifted out of test_model_accuracy, where it was a closure and therefore
# untestable. Pure function; the figures are unchanged by the move.
def compare_sequences(seq1, seq2):
    try:
        # Remove any surrounding quotes and split to lists of numbers
        seq1_clean = seq1.strip().strip('"\'')
        seq2_clean = seq2.strip().strip('"\'')
        seq1_nums = [int(x.strip()) for x in seq1_clean.split(',')]
        seq2_nums = [int(x.strip()) for x in seq2_clean.split(',')]
        # Compare only up to length of first sequence
        match = seq1_nums == seq2_nums[:len(seq1_nums)]
        #print(f"Comparing {seq1_nums} vs {seq2_nums[:len(seq1_nums)]} -> {match}")
        return match
    except (ValueError, AttributeError):
        # Return False if conversion fails (e.g. '*not found' case)
        #print(f"Comparison failed for seq1='{seq1}' vs seq2='{seq2}' -> False")
        return False


def test_model_equivalence(df, model_names, verbose=False):
    for model in model_names:
        # Get all eval columns for this model that start with the model name and end with '_eval'
        eval_cols = _eval_cols(df, model)
        if verbose:
            print("Model: " + model + " eval_cols: " + str(eval_cols))
        
        # For each row, calculate equivalence percentage
        def calc_row_equivalence(row):
            valid_values = []
            for col in eval_cols:
                val = row[col]
                if pd.notna(val) and val != '*not found':
                    valid_values.append(val)
            
            if len(valid_values) < 2:  # Need at least 2 valid values to compare
                return 0.0
                
            # Compare each pair of valid values
            total_pairs = 0
            matching_pairs = 0
            for i in range(len(valid_values)):
                for j in range(i + 1, len(valid_values)):
                    total_pairs += 1
                    if valid_values[i] == valid_values[j]:
                        matching_pairs += 1
                        
            return matching_pairs / total_pairs if total_pairs > 0 else 0.0
            
        # Calculate equivalence for each row and store in new column
        df[f'equivalence_{model}'] = df.apply(calc_row_equivalence, axis=1)


def test_model_accuracy(df, model_names, verbose=False):
    for model in model_names:
        eval_cols = _eval_cols(df, model)
        if verbose:
            print(f"Eval columns for {model}: {eval_cols}")
        
        if eval_cols:  # If there are eval columns for this model
            # Calculate accuracy for each eval column
            for eval_col in eval_cols:
                # Construct the accuracy column name by:
                # 1. Starting with 'accuracy-test-' prefix
                # 2. Appending the model name
                # 3. Appending the part of eval_col after the model name (skipping the underscore)
                # Example: if eval_col is 'model_1_1_eval' and model is 'model_1', this becomes 'accuracy-test-model_1_1'
                accuracy_col = f'accuracy-test-{model}{eval_col[len(model)+0:]}'
                
                
                
                # Apply comparison
                df[accuracy_col] = df.apply(
                    lambda row: int(compare_sequences(row['sequence'], row[eval_col])) 
                    if pd.notna(row[eval_col])
                    else 0, axis=1)
            
            # Calculate percentage accuracy across all eval columns for this model
            accuracy_cols = [f'accuracy-test-{c}' for c in _eval_cols(df, model)]
            df[f'accuracy-percentage-{model}'] = df[accuracy_cols].mean(axis=1) * 100


# Function to classify content
def classify_content(text):
    if pd.isna(text):
        return "Not found"
    
    text = str(text).lower()
    
    if "*not found" in text:
        return "Not found"
        
    # Known sequence keywords
    known_sequences = ['fibonacci', 'prime', 'binary', 'mersenne', 'triangular', 
                      'square', 'cube', 'powers', 'factorial', 'arithmetic', 'geometric', 'binomial'
                      'superfactorial', 'exponential', 'logarithmic', 'permutation', 'combination',
                     'permutate', 'combinatorial', 'sequence', 'number', 'series', 'pattern', 'formula',
                     'congruence', 'congruent', 'congruence', 'congruent', 'congruence', 'congruent', 'Catalan'
                     'Mersenne', 'Lucas', 'Pell', 'Lucas-Lehmer', 'Pascal', 'Fibonacci', 'Lucas', 'Pell', 'Lucas-Lehmer', 'Pascal']
    
    # Check if any known sequence keyword is in the text
    if any(seq in text for seq in known_sequences) or text.isalpha():
        return "Known sequence"
        
    # If it contains mathematical operations and isn't a known sequence
    return "Pure math"


# ============ UNIFIED DATA PROCESSING ============
def process_formula_data(df, model_names=None):
    """Volume, total-type and accurate-type counts, per model and complexity."""
    model_names = MODEL_ORDER if model_names is None else model_names
    
    # 1. TOTAL FORMULA COUNTS
    filtered_df = df[df['Complexity'].isin([1, 2, 3])]
    formula_count_cols = [col for col in df.columns if col.endswith('_number_formula')]
    total_formulas = filtered_df.groupby('Complexity')[formula_count_cols].sum()
    
    total_counts = []
    for complexity in [1, 2, 3]:
        for col in formula_count_cols:
            model_name = col.replace('_number_formula', '')
            total_counts.append({
                'Complexity': complexity,
                'Model': model_name,
                'Count': total_formulas.loc[complexity, col]
            })
    
    # 2. TOTAL CLASSIFICATION DISTRIBUTION
    classification_cols = [col for col in df.columns if col.endswith('_classification')]
    formula_columns = [col for col in df.columns if col.endswith('_number_formula')]
    
    total_classifications = []
    for complexity in [1, 2, 3]:
        mask = df['Complexity'] == complexity
        for formula_col, class_col in zip(formula_columns, classification_cols):
            model_name = formula_col.replace('_number_formula', '')
            formulas = df[mask][formula_col]
            classifications = df[mask][class_col]
            
            for classification in ['Known sequence', 'Pure math', 'Not found']:
                class_mask = classifications == classification
                if complexity == 3 and classification == 'Not found':
                    count = len(classifications[class_mask])
                else:
                    count = formulas[class_mask].sum()
                
                total_classifications.append({
                    'Complexity': complexity,
                    'Model': model_name,
                    'Classification': classification,
                    'Count': count
                })
    
    # 3. ACCURATE CLASSIFICATION DISTRIBUTION
    accurate_classifications = []
    for complexity in [1, 2, 3]:
        complexity_mask = df['Complexity'] == complexity
        
        for model in model_names:
            accuracy_col = f'accuracy-test-{model}_1_eval'
            classification_col = f'{model}_classification'
            
            if accuracy_col in df.columns and classification_col in df.columns:
                # noqa: E712 -- elementwise comparison building a boolean
                # mask over a Series. `if col:` would raise; this is not a
                # truth test. Verbatim from the published script.
                correct_mask = df[complexity_mask][accuracy_col] == True  # noqa: E712
                correct_responses = df[complexity_mask][correct_mask]
                
                if len(correct_responses) > 0:
                    type_counts = correct_responses[classification_col].value_counts()
                    for classification in ['Known sequence', 'Pure math', 'Not found']:
                        accurate_classifications.append({
                            'Complexity': complexity,
                            'Model': model,
                            'Classification': classification,
                            'Count': type_counts.get(classification, 0)
                        })
                else:
                    for classification in ['Known sequence', 'Pure math', 'Not found']:
                        accurate_classifications.append({
                            'Complexity': complexity,
                            'Model': model,
                            'Classification': classification,
                            'Count': 0
                        })
    
    return (pd.DataFrame(total_counts), 
            pd.DataFrame(total_classifications), 
            pd.DataFrame(accurate_classifications))


def add_measures(df, models=None, verbose=False):
    """Every derived column the two figures need, added in place.

    Order matters: accuracy reads the columns equivalence does not, but the
    classification counts read the accuracy columns, and ``process_formula_data``
    reads all of them.
    """
    models = MODEL_ORDER if models is None else models
    test_model_equivalence(df, models, verbose=verbose)
    test_model_accuracy(df, models, verbose=verbose)
    _add_formula_counts(df, models, verbose=verbose)
    _add_classifications(df, models)
    return df


def _add_formula_counts(df, models, verbose=False):
    """Count valid formulae per model per row."""
    model_names = models
    # Count number of valid formulas per model
    for model in model_names:
        # Get evaluation columns for this model
        eval_cols = _eval_cols(df, model)
        if verbose:
            print(f'evaluation columns for {model}: {eval_cols}')
    
        # Count valid formulas (not empty, null, inf, *not found)
        df[f'{model}_number_formula'] = df[eval_cols].apply(
            lambda row: sum(
                pd.notna(x) and x != float('inf') and x != '*not found  ' and x != '*not found'
                for x in row
            ), 
            axis=1
        )

    return df


def _add_classifications(df, models):
    """Label each model's raw answer as known sequence / pure math / not found."""
    model_names = models
    # Add classification columns for each model
    for model in model_names:
        # Get the model's main output column (without eval- prefix)
        model_col = model
    
        # Create new classification column name
        classification_col = f"{model}_classification"
    
        # Apply classification function and create new column
        df[classification_col] = df[model_col].apply(classify_content)

    return df


def equivalence_table(df, models=None):
    """Mean equivalence percentage per model and complexity -- the top panel."""
    models = MODEL_ORDER if models is None else models
    model_names = models
    equiv_results = []
    for model in model_names:
        equiv_col = f'equivalence_{model}'  # Changed from equivalence-{model}
        if equiv_col in df.columns:  # Check if equivalence column exists for model
            # Calculate mean equivalence percentage per complexity level
            grouped = df.groupby('Complexity')[equiv_col].mean() * 100
            equiv_results.append(pd.DataFrame({
                'Model': model,
                'Complexity': grouped.index,
                'Equivalence %': grouped.values
            }))

    # Combine results
    equiv_df = pd.concat(equiv_results)

    # Filter for complexity values 1-3
    equiv_df = equiv_df[equiv_df['Complexity'].isin([1, 2, 3])]

    return equiv_df


def accuracy_table(df):
    """Mean accuracy percentage per complexity -- the bottom panel."""
    accuracy_percentage_cols = [c for c in df.columns if "accuracy-percentage-" in c]
    accuracy_percentage_cols = [col for col in df.columns if 'accuracy-percentage-' in col]

    # Filter for complexities 1, 2, 3 and group by complexity
    filtered_df = df[df['Complexity'].isin([1, 2, 3])]
    accuracy_by_complexity = filtered_df.groupby('Complexity')[accuracy_percentage_cols].mean()

    return accuracy_by_complexity


def plot_equivalence_and_accuracy(df, equiv_df=None, out_dir=None, models=None):
    """Percentage of Equivalence between Formulae + Accuracy by Complexity."""
    model_names = MODEL_ORDER if models is None else models
    equiv_df = equivalence_table(df, model_names) if equiv_df is None else equiv_df
    accuracy_by_complexity = accuracy_table(df)
    accuracy_percentage_cols = [c for c in df.columns if "accuracy-percentage-" in c]
    PLOTS_DIR = Path(out_dir) if out_dir else plots_dir()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Each figure sets its own style. In the original script these were global
    # statements between the two plotting blocks, so the figures came out right
    # only when run in that one order -- calling them the other way round
    # silently restyled this one.
    plt.rcdefaults()
    # Create a single figure with two subplots stacked vertically
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(21.6, 16))  # Reduced width by 10% (24 -> 21.6)
    plt.rcParams.update({'font.size': 20})  # Double the font size



    # Generate distinct colors using a colormap
    colors = plt.cm.tab20(np.linspace(0, 1, len(model_names)))

    # Plot equivalence subplot
    for i, model in enumerate(model_names):
        model_data = equiv_df[equiv_df['Model'] == model]
        ax1.plot(model_data['Complexity'],
                 model_data['Equivalence %'],
                 marker='o',  # Add circular markers
                 markersize=10,  # Make markers larger
                 linewidth=2,  # Make lines thicker
                 label=get_model_display_name(model),
                 color=colors[i])

    ax1.set_xlabel('Complexity', fontsize=25)
    ax1.set_ylabel('Equivalence Percentage', fontsize=30)
    ax1.set_title('Percentage of Equivalence between Formulae', fontsize=35)
    ax1.grid(True)
    ax1.set_xticks([1, 2, 3])
    ax1.set_xticklabels(['Low', 'Medium', 'High'])
    ax1.tick_params(axis='both', which='major', labelsize=26)  # Increased by 30%

    # Plot accuracy subplot
    for model, color in zip(model_names, colors):
        col = f'accuracy-percentage-{model}'
        if col in accuracy_percentage_cols:  # Only plot if model exists in data
            ax2.plot(accuracy_by_complexity.index, accuracy_by_complexity[col],
                    marker='o', label=get_model_display_name(model), color=color, markersize=10, linewidth=2)

    ax2.set_xlabel('Complexity', fontsize=25)
    ax2.set_ylabel('Accuracy (%)', fontsize=30)
    ax2.set_title('Accuracy by Complexity', fontsize=35)
    ax2.grid(True)
    ax2.set_xticks([1, 2, 3])
    ax2.set_xticklabels(['Low', 'Medium', 'High'])
    ax2.tick_params(axis='both', which='major', labelsize=26)  # Increased by 30%

    # Create a single legend for both subplots positioned on the right
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.0, 0.5), fontsize=30)  # Increased font size by 0.5x (20 -> 30)

    plt.tight_layout()
    plt.subplots_adjust(right=0.70, hspace=0.3)  # Reduced right margin to leave more space for legend
    # ----------------

    # SAVE FIRST (vector, high quality)
    fig.savefig(PLOTS_DIR / "figure03-up.pdf", bbox_inches="tight", dpi=600)
    fig.savefig(PLOTS_DIR / "figure03-up.png", bbox_inches="tight")
    # Close it. The inline backend auto-displays every open figure at the end of
    # a notebook cell, so leaving it open embeds a second full-resolution copy
    # alongside whatever the cell meant to show.
    plt.close(fig)

    return fig


def plot_integrated(df, out_dir=None, models=None):
    """Integrated Formulae Analysis: type distribution and volume."""
    model_names = MODEL_ORDER if models is None else models
    volume_data, total_class_data, accurate_class_data = process_formula_data(df, model_names)
    PLOTS_DIR = Path(out_dir) if out_dir else plots_dir()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # See plot_equivalence_and_accuracy: style is set here rather than left as
    # global state between the two figures.
    plt.rcdefaults()
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"font.family": "serif", "font.size": 10})
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
    plt.subplots_adjust(top=0.84, bottom=0.12, left=0.13, right=0.83, hspace=0.25)

    # Define scientific color schemes
    class_colors = ['#3498DB', '#2ECC71', '#E74C3C']  # Blue, Green, Red
    class_labels = ['Known sequence', 'Pure math', 'Not found']

    # ============ INTEGRATED VISUALIZATION: STACKED BARS WITH VOLUME OVERLAY ============
    display_models = [get_model_display_name(model) for model in model_names]
    x_positions = np.arange(len(model_names))
    bar_width = 0.4
    volume_axes = []

    for idx, complexity in enumerate([1, 2, 3]):
        ax = axes[idx]
    
        # Get data for this complexity
        total_data = total_class_data[total_class_data['Complexity'] == complexity]
        accurate_data = accurate_class_data[accurate_class_data['Complexity'] == complexity]
        volume_data_comp = volume_data[volume_data['Complexity'] == complexity]
    
        # Plot total formulas (left bars) - with lighter colors
        bottom_total = np.zeros(len(model_names))
        for i, classification in enumerate(class_labels):
            class_counts = []
            for model in model_names:
                count = total_data[(total_data['Model'] == model) & 
                                  (total_data['Classification'] == classification)]['Count']
                class_counts.append(count.iloc[0] if len(count) > 0 else 0)
        
            ax.bar(x_positions - bar_width/2, class_counts, bar_width, bottom=bottom_total,
                   label=f'{classification} (Total)' if idx == 0 else "",
                   color=class_colors[i], alpha=0.4, edgecolor='white', linewidth=0.5)
            bottom_total += np.array(class_counts)
    
        # Plot accurate formulas (right bars) - with full colors
        bottom_accurate = np.zeros(len(model_names))
        for i, classification in enumerate(class_labels):
            class_counts = []
            for model in model_names:
                count = accurate_data[(accurate_data['Model'] == model) & 
                                     (accurate_data['Classification'] == classification)]['Count']
                class_counts.append(count.iloc[0] if len(count) > 0 else 0)
        
            ax.bar(x_positions + bar_width/2, class_counts, bar_width, bottom=bottom_accurate,
                   label=f'{classification} (Accurate)' if idx == 0 else "",
                   color=class_colors[i], alpha=0.9, edgecolor='white', linewidth=0.5)
            bottom_accurate += np.array(class_counts)
    
        # ============ OVERLAY: LINE PLOT FOR VALID FORMULA VOLUME ============
        ax2 = ax.twinx()
        volume_axes.append(ax2)
    
        volume_counts = []
        for model in model_names:
            count = volume_data_comp[volume_data_comp['Model'] == model]['Count']
            volume_counts.append(count.iloc[0] if len(count) > 0 else 0)
    
        # Plot volume as a line with markers on secondary axis
        ax2.plot(x_positions, volume_counts, 
                        color='black', linewidth=1.2, marker='D', 
                        markersize=4, markerfacecolor='gold', 
                        markeredgecolor='black', markeredgewidth=0.8,
                        label='Valid Formulas Volume' if idx == 0 else "",
                        zorder=10, alpha=0.7)
    
        # Styling for primary axis (Formula Types)
        ax.set_title(f'Complexity {complexity}', fontweight='bold', pad=16)
        ax.set_xticks(x_positions)
        ax.set_axisbelow(True)
        ax.grid(True, alpha=1, axis='y', zorder=0)
        ax.grid(True, alpha=0.15, axis='x', zorder=0, linewidth=0.8)
        for x in x_positions:
            ax.axvline(x=x, color='black', alpha=0.10, linewidth=0.8, zorder=0)
        ax.set_ylabel('Formulae Type Distribution Count', labelpad=12, fontsize=22)
    
        # Styling for secondary axis (Volume)
        ax2.set_ylabel('Valid Formulas Volume', color='darkgoldenrod', fontweight='bold', labelpad=14)
        ax2.tick_params(axis='y', labelcolor='darkgoldenrod')
        ax2.spines['right'].set_color('darkgoldenrod')
        ax2.spines['right'].set_linewidth(2)
        ax2.yaxis.set_label_position("right")
        ax2.yaxis.tick_right()
        max_vol = max(volume_counts) if volume_counts else 0
        if max_vol <= 3:
            ax2.set_yticks([0, 1, 2, 3])
            ax2.set_ylim(0, 3)
    
        # Add subtle background shading to distinguish volume line
        ax.axhspan(ax.get_ylim()[0], ax.get_ylim()[1], alpha=0.02, color='yellow', zorder=0)
    
        # Set consistent y-axis limits for better comparison
        if idx == 0:
            ax.set_ylim(0, 70)
        elif idx == 1:
            ax.set_ylim(0, 65)

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

    fig.legend(sorted_handles, sorted_labels, 
              loc='upper center', bbox_to_anchor=(0.5, 0.95),
              ncol=4, frameon=True, fancybox=True, shadow=True)

    # Add overall title
    fig.suptitle('Integrated Formulae Analysis: Type Distribution and Volume by Model Complexity', 
                fontweight='bold', y=0.98, fontsize=35)

    fig.savefig(PLOTS_DIR / "figure03-bottom.pdf", bbox_inches="tight", dpi=600)
    fig.savefig(PLOTS_DIR / "figure03-bottom.png", bbox_inches="tight")
    plt.close(fig)

    return fig


def print_summary(volume_data, total_class_data, accurate_class_data):
    """The console report the original script printed after the figures."""
    class_labels = list(CLASS_LABELS)
    # ============ COMPREHENSIVE STATISTICAL SUMMARY ============
    print("\n" + "="*90)
    print("UNIFIED FORMULA ANALYSIS RESULTS")
    print("="*90)

    print("\n📊 FORMULA GENERATION VOLUME")
    print("-" * 50)
    volume_summary = volume_data.pivot(index='Model', columns='Complexity', values='Count')
    volume_summary.columns = [f'Complexity {int(i)}' for i in volume_summary.columns]
    volume_summary.index = [get_model_display_name(model) for model in volume_summary.index]
    print(volume_summary.to_string())

    print("\n📝 TOTAL FORMULA TYPE DISTRIBUTION")
    print("-" * 50)
    for classification in class_labels:
        print(f"\n{classification}:")
        class_summary = total_class_data[total_class_data['Classification'] == classification].pivot(
            index='Model', columns='Complexity', values='Count').fillna(0)
        class_summary.columns = [f'Complexity {int(i)}' for i in class_summary.columns]
        class_summary.index = [get_model_display_name(model) for model in class_summary.index]
        print(class_summary.head(8).to_string())

    print("\n✅ ACCURATE FORMULA TYPE DISTRIBUTION")
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
    total_volume = volume_data.groupby('Model')['Count'].sum().sort_values(ascending=False)
    total_volume.index = [get_model_display_name(model) for model in total_volume.index]
    print(f"• Highest formula generation volume: {total_volume.index[0]} ({total_volume.iloc[0]:.0f} formulas)")

    # Most consistent across complexities
    volume_consistency = volume_data.pivot(index='Model', columns='Complexity', values='Count').std(axis=1).sort_values()
    volume_consistency.index = [get_model_display_name(model) for model in volume_consistency.index]
    print(f"• Most consistent generation across complexities: {volume_consistency.index[0]}")

    # Accuracy insights
    total_accurate = accurate_class_data.groupby('Model')['Count'].sum().sort_values(ascending=False)
    total_accurate.index = [get_model_display_name(model) for model in total_accurate.index]
    print(f"• Highest accurate formula count: {total_accurate.index[0]} ({total_accurate.iloc[0]:.0f} accurate formulas)")

    print("\n" + "="*90)



def emit_summary_measures(df, equiv_df, volume_data, total_class_data, model_names=None):
    """Write the measures the family-evolution figures consume.

    These numbers were previously only drawn, never saved, so the summary
    driving Supplementary Figures 5 and 6 had to be maintained by hand. Emitting
    them keys every value to a canonical model name via the registry.

    Two accuracy definitions are in play and they answer different questions.
    ``Accuracy`` is the fraction of sequences the model got right *at all* --
    any one of its formulae reproduced the target -- which is what the evolution
    figures were built from. ``Accuracy (per answer)`` is the fraction of the
    model's individual answers that were right, which is what the formulae
    figures plot. Both are emitted so neither figure silently borrows the
    other's meaning.
    """
    from .model_summary import CASE_FORMULAE as _CASE, write_long as _write_long
    from .registry import by_column as _by_column

    model_names = MODEL_ORDER if model_names is None else model_names

    def _canonical(key):
        model = _by_column(A_FORMULA, key)
        return model.name if model is not None else None

    _records = []

    # Two accuracy definitions are in play, and they answer different questions.
    #   'Accuracy'            -- on what fraction of sequences did the model get it
    #                            right AT ALL (any of its formulae reproduced the
    #                            target). This is what Supplementary Figures 5 and 6
    #                            were built from; verified to reproduce the legacy
    #                            comparison_models_summary.csv exactly.
    #   'Accuracy (per answer)' -- what fraction of the model's individual answers
    #                            were right. This is what Figures 3 and 4 plot.
    # Both are emitted so neither figure silently borrows the other's meaning.
    for _model in model_names:
        _name = _canonical(_model)
        if _name is None:
            continue
        _acc_cols = [f'accuracy-test-{_c}' for _c in _eval_cols(df, _model)]
        _acc_cols = [_c for _c in _acc_cols if _c in df.columns]
        if not _acc_cols:
            continue
        for _cx in (1, 2, 3):
            _sub = df[df['Complexity'] == _cx]
            _solved = (_sub[_acc_cols].astype(bool).any(axis=1)).mean() * 100
            _records.append({'model': _name, 'complexity': _cx,
                             'measure': 'Accuracy', 'value': float(_solved)})
            _per_answer = _sub[_acc_cols].astype(float).mean(axis=1).mean() * 100
            _records.append({'model': _name, 'complexity': _cx,
                             'measure': 'Accuracy (per answer)', 'value': float(_per_answer)})

    for _, _r in equiv_df.iterrows():
        _name = _canonical(_r['Model'])
        if _name is not None:
            _records.append({'model': _name, 'complexity': int(_r['Complexity']),
                             'measure': 'Equivalence', 'value': float(_r['Equivalence %'])})

    for _, _r in volume_data.iterrows():
        _name = _canonical(_r['Model'])
        if _name is not None:
            _records.append({'model': _name, 'complexity': int(_r['Complexity']),
                             'measure': 'Valid Instances', 'value': float(_r['Count'])})

    for _, _r in total_class_data.iterrows():
        _name = _canonical(_r['Model'])
        if _name is not None:
            _records.append({'model': _name, 'complexity': int(_r['Complexity']),
                             'measure': _r['Classification'], 'value': float(_r['Count'])})

    # The original wrote the file from inside its own print statement, so
    # lifting it into a function that also returns the path wrote it twice.
    # Harmless -- the content is identical -- but there is no reason to.
    path = _write_long(_CASE, _records)
    print(f"\nemitted {len(_records)} summary measures to {path}")
    return path
