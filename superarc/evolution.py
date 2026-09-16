"""Family-evolution figures: Supplementary Figures 5 and 6.

Owner of the two panels printed on pages 32 and 33 of the Supplementary
Information -- "Evolution of Model Performance (Part 1/2)" -- and of the two
CSVs they are drawn from.

The bodies below are the published ones, sliced verbatim out of
``35_summary_statistics.py``, which is now a forwarder onto this module. Four
departures, each declared:

EDIT 1 -- ``FAMILY_ORDER`` is derived from :mod:`superarc.registry` rather than
    hand-written. The two were diffed elementwise before collapsing, as
    ``tests/test_structure.py`` required: six of the eight families are
    character-for-character identical, and the differences are inert or
    cosmetic.

    * the script listed ``'Grok'`` ahead of ``Grok-3``. No model of that name
      exists in any dataset, so ``order.index`` merely shifted every Grok rank
      by one, uniformly -- the drawn order is unchanged.
    * the script listed ``'DeepSeek-R1-0525'``, annotated "Included both
      versions for safety". It is not in the data either. Which of 0525/0528 is
      authoritative is still an open question for the authors; it is not decided
      here, because neither figure ever drew it.
    * the script keyed the family ``'Deepseek'``; the registry spells it
      ``'DeepSeek'``. Cosmetic, and internally consistent in both, because each
      spelling was paired with its own ``get_family``. Hence EDIT 2.

EDIT 2 -- ``get_family`` returns ``'DeepSeek'``, paired with EDIT 1. Changing
    one without the other silently drops the whole family out of the figures,
    which is precisely the failure mode ``KNOWN_DUPLICATES`` was declared to
    prevent.

EDIT 3 -- the input is read through ``REPO_ROOT`` instead of a bare relative
    name, so the module works from a notebook subdirectory. Same edit, same
    reason, as in :mod:`superarc.languages`.

EDIT 4 -- ``produce()`` replaces the ``__main__`` block, so a notebook and the
    parity gate call the same entry point.

Verified after collapsing: both CSVs are byte-identical to the pre-collapse run,
Part 2 is pixel-identical to ``plots/highResolution/figure15.pdf``, and Part 1
sits 0.894% from ``figure14.pdf`` -- the Gemini-2.5-Pro column correction already
recorded in ``superarc.parity``, and Gemini is a Part 1 family.

NINE families are in the data; EIGHT are drawn. ``cursor_small`` is in
``comparison_models_summary.csv``, is resolved by the registry, and appears in
both output CSVs (three rows each, sorted last) -- but the published script
splits ``FAMILY_ORDER`` four and four, and Cursor is in neither half, so
Cursor-Small has never been plotted. The printed captions name eight families,
so this is left exactly as published rather than corrected. It is recorded here
because a family that is in the table and not in the figure is invisible
otherwise.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns  # noqa: F401  imported by the published script; kept because
                       # importing it applies no style here, but removing an import
                       # is a change to the environment the figures were drawn in

from . import REPO_ROOT, plots_dir as _plots_dir
from .registry import (
    A_FORMULA as _A,
    B_SCRIPT as _B,
    FAMILY_ORDER as _REGISTRY_FAMILY_ORDER,
    display_name as _display_name,
    models_for as _models_for,
    sorted_by_family as _sorted_by_family,
)

def _name_map(dataset):
    """Column -> published label, from the registry that owns model identity.

    These were two hand-written dicts. Checked elementwise before removal:
    FORMULA_NAME_MAP agreed with the registry on all 28 entries, and
    SCRIPT_NAME_MAP on 27 of 28. The single disagreement is the Gemini column of
    the script dataset, and it is a real defect rather than a naming convention
    -- see the note below.
    """
    return {m.column(dataset): _display_name(dataset, m.column(dataset), published=True)
            for m in _models_for(dataset)}


# The script dataset carries three Gemini columns -- `gemini`, `gemini-thinking`
# and `gemini-2.5-pro` -- all fully populated, with 119, 162 and 750 scripts.
# The map this replaces sent `gemini` to "Gemini-2.5-Pro" and had no entry for
# `gemini-2.5-pro` at all, so these figures labelled the 119-script column
# Gemini-2.5-Pro and discarded the 750-script one. That is the same defect
# already corrected in the Integrated Script Analysis, where the valid-script
# volume went from 82 to 750; this file was never corrected with it, so the two
# figures in the paper disagreed with each other.
SCRIPT_NAME_MAP = _name_map(_B)
FORMULA_NAME_MAP = _name_map(_A)

# --- Mappings & Configuration -------------------------------------------------

# The eight families the published figures draw, in the published order. Cursor
# is in the registry and in the data; it is not here, because it was not printed.
# See the module docstring.
PUBLISHED_FAMILIES: tuple[str, ...] = tuple(
    f for f in _REGISTRY_FAMILY_ORDER if f != "Cursor"
)


def _family_order() -> dict[str, list[str]]:
    """EDIT 1: the within-family order, from the registry's ``Model.order``."""
    order: dict[str, list[str]] = {f: [] for f in PUBLISHED_FAMILIES}
    for model in _sorted_by_family():
        if model.family in order:
            order[model.family].append(model.name)
    return order


FAMILY_ORDER = _family_order()

# EDIT 5: the spelling printed in the Supplementary Information, which is not the
# registry's. Supplementary Figures 5 and 6 title two subplots "Deepseek -
# Scripts" and "Deepseek - Formulae"; the registry spells the family "DeepSeek",
# as the model labels do. Taking the registry spelling through to the canvas
# moved 0.0937% of Part 2's pixels -- two subplot titles, one character each --
# which is a change to a published figure for no reason beyond internal
# consistency. The key stays "DeepSeek" everywhere it is a key; only the drawn
# title is the published string. Re-measured after this: Part 2 is 0.0000%.
PUBLISHED_FAMILY_LABEL = {"DeepSeek": "Deepseek"}

TYPE_COLORS = {
    'Known sequence': '#2ca02c',  # Green
    'Pure math': '#1f77b4',       # Blue
    'Print': '#ff7f0e',           # Orange
    'Not found': '#d62728'        # Red
}

STACK_ORDER = ['Known sequence', 'Pure math', 'Print', 'Not found']
VALID_INSTANCES_COLOR = '#006400' # Dark Green

def get_family(model_name):
    m = model_name.lower()
    if 'gpt' in m or 'o1' in m:
        return 'OpenAI'
    elif 'claude' in m or 'opus' in m:
        return 'Claude'
    elif 'gemini' in m:
        return 'Gemini'
    elif 'llama' in m or 'meta' in m:
        return 'Meta'
    elif 'qwen' in m:
        return 'Qwen'
    elif 'mistral' in m or 'mixtral' in m:
        return 'Mistral'
    elif 'grok' in m:
        return 'Grok'
    elif 'deepseek' in m:
        return 'DeepSeek'
    elif 'cursor' in m:
        return 'Cursor'
    else:
        return 'Other'

def process_data_and_generate_csvs():
    # Load Raw Data
    try:
        df = pd.read_csv(REPO_ROOT / 'comparison_models_summary.csv')
    except FileNotFoundError:
        print("Error: comparison_models_summary.csv not found.")
        return None, None

    # Melt and Pivot
    id_vars = ['case', 'Measure', 'complexity']
    model_cols = [c for c in df.columns if c not in id_vars]
    melted = df.melt(id_vars=id_vars, value_vars=model_cols, var_name='model', value_name='value')
    
    pivot_df = melted.pivot_table(
        index=['case', 'complexity', 'model'], 
        columns='Measure', 
        values='value'
    ).reset_index()

    # Written alongside the figures so a parity run can redirect everything
    # this script produces with one environment variable.
    _base = str(_plots_dir())
    case_map = {
        'multiple script': os.path.join(_base, 'multi_script'),
        'multiple formulae': os.path.join(_base, 'multi_formula'),
    }

    consolidated_dfs = {}

    for case_name, output_dir in case_map.items():
        print(f"Processing {case_name}...")
        os.makedirs(output_dir, exist_ok=True)
        
        is_formula = 'formula' in case_name
        name_map = FORMULA_NAME_MAP if is_formula else SCRIPT_NAME_MAP
        target_values = list(name_map.values()) # The allowed list of consolidated names
        
        df_case = pivot_df[pivot_df['case'] == case_name].copy()
        if df_case.empty:
            continue

        # --- MAPPING & FILTERING LOGIC ---
        # 1. Map Raw Name -> Target Name
        # 2. Filter: Keep ONLY if Target Name is in the Values List of the Map
        
        def map_and_validate(raw_name):
            if raw_name in name_map:
                mapped = name_map[raw_name]
                # Ensure mapped name is in the target values (redundant if using dict, but safe)
                if mapped in target_values:
                    return mapped
            return None # Drop if not in map keys or mapped value not in target list
            
        df_case['clean_model'] = df_case['model'].apply(map_and_validate)
        
        # Drop rows where model wasn't valid
        df_case = df_case.dropna(subset=['clean_model'])
        
        if df_case.empty:
            print(f"No valid models found for {case_name} after filtering.")
            consolidated_dfs[case_name] = pd.DataFrame()
            continue

        # --- CONSOLIDATION ---
        # Group by Complexity + Clean Model -> Take Max
        numeric_cols = [c for c in df_case.columns if c not in ['case', 'complexity', 'model', 'clean_model']]
        
        df_consolidated = df_case.groupby(['complexity', 'clean_model'])[numeric_cols].max().reset_index()
        df_consolidated = df_consolidated.rename(columns={'clean_model': 'model'})
        df_consolidated['case'] = case_name # Restore case column
        
        # --- SAVE CSV ---
        # Define columns to save
        save_cols = ['case', 'complexity', 'model'] + STACK_ORDER + ['Accuracy', 'Equivalence', 'Valid Instances']
        save_cols = [c for c in save_cols if c in df_consolidated.columns]
        
        # Sort Chronologically (using FAMILY_ORDER)
        # We need a sort key.
        # Assign a rank based on FAMILY_ORDER
        def get_sort_key(row):
            fam = get_family(row['model'])
            model = row['model']
            
            fam_rank = list(FAMILY_ORDER.keys()).index(fam) if fam in FAMILY_ORDER else 100
            
            model_order = FAMILY_ORDER.get(fam, [])
            model_rank = model_order.index(model) if model in model_order else 100
            
            return (fam_rank, model_rank, row['complexity'])

        # We can't easily sort the dataframe with a complex key function directly in sort_values
        # So create a temporary rank column
        df_consolidated['sort_rank'] = df_consolidated.apply(
            lambda r: get_sort_key(r), axis=1
        )
        # Tuples compare element-wise, so this works for sorting
        # But pandas needs columns. Let's make it simple.
        # Just save it; the plotter does its own sorting.
        # The prompt says "summary csv should be sorted".
        # Let's try to sort it nicely.
        
        # Add explicit rank columns
        df_consolidated['fam_rank'] = df_consolidated['model'].apply(lambda m: list(FAMILY_ORDER.keys()).index(get_family(m)) if get_family(m) in FAMILY_ORDER else 100)
        df_consolidated['model_rank'] = df_consolidated.apply(lambda r: FAMILY_ORDER.get(get_family(r['model']), []).index(r['model']) if r['model'] in FAMILY_ORDER.get(get_family(r['model']), []) else 100, axis=1)
        
        df_consolidated = df_consolidated.sort_values(['fam_rank', 'model_rank', 'complexity'])
        
        # Drop helper columns
        df_final = df_consolidated.drop(columns=['fam_rank', 'model_rank', 'sort_rank'], errors='ignore')

        csv_path = os.path.join(output_dir, f"{case_name.replace(' ', '_')}_values.csv")
        df_final[save_cols].to_csv(csv_path, index=False)
        print(f"Saved sorted summary CSV to {csv_path}")
        
        consolidated_dfs[case_name] = df_final

    return consolidated_dfs

def aggregate_data(df):
    """
    Aggregates data across complexities to get one row per model.
    """
    if df is None or df.empty:
        return None
    
    count_cols = [c for c in STACK_ORDER if c in df.columns]
    if 'Valid Instances' in df.columns:
        count_cols.append('Valid Instances')
        
    pct_cols = ['Accuracy', 'Equivalence']
    
    df_counts = df.groupby('model')[count_cols].sum()
    df_pcts = df.groupby('model')[pct_cols].mean()
    
    result = pd.concat([df_counts, df_pcts], axis=1).reset_index()
    
    stack_cols_only = [c for c in STACK_ORDER if c in result.columns]
    result['Total'] = result[stack_cols_only].sum(axis=1)
    
    return result

def plot_evolution_subset(df_script, df_formula, families, part_num):
    # Overridable so a parity run can regenerate into a scratch path.
    output_dir = str(_plots_dir())
    
    n_rows = len(families)
    n_cols = 2 
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(26, 6 * n_rows), squeeze=False)
    fig.suptitle(f'Evolution of Model Performance by Family (Part {part_num})', fontsize=40, fontweight='bold', y=0.99)
    
    # Plot Helper
    def plot_subplot(ax, data, title_suffix, fam_name):
        if data.empty:
            ax.axis('off')
            return
        
        models = data['model'].tolist()
        x = np.arange(len(models))
        
        bottom = np.zeros(len(models))
        totals = data['Total'].replace(0, 1)
        
        for type_name in STACK_ORDER:
            if type_name not in data.columns: continue
            raw_vals = data[type_name].values
            pct_vals = (raw_vals / totals) * 100
            ax.bar(x, pct_vals, bottom=bottom, color=TYPE_COLORS[type_name], alpha=0.85, width=0.6)
            bottom += pct_vals
            
        ax2 = ax.twinx()
        if 'Accuracy' in data.columns:
            ax2.plot(x, data['Accuracy'], color='black', marker='o', 
                     linewidth=4, markersize=12, zorder=10)
        if 'Equivalence' in data.columns:
            ax2.plot(x, data['Equivalence'], color='purple', marker='D', 
                     linewidth=4, markersize=10, linestyle='--', markerfacecolor='white', zorder=10)
        
        ax3 = ax.twinx()
        ax3.spines["right"].set_position(("axes", 1.15))
        ax3.set_frame_on(True)
        ax3.patch.set_visible(False)
        
        if 'Valid Instances' in data.columns:
            ax3.plot(x, data['Valid Instances'], color=VALID_INSTANCES_COLOR, marker='*', 
                     linewidth=3, markersize=14, linestyle='-.', zorder=15)
            ax3.set_ylim(bottom=0)
            ax3.tick_params(axis='y', colors=VALID_INSTANCES_COLOR, labelsize=14)
            ax3.set_ylabel('Valid Instances (Count)', fontsize=16, color=VALID_INSTANCES_COLOR)
        else:
            ax3.axis('off')

        printed = PUBLISHED_FAMILY_LABEL.get(fam_name, fam_name)  # EDIT 5
        ax.set_title(f"{printed} - {title_suffix}", fontsize=24, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha='right', fontsize=16, fontweight='bold')
        ax.tick_params(axis='y', labelsize=14)
        ax2.tick_params(axis='y', labelsize=14)
        
        ax.set_ylim(0, 100)
        ax2.set_ylim(0, 110)
        
        ax.set_ylabel('Methodology (%)', fontsize=16)
        ax2.set_ylabel('Performance (%)', fontsize=16)
        
        ax.grid(axis='y', linestyle=':', alpha=0.5)

    for row_idx, fam in enumerate(families):
        # Prepare Data
        fam_script = df_script[df_script['model'].apply(get_family) == fam].copy() if df_script is not None else pd.DataFrame()
        fam_formula = df_formula[df_formula['model'].apply(get_family) == fam].copy() if df_formula is not None else pd.DataFrame()
        
        # Sort Data (Using FAMILY_ORDER)
        order = FAMILY_ORDER.get(fam, [])
        def get_rank(name):
            if name in order: return order.index(name)
            return 100 + len(name)
            
        if not fam_script.empty:
            fam_script['rank'] = fam_script['model'].apply(get_rank)
            fam_script = fam_script.sort_values('rank')
        
        if not fam_formula.empty:
            fam_formula['rank'] = fam_formula['model'].apply(get_rank)
            fam_formula = fam_formula.sort_values('rank')
            
        plot_subplot(axes[row_idx, 0], fam_script, "Scripts", fam)
        plot_subplot(axes[row_idx, 1], fam_formula, "Formulae", fam)

    # Legend
    handles = []
    labels = []
    
    for t in STACK_ORDER:
        handles.append(plt.Rectangle((0,0),1,1, color=TYPE_COLORS[t]))
        labels.append(t)
        
    handles.append(plt.Line2D([0], [0], color='black', marker='o', lw=4, markersize=16))
    labels.append('Accuracy')
    handles.append(plt.Line2D([0], [0], color='purple', marker='D', lw=4, markersize=14, linestyle='--', markerfacecolor='white'))
    labels.append('Equivalence')
    handles.append(plt.Line2D([0], [0], color=VALID_INSTANCES_COLOR, marker='*', lw=3, markersize=14, linestyle='-.'))
    labels.append('Valid Instances')
    
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), 
               ncol=len(labels), fontsize=24, frameon=False)
    
    plt.tight_layout(rect=[0, 0, 0.92, 0.96])
    print("---------------------------------")
    print("---------------------------------")
    print(output_dir)
    save_path = f"{output_dir}/ALL_FAMILIES_evolution_Part{part_num}.pdf"
    plt.savefig(save_path, bbox_inches='tight', dpi=600)
    
    save_path = f"{output_dir}/ALL_FAMILIES_evolution_Part{part_num}.svg"
    plt.savefig(save_path, bbox_inches='tight', dpi=600)
    
    print(f"Saved combined plot to {save_path}")
    plt.close()


def produce(out_dir=None):
    """Regenerate Supplementary Figures 5 and 6, and the two CSVs behind them.

    EDIT 4: what the ``__main__`` block did, as a function, so the notebook and
    ``superarc.parity`` drive the identical path.
    """
    print("Starting 35_summary_statistics...")
    dfs = process_data_and_generate_csvs()
    if not (dfs and 'multiple script' in dfs and 'multiple formulae' in dfs):
        raise RuntimeError("Could not generate plots due to missing data.")

    df_script_agg = aggregate_data(dfs['multiple script'])
    df_formula_agg = aggregate_data(dfs['multiple formulae'])

    all_families = list(FAMILY_ORDER.keys())
    part1_families = all_families[:4]
    part2_families = all_families[4:]

    plot_evolution_subset(df_script_agg, df_formula_agg, part1_families, 1)
    plot_evolution_subset(df_script_agg, df_formula_agg, part2_families, 2)
    return df_script_agg, df_formula_agg


if __name__ == "__main__":
    produce()
