import pandas as pd
import numpy as np
from pathlib import Path

# Output directory. Overridable so a verification run can regenerate the
# figures into a scratch path and diff them against the published ones
# without touching the tracked originals.
import os as _os
PLOTS_DIR = Path(_os.environ.get("SUPERARC_PLOTS_DIR",
                                 Path(__file__).resolve().parent / "new_plots"))
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv('multi-python-script-time-series.csv')
df

# 'chatgpt-o1' is replaced by 'chatgpt-o1'
model_names = [
       'chatgpt-4o',
       'gpt-4o-mini',
       'gemini-2.5-pro',
       'gemini',
       'gemini-thinking',
       'claude-3-5-sonnet',
       'chatgpt-o1',
       'mistral',
       'meta',
       'cursor_small',
       'o1_mini',
       'grok',
       'grok-3', 
       'qwen',
       'deepseek',
       'chatgpt-4.5',
       'claude-3.7',
       'deepseek_r1_0525',
       'llama_4_scout',
       'qwen3',
       'chatgpt_5',
       'opus_4',
       'mistral_large2405',
       'claude_sonnet_4',
       'grok_4.1',
       'gpt-5.2',
       'claude-4.5',
       'gemini-3-pro',
       'mistral-large-3',
       
       ]

# Function to split string lists into separate columns
def split_model_scripts(df, model_name):
    # Convert string representation of list to actual list and get max length
    def safe_eval(x):
        if pd.isna(x) or x == "" or x == "*not found":
            return ["*not found"]  # Return "*not found" for empty/nan/not found cases
        return eval(x)
    
    df[model_name] = df[model_name].apply(safe_eval)
    max_scripts = df[model_name].str.len().max()
    
    # Create new columns for each script
    for i in range(max_scripts):
        col_name = f"{model_name}_script_{i+1}"
        df[col_name] = df[model_name].apply(lambda x: x[i] if i < len(x) else "*not found")
    
    # Drop original column
    df.drop(columns=[model_name], inplace=True)

# Apply the split function to each model column
for model in model_names:
    split_model_scripts(df, model)
    
df.columns

df['chatgpt-4.5_script_3'].head()

# Execute generated scripts under the declared environment.
#
# This replaces an inline exec(code, globals()) that had two problems: it let
# every script import whatever happened to be installed on the machine, and it
# ran all ~12,000 scripts in one shared namespace.
#
# The import question is not cosmetic. Four o1-Preview scripts are
# `import sympy; print(list(sympy.primerange(2,31)))`. The published run had no
# sympy installed so they failed; with sympy present they succeed and
# o1-Preview's complexity-2 accuracy moves from 20.00% to 26.67%, changing
# published Figures 5 and 6. superarc.sandbox pins this to the standard library.
#
# Namespace sharing is deliberately preserved (Executor defaults to
# isolate=False) because it is what produced the published figures. See
# superarc/sandbox.py for the measured cost of switching it off.
from superarc.sandbox import Executor
from superarc.registry import B_SCRIPT as _B_SCRIPT, by_column as _by_column, validate_columns as _validate_columns

# Fail loudly if the data contains a model column the registry does not know.
# Previously any unmapped column was dropped from the figures without a word,
# which is how a Gemini column silently vanished from each of Figures 5 and 6.
_validate_columns(_B_SCRIPT, [c for c in model_names])

execute_code_safely = Executor()

# Get list of script columns
script_columns = [col for col in df.columns if 'script' in col]

# Create new columns for results
for col in script_columns:
    result_col = f"{col}_result"
    df[result_col] = df[col].apply(execute_code_safely)

df.columns


#### REARRANGING COLUMNS

# Get all script columns (excluding result columns)
script_cols = [col for col in df.columns if '_script_' in col and not col.endswith('_result')]

# Create new column order
new_cols = ['sequence', 'Complexity']  # Keep these first

# For each script column, add it and its corresponding result column
for script_col in script_cols:
    result_col = f"{script_col}_result"
    new_cols.extend([script_col, result_col])

# Reorder the DataFrame columns
df = df[new_cols]

# Verify new column order
df.columns

# Get unique complexity values
complexities = df['Complexity'].unique()

# Get list of script columns and their corresponding result columns
script_cols = [col for col in df.columns if '_script_' in col]
result_cols = [col for col in df.columns if '_result' in col]

# Group columns by model
model_groups = {}
for model in model_names:
    model_groups[model] = {
        'scripts': [col for col in script_cols if model in col],
        'results': [col for col in result_cols if model in col]
    }

print("Random sampling of scripts and their evaluation results by model and complexity:")
print("-" * 80)

# For each complexity level
for complexity in complexities:
    print(f"\nComplexity Level: {complexity}")
    print("-" * 40)
    
    # For each model
    for model in model_names:
        print(f"\n{model.upper()}:")
        
        # Get rows for current complexity
        complexity_mask = df['Complexity'] == complexity
        subset = df[complexity_mask]
        
        # Sample 10 rows (or less if fewer rows exist)
        sample_size = min(10, len(subset))
        sampled_rows = subset.sample(n=sample_size, random_state=42)
        
        # For each sampled row
        for idx, row in sampled_rows.iterrows():
            print("\nSequence:", row['sequence'])
            
            # Print each script and its result for this model
            for script_col, result_col in zip(model_groups[model]['scripts'], 
                                            model_groups[model]['results']):
                print(f"\n{script_col}:")
                print(row[script_col])
                print(f"{result_col}:")
                print(row[result_col])
                
        print("-" * 40)

# Get list of result columns
result_columns = [col for col in df.columns if '_result' in col]

# Function to clean and format result strings
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
    except:
        return result

# Create new formatted columns
for col in result_columns:
    new_col = f"{col}_formatted"
    df[new_col] = df[col].apply(clean_result)

df.columns

## Rearranging columns

# Get all model script columns and their formatted results
script_cols = [col for col in df.columns if '_script_' in col and not ('_result' in col or '_formatted' in col or '_accuracy' in col or '_classification' in col)]
formatted_result_cols = [col for col in df.columns if '_result_formatted' in col]

# Create pairs of script and formatted result columns
column_pairs = []
for script_col in script_cols:
    base_name = script_col.replace('_script_', '_script_')
    formatted_col = f"{script_col}_result_formatted"
    if formatted_col in formatted_result_cols:
        column_pairs.append((script_col, formatted_col))

# Create list of columns in desired order
new_column_order = ['sequence', 'Complexity']
for script_col, formatted_col in column_pairs:
    new_column_order.extend([script_col, formatted_col])

# Add any remaining columns that weren't paired
remaining_cols = [col for col in df.columns if col not in new_column_order]
new_column_order.extend(remaining_cols)

# Reorder the dataframe columns
df = df[new_column_order]

# Get list of formatted result columns
formatted_result_columns = [col for col in df.columns if '_result_formatted' in col]

# Function to compare sequences
def compare_sequences(seq1, seq2):
    # Remove any whitespace and split on commas
    seq1_clean = [x.strip() for x in seq1.split(',')]
    seq2_clean = [x.strip() for x in seq2.split(',')]
    return seq1_clean == seq2_clean

# Create accuracy columns
for col in formatted_result_columns:
    # Create new column name by replacing 'result_formatted' with 'accuracy'
    accuracy_col = col.replace('_result_formatted', '_accuracy')
    
    # Compare sequences and store boolean result
    df[accuracy_col] = df.apply(lambda row: compare_sequences(row[col], row['sequence']), axis=1)

# df.columns

# Get list of formatted result columns and their corresponding accuracy columns
formatted_and_accuracy_pairs = []
for col in df.columns:
    if '_result_formatted' in col:
        formatted_col = col
        accuracy_col = col.replace('_result_formatted', '_accuracy')
        formatted_and_accuracy_pairs.append((formatted_col, accuracy_col))

# Create new column order
new_column_order = ['sequence', 'Complexity']

# Add script columns and their corresponding formatted/accuracy columns
for script_col in [col for col in df.columns if '_script_' in col and not '_result' in col]:
    new_column_order.append(script_col)
    formatted_col = script_col + '_result_formatted'
    accuracy_col = script_col + '_accuracy'
    if formatted_col in df.columns:
        new_column_order.append(formatted_col)
        if accuracy_col in df.columns:
            new_column_order.append(accuracy_col)

# Add any remaining columns that weren't included
remaining_cols = [col for col in df.columns if col not in new_column_order]
new_column_order.extend(remaining_cols)

# Reorder the dataframe columns
df = df[new_column_order]

# Get unique models and complexities
models = ['gemini', 'claude-3-5-sonnet', 'chatgpt-1o', 'mistral', 'meta', 'o1_mini', 'cursor_small', 'gemini-thinking']
complexities = df['Complexity'].unique()

# Create sample log
print("Sample comparison log:")
print("-" * 80)

for model in models:
    print(f"\nModel: {model}")
    print("-" * 40)
    
    for complexity in complexities:
        print(f"\nComplexity {complexity}:")
        
        # Filter rows for this model and complexity
        model_data = df[df['Complexity'] == complexity].copy()
        
        # Get all script columns for this model
        script_cols = [col for col in df.columns if f'{model}_script_' in col and 
                      'result' not in col and 'accuracy' not in col]
        
        # Sample 10 random rows (or all if less than 10)
        samples = model_data.sample(n=min(10, len(model_data)))
        
        for _, row in samples.iterrows():
            print(f"\nOriginal sequence: {row['sequence']}")
            
            # Print info for each script version
            for script_col in script_cols:
                result_col = f"{script_col}_result_formatted"
                accuracy_col = f"{script_col}_accuracy"
                
                print(f"\nScript version {script_col[-1]}:")
                print(f"Generated script:\n{row[script_col]}")
                print(f"Generated sequence: {row[result_col]}")
                print(f"Accurate: {row[accuracy_col]}")

# Get list of script columns (excluding result, accuracy and formatted columns)
script_columns = [col for col in df.columns if '_script_' in col and 
                 'result' not in col and 
                 'accuracy' not in col and 
                 'formatted' not in col]

# Function to classify script type
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

# Create classification columns
for col in script_columns:
    classification_col = f"{col}_classification"
    df[classification_col] = df[col].apply(classify_script)

df.columns

# Get all column names
cols = list(df.columns)

# Create a new column order
new_cols = []

# Always keep sequence and Complexity first
new_cols.extend(['sequence', 'Complexity'])

# Get base script names (without _classification, _result etc)
base_scripts = list(set([col.replace('_classification','') for col in cols if '_classification' in col]))

# For each base script name, add all related columns in order
for base in base_scripts:
    script_col = base
    classification_col = f"{base}_classification"
    result_col = f"{base}_result"
    result_formatted_col = f"{base}_result_formatted"
    accuracy_col = f"{base}_accuracy"
    
    # Add columns if they exist
    if script_col in cols:
        new_cols.append(script_col)
    if classification_col in cols:
        new_cols.append(classification_col)
    if result_col in cols:
        new_cols.append(result_col)
    if result_formatted_col in cols:
        new_cols.append(result_formatted_col)
    if accuracy_col in cols:
        new_cols.append(accuracy_col)

# Reorder the dataframe columns
df = df[new_cols]

# Print classifications for each model and complexity level
for model in model_names:
    print(f"\nModel: {model}")
    print("-" * (len(model) + 7))
    
    # Get script and classification columns for this model
    script_cols = [col for col in script_columns if model in col]
    class_cols = [col for col in df.columns if '_classification' in col and model in col]
    
    for complexity in [1, 2, 3]:
        print(f"\nComplexity Level {complexity}:")
        
        # Filter data for this complexity level
        complexity_data = df[df['Complexity'] == complexity]
        
        # Combine all scripts and classifications for this model/complexity
        all_scripts = []
        all_classifications = []
        
        for script_col, class_col in zip(script_cols, class_cols):
            mask = complexity_data[script_col].notna()
            scripts = complexity_data[mask][script_col]
            classifications = complexity_data[mask][class_col]
            
            all_scripts.extend(list(scripts))
            all_classifications.extend(list(classifications))
            
        # Sample at least 10 cases randomly if available
        sample_size = min(10, len(all_scripts))
        if sample_size > 0:
            indices = np.random.choice(len(all_scripts), sample_size, replace=False)
            sampled_scripts = [all_scripts[i] for i in indices]
            sampled_classifications = [all_classifications[i] for i in indices]
            
            for script, classification in zip(sampled_scripts, sampled_classifications):
                print(f"\nScript: {script}")
                print(f"Classification: {classification}")
                print("-" * 40)

# Calculate equivalence percentage for each model
for model in model_names:
    # Get result columns for this model
    result_cols = [col for col in df.columns if f'{model}_script' in col and '_result_formatted' in col]
    
    # Create new column for equivalence percentage
    equiv_col = f'{model}_equivalence_percentage'
    
    # Calculate equivalence for each row
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
    
    df[equiv_col] = df.apply(calculate_equivalence, axis=1)

df.columns

# Print duplicated columns with exactly the same name
for model in model_names:
    # Get all columns for this model
    model_cols = [col for col in df.columns if model in col]
    
    # Find duplicates by comparing column names
    seen_cols = set()
    for col in model_cols:
        if col in seen_cols:
            print(f"Duplicate column name found for {model}:")
            print(f"  {col}")
            print("-" * 40)
        seen_cols.add(col)

# Print equivalence results for each model
for model in model_names:
    print(f"\n{'-'*40}\nResults for {model}:")
    
    # Get script and equivalence columns
    script_cols = [col for col in df.columns if f'{model}_script' in col and '_result_formatted' in col]
    equiv_col = f'{model}_equivalence_percentage'
    
    # Print results for each row
    for idx, row in df.iterrows():
        print(f"\nSequence {idx}:")
        print("Scripts output:")
        for col in script_cols:
            print(f"{col}: {row[col]}")
        print(f"Equivalence: {row[equiv_col]:.2f}%")

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set scientific plotting style
plt.style.use('seaborn-v0_8-whitegrid')

# Create figure with subplots stacked vertically
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(21.6, 16))
plt.rcParams.update({'font.size': 20, 'font.family': 'serif'})

# Define ordered model list and display name mapping
def get_model_display_name(model_key):
    """Canonical display label for a dataset column.

    Backed by superarc.registry so there is one definition instead of the two
    that previously disagreed: the copy used for Figure 5 mapped
    `gemini-2.5-pro` to "Gemini-2.5-Pro", while the copy used for Figure 6
    mapped `gemini` to it, so the same label denoted different data in the two
    published figures.

    Unregistered keys are returned unchanged, which is what the downstream
    `valid_models` filter uses to exclude declared-retired columns.
    """
    model = _by_column(_B_SCRIPT, model_key)
    return model.name if model is not None else model_key

# Generate distinct markers for scientific clarity
markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', '8']

# ============ PLOT 1: EQUIVALENCE PERCENTAGE ============
# Calculate average equivalence percentage by model and complexity
avg_by_model_complexity = []

for model in model_names:
    equiv_col = f'{model}_equivalence_percentage'
    if equiv_col in df.columns:
        model_data = df[df['Complexity'].isin([1,2,3])].groupby('Complexity')[equiv_col].mean().reset_index()
        model_data['Model'] = model
        avg_by_model_complexity.append(model_data)

# Combine all model data for equivalence
equiv_plot_data = pd.concat(avg_by_model_complexity)

# Plot equivalence lines
for i, model in enumerate(model_names):
    display_name = get_model_display_name(model)
    if display_name != model:  # Only plot if mapped
        model_data = equiv_plot_data[equiv_plot_data['Model'] == model]
        if not model_data.empty:
            ax1.plot(model_data['Complexity'], 
                    model_data[f'{model}_equivalence_percentage'],
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
# Get accuracy columns for each model
accuracy_cols = [col for col in df.columns if '_accuracy' in col]
accuracy_data = []

# Calculate average accuracy for each model and complexity
for model in model_names:
    model_accuracy_cols = [col for col in accuracy_cols if model.lower() in col.lower()]
    if model_accuracy_cols:
        model_data = df[df['Complexity'].isin([1,2,3])].groupby('Complexity')[model_accuracy_cols].mean().mean(axis=1).reset_index()
        model_data.columns = ['Complexity', 'Accuracy']
        model_data['Accuracy'] = model_data['Accuracy'] * 100  # Convert to percentage
        model_data['Model'] = model
        accuracy_data.append(model_data)

# Combine all model data for accuracy
if accuracy_data:
    acc_plot_data = pd.concat(accuracy_data)
    
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


#plt.show()

# ============ SUMMARY STATISTICS TABLE ============
print("\n" + "="*80)
print("PERFORMANCE SUMMARY STATISTICS")
print("="*80)

# Equivalence summary
print("\n SCRIPT EQUIVALENCE ANALYSIS (%)")
print("-" * 60)
equiv_summary = equiv_plot_data.pivot(index='Model', columns='Complexity', 
                                     values=[col for col in equiv_plot_data.columns if '_equivalence_percentage' in col][0])
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

# Get columns that match the pattern for each model
script_columns = []

# Define the pattern to match: model_name_script_number
for model in ['qwen', 'deepseek', 'grok-3', 'chatgpt-4.5']:
    matching_columns = [col for col in df.columns 
                       if col.startswith(f'{model}_script_') and 
                       col.replace(f'{model}_script_', '').isdigit()]
    script_columns.extend(matching_columns)

# # Print the matching columns
# print("Found script columns:")
# for col in sorted(script_columns):
#     print(col)

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set scientific styling
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.family': 'serif', 'font.size': 10})

def get_model_display_name(model_key):
    """Canonical display label for a dataset column.

    Backed by superarc.registry so there is one definition instead of the two
    that previously disagreed: the copy used for Figure 5 mapped
    `gemini-2.5-pro` to "Gemini-2.5-Pro", while the copy used for Figure 6
    mapped `gemini` to it, so the same label denoted different data in the two
    published figures.

    Unregistered keys are returned unchanged, which is what the downstream
    `valid_models` filter uses to exclude declared-retired columns.
    """
    model = _by_column(_B_SCRIPT, model_key)
    return model.name if model is not None else model_key

# Define ordered model list
ordered_models = ['chatgpt-4o', 'gpt-4o-mini', 'gemini-2.5-pro', 'gemini', 'gemini-thinking',
                  'claude-3-5-sonnet', 'chatgpt-o1', 'mistral', 'meta', 'cursor_small',
                  'o1_mini', 'grok', 'grok-3', 'qwen', 'deepseek', 'chatgpt-4.5',
                  'claude-3.7', 'deepseek_r1_0525', 'llama_4_scout', 'qwen3',
                  'chatgpt_5', 'opus_4', 'mistral_large2405', 'claude_sonnet_4',
                  'grok_4.1', 'gpt-5.2', 'claude-4.5', 'gemini-3-pro', 'mistral-large-3',
                  ]

# Filter to valid models only
valid_models = [model for model in ordered_models if get_model_display_name(model) != model]

# Remove duplicate columns
df = df.loc[:, ~df.columns.duplicated()]

# ============ UNIFIED DATA PROCESSING ============
def process_model_data():
    """Process all three types of data in one unified function"""
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
    
    return (pd.DataFrame(script_counts), 
            pd.DataFrame(total_classifications), 
            pd.DataFrame(accurate_classifications))

# Process all data
script_data, total_class_data, accurate_class_data = process_model_data()

# Fix for ChatGPT-4o-Mini at complexity 3
script_data.loc[(script_data['Model'] == 'gpt-4o-mini') & 
                (script_data['Complexity'] == 3), 'Count'] = 1

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
    line = ax2.plot(x_positions, volume_counts, 
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



#plt.show()

# ============ COMPREHENSIVE STATISTICAL SUMMARY ============
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
