import pandas as pd
import numpy as np
import ast
import sys
import io
import re
import matplotlib.pyplot as plt
import os
import math

# --- Constants & Helpers ---

NAME_MAPPING = {
    'gpt_4o': 'ChatGPT-4o', 
    'gpt_4o_mini': 'ChatGPT-4o-Mini', 
    'gemini_2.5_pro': 'Gemini-2.5-Pro', 
    'gemini': 'Gemini', 
    'claude_3.5': 'Claude-3.5', 
    'o1_preview': 'o1-Preview', 
    'mistral': 'Mistral', 
    'meta': 'Meta', 
    'cursor_small': 'Cursor-Small', 
    'o1_mini': 'o1-Mini', 
    'grok_3': 'Grok-3', 
    'grok4': 'Grok-4', 
    'qwen': 'Qwen', 
    'deepseek': 'DeepSeek', 
    'chatgpt_4.5': 'ChatGPT-4.5', 
    'claude_3.7': 'Claude-3.7', 
    'deepseek_r1_0528': 'DeepSeek-R1-0525', 
    'llama_4_scout': 'Llama-4-Scout', 
    'qwen3': 'Qwen-3', 
    'chatgpt_5': 'ChatGPT-5', 
    'opus_4': 'Claude-Opus-4', 
    'mistral_large2405': 'Mistral-Large-2405', 
    'claude_sonnet_4': 'Claude-Sonnet-4', 
    'grok_4.1': 'Grok-4.1', 
    'gpt-5.2': 'ChatGPT-5.2', 
    'claude-4.5': 'Claude-4.5', 
    'gemini-3-pro': 'Gemini-3-Pro', 
    'mistral-large-3': 'Mistral-Large-3',
    # Additional mappings for consistency
    'grok-3': 'Grok-3',
    'claude-3-5-sonnet': 'Claude-3.5-Sonnet',
    'chatgpt-4o': 'ChatGPT-4o',
    'chatgpt-o1': 'ChatGPT-o1',
    'grok': 'Grok',
    'gemini-thinking': 'Gemini-Thinking',
    'deepseek_r1_0525': 'DeepSeek-R1-0525',
    'gpt-4o-mini': 'ChatGPT-4o-Mini',
    'gemini-2.5-pro': 'Gemini-2.5-Pro',
}

def get_display_name(name):
    return NAME_MAPPING.get(name, name)

# --- Metric Calculation Helpers ---

def normalize_sequence(seq_str):
    if not isinstance(seq_str, str):
        return ""
    # Remove brackets, quotes, extra spaces
    s = re.sub(r'[\[\]"\'\s]', '', seq_str)
    # Ensure commas are the separator
    return s

def is_valid_script(script_code):
    """
    Checks if a script is a valid instance.
    Criteria:
    - Not just simple prints
    - Not 'Not found' or '***'
    - Not empty
    """
    if not isinstance(script_code, str):
        return False
        
    code = script_code.strip()
    if not code:
        return False
        
    # Check for Not Found
    if re.search(r'(?i)not\s*found', code) or '***' in code:
        return False
        
    # Check for simple prints of lists/values
    # e.g. print([2, 4, 6]) or print("2, 4, 6")
    # This regex looks for print call wrapping a list or string literal
    if re.match(r'^print\s*\(\s*[\["\'].*[\]"\']\s*\)\s*$', code):
        return False
        
    return True

def is_valid_formula(formula_str, target_seq):
    """
    Checks if a formula is a valid instance.
    Criteria:
    - Not ordinal (1, 2, 3...)
    - Not print statements
    - Not a copy of the target sequence
    - Not 'Not found'
    """
    if not isinstance(formula_str, str):
        return False
        
    f = formula_str.strip()
    if not f:
        return False
        
    # Check for Not Found
    if re.search(r'(?i)not\s*found', f) or '***' in f:
        return False
        
    # Check for prints
    if f.startswith('print('):
        return False
        
    # Check if it matches target sequence (copy-paste)
    norm_f = normalize_sequence(f)
    norm_target = normalize_sequence(target_seq)
    if norm_f == norm_target:
        return False
        
    # Check for ordinal (simple counting 1, 2, 3...)
    # We can approximate this by checking if it matches the target sequence if the target was 1,2,3...
    # But "ordinal" usually refers to the content being "1, 2, 3, 4, ..." regardless of target.
    # If the formula string ITSELF looks like a sequence of numbers, it's likely invalid (just outputting numbers)
    # unless the task WAS to output numbers. But usually formula is "2n" or "n^2".
    # If the formula contains commas and numbers, it's likely a sequence, not a formula.
    if re.match(r'^[\d,\s]+$', f):
        return False
        
    return True

def classify_instance(instance_str, target_seq, is_script=True):
    """
    Classifies an instance into one of 4 types:
    1) Known sequence
    2) Not found
    3) Pure math
    4) Print
    """
    if not isinstance(instance_str, str):
        return 'Not found'
        
    code = instance_str.strip()
    if not code:
        return 'Not found'
        
    # 1. Check for Not Found
    if re.search(r'(?i)not\s*found', code) or '***' in code:
        return 'Not found'
        
    # 2. Check for Print
    if is_script:
        # Script: print([...]) or print("...")
        if re.match(r'^print\s*\(\s*[\["\'].*[\]"\']\s*\)\s*$', code):
            return 'Print'
    else:
        # Formula: print(...) or copy of target
        if code.startswith('print('):
            return 'Print'
        # Check copy of target
        norm_code = normalize_sequence(code)
        norm_target = normalize_sequence(target_seq)
        if norm_code == norm_target and norm_code:
            return 'Print'
            
    # 3. Check for Known Sequence
    # Keywords based on common sequence names
    keywords = ['fibonacci', 'prime', 'lucas', 'pell', 'triangular', 'factorial', 
                'arithmetic', 'geometric', 'progression', 'linear_sequence']
    code_lower = code.lower()
    if any(k in code_lower for k in keywords):
        return 'Known sequence'
        
    # 4. Pure Math (Default for others)
    # If it's a script/formula that didn't match above, we assume it's attempting logic/math
    return 'Pure math'

def capture_output(script):
    # Capture stdout from executing the script
    old_stdout = sys.stdout
    sys.stdout = mystdout = io.StringIO()
    try:
        # Handle "print" in the script
        exec(script, {'__builtins__': __builtins__}, {})
        output = mystdout.getvalue().strip()
    except Exception as e:
        output = "*error"
    finally:
        sys.stdout = old_stdout
    return output

def evaluate_script_list(script_list_str):
    try:
        scripts = ast.literal_eval(script_list_str)
    except:
        return []
    
    outputs = []
    for script in scripts:
        out = capture_output(script)
        # Normalize output (remove brackets if printed as list)
        norm_out = normalize_sequence(out)
        outputs.append(norm_out)
    return outputs

def calculate_row_metrics(row, evaluations, sequence):
    norm_seq = normalize_sequence(sequence)
    
    # Filter out errors and empty
    valid_evals = [e for e in evaluations if e != "*error" and e != "" and e != "*notfound"]
    
    if not valid_evals:
        return 0, 0 # Accuracy, Equivalence
    
    # Accuracy: At least one matches sequence (Pass@k)
    accuracy = 100 if any(e == norm_seq for e in valid_evals) else 0
    
    # Equivalence: All valid evaluations are identical
    if len(valid_evals) > 0:
        equivalence = 100 if len(set(valid_evals)) == 1 else 0
    else:
        equivalence = 0
        
    return accuracy, equivalence

# --- Data Processing Functions ---

def process_multi_script():
    print("Processing Multi-Script Experiment...")
    df = pd.read_csv('multi-python-script-time-series.csv')
    model_cols = [c for c in df.columns if c not in ['sequence', 'Complexity']]
    
    results = []
    
    for model in model_cols:
        accuracies = []
        equivalences = []
        valid_counts = []
        
        # Type Counts
        cnt_known = []
        cnt_not_found = []
        cnt_pure_math = []
        cnt_print = []
        
        complexities = []
        
        for idx, row in df.iterrows():
            evals = evaluate_script_list(row[model])
            acc, equiv = calculate_row_metrics(row, evals, row['sequence'])
            
            # Calculate Valid Instances and Classify
            try:
                scripts = ast.literal_eval(row[model])
            except:
                scripts = []
            
            v_count = sum(1 for s in scripts if is_valid_script(s))
            
            # Classification
            c_known = 0
            c_not_found = 0
            c_pure = 0
            c_print = 0
            
            for s in scripts:
                cat = classify_instance(s, row['sequence'], is_script=True)
                if cat == 'Known sequence': c_known += 1
                elif cat == 'Not found': c_not_found += 1
                elif cat == 'Pure math': c_pure += 1
                elif cat == 'Print': c_print += 1
            
            accuracies.append(acc)
            equivalences.append(equiv)
            valid_counts.append(v_count)
            
            cnt_known.append(c_known)
            cnt_not_found.append(c_not_found)
            cnt_pure_math.append(c_pure)
            cnt_print.append(c_print)
            
            complexities.append(row['Complexity'])
            
        # Create temp df to group by complexity
        temp_df = pd.DataFrame({
            'complexity': complexities,
            'accuracy': accuracies,
            'equivalence': equivalences,
            'valid_count': valid_counts,
            'Known sequence': cnt_known,
            'Not found': cnt_not_found,
            'Pure math': cnt_pure_math,
            'Print': cnt_print
        })
        
        grouped_mean = temp_df.groupby('complexity').mean()
        grouped_sum = temp_df.groupby('complexity').sum()
        
        for comp in [1, 2, 3]:
            if comp in grouped_mean.index:
                # Standard Metrics
                results.append({
                    'case': 'multiple script',
                    'Measure': 'Accuracy',
                    'complexity': comp,
                    'model': model,
                    'value': grouped_mean.loc[comp, 'accuracy']
                })
                results.append({
                    'case': 'multiple script',
                    'Measure': 'Equivalence',
                    'complexity': comp,
                    'model': model,
                    'value': grouped_mean.loc[comp, 'equivalence']
                })
                results.append({
                    'case': 'multiple script',
                    'Measure': 'Valid Instances',
                    'complexity': comp,
                    'model': model,
                    'value': grouped_sum.loc[comp, 'valid_count']
                })
                
                # Type Metrics
                results.append({'case': 'multiple script', 'Measure': 'Known sequence', 'complexity': comp, 'model': model, 'value': grouped_sum.loc[comp, 'Known sequence']})
                results.append({'case': 'multiple script', 'Measure': 'Not found', 'complexity': comp, 'model': model, 'value': grouped_sum.loc[comp, 'Not found']})
                results.append({'case': 'multiple script', 'Measure': 'Pure math', 'complexity': comp, 'model': model, 'value': grouped_sum.loc[comp, 'Pure math']})
                results.append({'case': 'multiple script', 'Measure': 'Print', 'complexity': comp, 'model': model, 'value': grouped_sum.loc[comp, 'Print']})
                
    return pd.DataFrame(results)

def process_multi_formula():
    print("Processing Multi-Formula Experiment...")
    try:
        df = pd.read_csv('multi-formula-time-series.csv', encoding='utf-8')
    except UnicodeDecodeError:
        print("UTF-8 failed, trying latin1...")
        df = pd.read_csv('multi-formula-time-series.csv', encoding='latin1')
    except:
        print("Fallback to engine='python' with replace...")
        df = pd.read_csv('multi-formula-time-series.csv', engine='python', encoding_errors='replace')
    
    # Identify models by looking for _1, _2 suffixes
    eval_cols = [c for c in df.columns if c.endswith('_eval')]
    
    # Extract model base names (e.g. chatgpt_4.5 from chatgpt_4.5_1_eval)
    # We want to preserve the order of appearance in the file columns
    models = []
    seen = set()
    for c in df.columns:
        if c.endswith('_eval'):
            match = re.match(r'(.+)_\d+_eval$', c)
            if match:
                m = match.group(1)
                if m not in seen:
                    models.append(m)
                    seen.add(m)
            
    results = []
    
    for model in models:
        # Find all eval columns for this model
        model_eval_cols = sorted([c for c in eval_cols if c.startswith(model + '_') and re.match(rf'^{re.escape(model)}_\d+_eval$', c)])
        
        # Find corresponding raw formula columns
        # They should be just removing '_eval' suffix
        model_raw_cols = [c.replace('_eval', '') for c in model_eval_cols]
        
        if not model_eval_cols:
            continue
            
        accuracies = []
        equivalences = []
        valid_counts = []
        
        # Type Counts
        cnt_known = []
        cnt_not_found = []
        cnt_pure_math = []
        cnt_print = []
        
        complexities = []
        
        for idx, row in df.iterrows():
            evals = []
            v_count = 0
            
            c_known = 0
            c_not_found = 0
            c_pure = 0
            c_print = 0
            
            # Process Evals
            for col in model_eval_cols:
                val = str(row[col])
                if val == '*not found' or val == 'nan':
                    continue
                evals.append(normalize_sequence(val))
            
            # Process Raw Formulae for Validity and Classification
            for col in model_raw_cols:
                if col not in df.columns:
                    continue
                raw_val = str(row[col])
                if is_valid_formula(raw_val, row['sequence']):
                    v_count += 1
                    
                cat = classify_instance(raw_val, row['sequence'], is_script=False)
                if cat == 'Known sequence': c_known += 1
                elif cat == 'Not found': c_not_found += 1
                elif cat == 'Pure math': c_pure += 1
                elif cat == 'Print': c_print += 1
            
            acc, equiv = calculate_row_metrics(row, evals, row['sequence'])
            
            accuracies.append(acc)
            equivalences.append(equiv)
            valid_counts.append(v_count)
            
            cnt_known.append(c_known)
            cnt_not_found.append(c_not_found)
            cnt_pure_math.append(c_pure)
            cnt_print.append(c_print)
            
            complexities.append(row['Complexity'])
            
        temp_df = pd.DataFrame({
            'complexity': complexities,
            'accuracy': accuracies,
            'equivalence': equivalences,
            'valid_count': valid_counts,
            'Known sequence': cnt_known,
            'Not found': cnt_not_found,
            'Pure math': cnt_pure_math,
            'Print': cnt_print
        })
        
        grouped_mean = temp_df.groupby('complexity').mean()
        grouped_sum = temp_df.groupby('complexity').sum()
        
        for comp in [1, 2, 3]:
            if comp in grouped_mean.index:
                # Standard Metrics
                results.append({
                    'case': 'multiple formulae',
                    'Measure': 'Accuracy',
                    'complexity': comp,
                    'model': model,
                    'value': grouped_mean.loc[comp, 'accuracy']
                })
                results.append({
                    'case': 'multiple formulae',
                    'Measure': 'Equivalence',
                    'complexity': comp,
                    'model': model,
                    'value': grouped_mean.loc[comp, 'equivalence']
                })
                results.append({
                    'case': 'multiple formulae',
                    'Measure': 'Valid Instances',
                    'complexity': comp,
                    'model': model,
                    'value': grouped_sum.loc[comp, 'valid_count']
                })
                
                # Type Metrics
                results.append({'case': 'multiple formulae', 'Measure': 'Known sequence', 'complexity': comp, 'model': model, 'value': grouped_sum.loc[comp, 'Known sequence']})
                results.append({'case': 'multiple formulae', 'Measure': 'Not found', 'complexity': comp, 'model': model, 'value': grouped_sum.loc[comp, 'Not found']})
                results.append({'case': 'multiple formulae', 'Measure': 'Pure math', 'complexity': comp, 'model': model, 'value': grouped_sum.loc[comp, 'Pure math']})
                results.append({'case': 'multiple formulae', 'Measure': 'Print', 'complexity': comp, 'model': model, 'value': grouped_sum.loc[comp, 'Print']})
                
    return pd.DataFrame(results)

# --- Model Sorting Helpers ---

def get_family(name):
    n = name.lower()
    if 'cursor' in n: return 'Cursor' # Will be filtered later if needed, but user wants to remove it.
    # Actually user said "remove cursor family", so we can just return 'Cursor' and filter it out in generate_plots/main.
    
    if 'grok' in n: return 'Grok'
    if 'gpt' in n or 'o1' in n or 'chatgpt' in n: return 'OpenAI'
    if 'claude' in n or 'opus' in n: return 'Claude'
    if 'gemini' in n: return 'Gemini'
    if 'mistral' in n: return 'Mistral'
    if 'qwen' in n: return 'Qwen' # Covers Qwen and Qwen 3
    if 'deepseek' in n: return 'Deepseek'
    if 'meta' in n or 'llama' in n: return 'Meta' # Covers meta and llama_4_scout
    
    return 'Other'

def extract_version(name):
    n = name.lower()
    val = 0.0
    
    # Specific Overrides for Base Versions
    if 'o1' in n:
        val = 6.0
    elif 'gpt-5' in n or 'chatgpt_5' in n or 'gpt-5' in n:
        val = 5.0
    elif '4.5' in n:
        val = 4.5
    elif '4o' in n:
        val = 4.4
    elif '4' in n and ('gpt' in n or 'chatgpt' in n):
        val = 4.0
    elif '3.5' in n or '3-5' in n:
        val = 3.5
    elif 'grok' in n:
        if '4.1' in n: val = 4.1
        elif '4' in n: val = 4.0
        elif '3' in n: val = 3.0
        else: val = 1.0
    elif 'claude' in n or 'opus' in n:
        if '4.5' in n: val = 4.5
        elif 'opus' in n and '4' in n: val = 4.1 # Opus 4?
        elif 'sonnet' in n and '4' in n: val = 4.05 # Sonnet 4?
        elif '3.7' in n: val = 3.7
        elif '3.5' in n or '3-5' in n: val = 3.5
        elif 'opus' in n: val = 3.0 # Claude 3 Opus
    elif 'gemini' in n:
        if '3' in n: val = 3.0
        elif '2.5' in n or '2_5' in n: val = 2.5
        elif 'thinking' in n: val = 1.5
        else: val = 1.0
    elif 'mistral' in n:
        if 'large-3' in n: val = 3.0
        elif '2405' in n: val = 2.0
        else: val = 1.0
    elif 'qwen' in n:
        if '3' in n: val = 3.0
        else: val = 1.0
    elif 'deepseek' in n:
        if 'r1' in n: val = 3.0
        else: val = 2.0
    elif 'llama' in n:
        if '4' in n: val = 4.0
        else: val = 3.0
    else:
        # Fallback regex
        match = re.search(r'(\d+)[._-](\d+)', n)
        if match:
            val = float(f"{match.group(1)}.{match.group(2)}")
        else:
            match = re.search(r'(\d+)', n)
            if match:
                val = float(match.group(1))
            
    # Modifiers
    if 'mini' in n: val += 0.05
    if 'preview' in n: val -= 0.02
    if 'thinking' in n: val += 0.01
    
    # Deepseek dates
    if '0525' in n: val += 0.01
    if '0528' in n: val += 0.02
    
    return val

def get_sorted_families(models_list):
    families = {}
    for m in models_list:
        fam = get_family(m)
        if fam not in families:
            families[fam] = []
        families[fam].append(m)
    
    for fam in families:
        # Sort by version then name
        families[fam].sort(key=lambda x: (extract_version(x), x))
        
    return families

# --- Plotting Functions ---

def generate_plots(pivot_df, models_list):
    # Setup directories
    os.makedirs('plots/multi_formula', exist_ok=True)
    os.makedirs('plots/multi_script', exist_ok=True)
    
    # Map case names to directory
    case_map = {
        'multiple formulae': 'plots/multi_formula',
        'multiple script': 'plots/multi_script'
    }
    
    measures = ['Accuracy', 'Equivalence', 'Valid Instances']
    type_measures = ['Known sequence', 'Not found', 'Pure math', 'Print']
    
    # Process each case
    for case_name, output_dir in case_map.items():
        print(f"Generating plots for {case_name}...")
        
        # Filter data for this case
        df_case = pivot_df[pivot_df['case'] == case_name]
        
        if df_case.empty:
            print(f"No data for {case_name}")
            continue
            
        # Get models present in this case
        case_models = [m for m in models_list if m in df_case.columns]
        
        # Filter out models with no data AND exclude Cursor family
        valid_models = []
        for m in case_models:
            # Check if family is Cursor
            if get_family(m) == 'Cursor':
                continue
                
            if df_case[m].sum() > 0 or df_case[m].max() > 0:
                valid_models.append(m)
        
        if not valid_models:
            continue

        # Group and Sort by Family
        families = get_sorted_families(valid_models)
        
        # --- 1. Individual Family Plots (Standard Metrics) ---
        for fam, members in families.items():
            if not members:
                continue
            
            # Standard Metrics
            for measure in measures:
                plt.figure(figsize=(10, 6))
                plt.rcParams.update({'font.size': 14})
                
                # Get data for this measure
                m_data = df_case[df_case['Measure'] == measure].sort_values('complexity')
                x = m_data['complexity']
                
                # Plot each member
                for m in members:
                    y = m_data[m]
                    label = get_display_name(m)
                    plt.plot(x, y, 'o-', label=label, linewidth=2, markersize=8)
                
                plt.xlabel('Complexity')
                
                if measure == 'Valid Instances':
                    plt.ylabel('Count')
                else:
                    plt.ylabel(f'{measure} (%)')
                    
                plt.title(f'{fam}: {measure} Comparison')
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.grid(True, alpha=0.3)
                plt.xticks([1, 2, 3])
                
                if measure != 'Valid Instances':
                    plt.yticks(np.arange(0, 105, 20))
                    plt.ylim(-5, 105)
                
                plt.tight_layout()
                
                safe_fam = re.sub(r'[^\w\-]', '_', fam)
                safe_measure = measure.lower().replace(' ', '_')
                plt.savefig(f"{output_dir}/{safe_measure}_{safe_fam}_combined.png")
                plt.close()
            
            # --- Type Analysis Plot (2x2 Grid per Family) ---
            # "This plot must be made by family, by complexity. One subplot by type"
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f'{fam}: Instance Type Analysis', fontsize=20)
            
            axes_flat = axes.flatten()
            
            for idx, t_measure in enumerate(type_measures):
                ax = axes_flat[idx]
                
                # Get data for this measure
                t_data = df_case[df_case['Measure'] == t_measure].sort_values('complexity')
                
                if t_data.empty:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
                    continue
                    
                x = t_data['complexity']
                
                for m in members:
                    y = t_data[m]
                    label = get_display_name(m)
                    ax.plot(x, y, 'o-', label=label, linewidth=2)
                    
                ax.set_title(t_measure, fontsize=16)
                ax.set_xlabel('Complexity')
                ax.set_ylabel('Count')
                ax.set_xticks([1, 2, 3])
                ax.grid(True, alpha=0.3)
                if idx == 0: # Only legend in first subplot to avoid clutter? Or outside?
                    ax.legend(fontsize=8, loc='best')
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            safe_fam = re.sub(r'[^\w\-]', '_', fam)
            plt.savefig(f"{output_dir}/types_{safe_fam}_grid.png")
            plt.close()

        # --- 2. Unified Grid Plot (All Families) ---
        # Existing Grid Plot logic for standard measures
        for measure in measures:
            # Only include families that have members in this case
            active_families = {k: v for k, v in families.items() if v}
            n_families = len(active_families)
            
            if n_families == 0:
                continue
                
            cols = 3
            rows = math.ceil(n_families / cols)
            
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))
            fig.suptitle(f'{measure} - All Families ({case_name.title()})', fontsize=20)
            
            # Ensure axes is iterable even if 1x1
            if n_families == 1:
                axes_flat = [axes]
            else:
                axes_flat = axes.flatten()
            
            fam_names = sorted(active_families.keys())
            
            for idx, fam in enumerate(fam_names):
                members = active_families[fam]
                ax = axes_flat[idx]
                
                m_data = df_case[df_case['Measure'] == measure].sort_values('complexity')
                x = m_data['complexity']
                
                for m in members:
                    y = m_data[m]
                    label = get_display_name(m)
                    ax.plot(x, y, 'o-', label=label, linewidth=2)
                
                ax.set_title(fam, fontsize=16)
                ax.set_xlabel('Complexity')
                
                if measure == 'Valid Instances':
                    ax.set_ylabel('Count')
                else:
                    ax.set_ylabel(f'{measure} (%)')
                    ax.set_yticks(np.arange(0, 105, 20))
                    ax.set_ylim(-5, 105)
                    
                ax.set_xticks([1, 2, 3])
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8, loc='best')
            
            # Hide empty subplots
            for idx in range(n_families, len(axes_flat)):
                axes_flat[idx].axis('off')
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            safe_measure = measure.lower().replace(' ', '_')
            plt.savefig(f"{output_dir}/ALL_FAMILIES_{safe_measure}_grid.png")
            plt.close()

        # --- 3. Grand Type Summary Plot (4 rows x max 4 families columns) ---
        if families:
            fam_names = sorted(families.keys())
            
            # Split into chunks of 4
            chunk_size = 4
            fam_chunks = [fam_names[i:i + chunk_size] for i in range(0, len(fam_names), chunk_size)]
            
            for chunk_idx, current_fams in enumerate(fam_chunks):
                n_cols = len(current_fams)
                n_rows = len(type_measures)
                
                # Dynamic size: Width based on num families, Height fixed-ish
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3.5), sharex=True, sharey='row')
                part_num = chunk_idx + 1
                fig.suptitle(f'Instance Type Distribution - Part {part_num} ({case_name.title()})', fontsize=24)
                
                # Handle axes shape consistency
                # If 1 col, axes is 1D array of shape (rows,). Reshape to (rows, 1)
                if n_cols == 1:
                    axes = axes[:, np.newaxis]
                # If 1 row (unlikely here), axes is (cols,). Reshape to (1, cols)
                if n_rows == 1:
                    axes = axes[np.newaxis, :]
                
                for col_idx, fam in enumerate(current_fams):
                    members = families[fam]
                    
                    for row_idx, t_measure in enumerate(type_measures):
                        ax = axes[row_idx, col_idx]
                        
                        # Data
                        t_data = df_case[df_case['Measure'] == t_measure].sort_values('complexity')
                        x = t_data['complexity']
                        
                        # Plot lines
                        for m in members:
                            y = t_data[m]
                            label = get_display_name(m)
                            ax.plot(x, y, 'o-', label=label, linewidth=2, alpha=0.8)
                        
                        # Styling
                        ax.grid(True, alpha=0.3)
                        
                        # Column Titles (Family Name)
                        if row_idx == 0:
                            ax.set_title(fam, fontsize=16, fontweight='bold')
                            
                        # Row Labels (Type Name) - Leftmost column only
                        if col_idx == 0:
                            ax.set_ylabel(t_measure, fontsize=14, fontweight='bold')
                        
                        # X-axis
                        if row_idx == n_rows - 1:
                            ax.set_xlabel('Complexity')
                            ax.set_xticks([1, 2, 3])
                        
                        # Legend per subplot (since models differ per family)
                        ax.legend(fontsize=8, loc='best')

                plt.tight_layout(rect=[0, 0.03, 1, 0.96])
                plt.savefig(f"{output_dir}/GRAND_TYPES_GRID_PART_{part_num}.png")
                plt.close()
            
        print(f"Saved plots to {output_dir}")

# --- Main Execution ---

def main():
    # 1. Calculate Metrics
    res_script = process_multi_script()
    res_formula = process_multi_formula()
    
    # 2. Identify Models Order
    all_models = []
    seen = set()
    
    def add_models(df_res):
        if df_res.empty: return
        # Extract unique models preserving order
        m_list = df_res['model'].drop_duplicates().tolist()
        for m in m_list:
            if m not in seen:
                all_models.append(m)
                seen.add(m)
                
    add_models(res_script)
    add_models(res_formula)
    
    # 3. Combine and Pivot
    combined = pd.concat([res_script, res_formula], ignore_index=True)
    
    pivot_df = combined.pivot_table(
        index=['case', 'Measure', 'complexity'], 
        columns='model', 
        values='value'
    )
    
    # Reindex columns to match observed order
    pivot_df = pivot_df.reindex(columns=all_models)
    pivot_df = pivot_df.reset_index()
    pivot_df = pivot_df.fillna(0)
    
    # 4. Save Summary CSV
    output_file = 'comparison_models_summary.csv'
    pivot_df.to_csv(output_file, index=False)
    print(f"Summary saved to {output_file}")
    
    # 5. Generate Plots
    # We need to know which models belong to which case to order them correctly per case
    # The 'all_models' list has them mixed (script then formula).
    # But for plotting, we want the order within the specific case.
    
    # Let's extract per-case model lists again for plotting
    script_models = res_script['model'].drop_duplicates().tolist()
    formula_models = res_formula['model'].drop_duplicates().tolist()
    
    # Generate plots for Script models
    generate_plots(pivot_df[pivot_df['case'] == 'multiple script'], script_models)
    
    # Generate plots for Formula models
    generate_plots(pivot_df[pivot_df['case'] == 'multiple formulae'], formula_models)

if __name__ == "__main__":
    main()
