# Simple Model Progression Analysis for Jupyter Notebook
# No external dependencies required (only pandas, numpy, matplotlib)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define the old ranking data
old_data = {
    'Model': ['ASI', 'chatgpt_4.5', 'o1_mini', 'claude_3.7', 'claude_3.5', 'o1_preview', 
              'gpt_4o_mini', 'cursor_small', 'gemini', 'mistral', 'qwen', 'deepseek', 
              'grok_3', 'gpt_4o', 'meta'],
    'p1': [1.00, 0.00, 0.00, 0.00, 0.06, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    'p2': [0.00, 1.00, 0.64, 0.81, 0.14, 0.29, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.02, 0.00, 0.00],
    'p3': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    'p4': [0.00, 0.00, 0.36, 0.19, 0.80, 0.71, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.98, 1.00, 1.00],
    'r1': [1.000, 0.000, 0.000, 0.000, 0.449, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
    'r2': [0.000, 0.419, 0.537, 0.407, 0.428, 0.423, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.318, 0.000, 0.000],
    'r3': [1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.762, 0.762, 0.762, 0.710, 0.710, 0.710, 0.000, 0.000, 0.000],
    'tst': [1.000, 0.042, 0.034, 0.033, 0.033, 0.012, 0.008, 0.008, 0.008, 0.007, 0.007, 0.007, 0.001, 0.000, 0.000]
}

# Define the new ranking data
new_data = {
    'Model': ['ASI', 'chatgpt_4.5', 'o1_mini', 'claude_3.7', 'claude_3.5', 'o1_preview', 
              'gpt_4o_mini', 'cursor_small', 'gemini', 'deepseek', 'qwen', 'mistral', 
              'llama_4_scout', 'grok_3', 'meta', 'opus_4', 'claude_sonnet_4', 'gemini_2.5_pro', 
              'mistral_large2405', 'qwen3', 'deepseek_r1_0528', 'grok4', 'gpt_4o'],
    'p1': [1.00, 0.00, 0.00, 0.00, 0.06, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 
           0.01, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    'p2': [0.00, 1.00, 0.64, 0.81, 0.14, 0.29, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 
           0.00, 0.02, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    'p3': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'p4': [0.00, 0.00, 0.36, 0.19, 0.80, 0.71, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 
           0.99, 0.98, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    'r1': [1.000, 0.000, 0.000, 0.000, 0.449, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 
           0.450, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
    'r2': [0.000, 0.419, 0.537, 0.407, 0.428, 0.423, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 
           0.000, 0.318, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
    'r3': [1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.762, 0.762, 0.762, 0.710, 0.710, 0.710, 
           0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
    'tst': [1.000, 0.042, 0.034, 0.033, 0.033, 0.012, 0.008, 0.008, 0.008, 0.007, 0.007, 0.007, 
            0.004, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
}

# Create DataFrames
df_old = pd.DataFrame(old_data).set_index('Model')
df_new = pd.DataFrame(new_data).set_index('Model')

# Define model families and their evolution
model_families = {
    'Claude': {'old': ['claude_3.5', 'claude_3.7'], 'new': ['opus_4', 'claude_sonnet_4']},
    'Gemini': {'old': ['gemini'], 'new': ['gemini_2.5_pro']},
    'Mistral': {'old': ['mistral'], 'new': ['mistral_large2405']},
    'Qwen': {'old': ['qwen'], 'new': ['qwen3']},
    'Deepseek': {'old': ['deepseek'], 'new': ['deepseek_r1_0528']},
    'Grok': {'old': ['grok_3'], 'new': ['grok4']},
    'LLaMA': {'old': [], 'new': ['llama_4_scout']}
}

def calculate_distance_to_asi(df, model_name, tst_weight=5):
    """Calculate weighted distance to ASI (TST score weighted more heavily)"""
    # ASI perfect scores
    asi_scores = {'p1': 1.0, 'p2': 0.0, 'p3': 0.0, 'p4': 0.0, 'r1': 1.0, 'r2': 0.0, 'r3': 1.0, 'tst': 1.0}
    
    if model_name not in df.index:
        return float('inf')
    
    model_row = df.loc[model_name]
    
    # Calculate weighted distance (TST gets higher weight)
    total_distance = 0
    total_weight = 0
    
    for metric in ['p1', 'p2', 'p3', 'p4', 'r1', 'r2', 'r3', 'tst']:
        weight = tst_weight if metric == 'tst' else 1
        diff = abs(asi_scores[metric] - model_row[metric])
        total_distance += (diff ** 2) * weight
        total_weight += weight
    
    return np.sqrt(total_distance / total_weight)

def analyze_progression():
    """Analyze model progression and return results"""
    results = []
    
    print("=== MODEL PROGRESSION TO ASI ANALYSIS ===")
    print("(Lower distance = closer to ASI = better performance)\n")
    
    for family_name, family_models in model_families.items():
        print(f"--- {family_name} Family ---")
        
        # Old models
        old_distances = []
        for model in family_models['old']:
            if model in df_old.index:
                dist = calculate_distance_to_asi(df_old, model)
                old_distances.append(dist)
                print(f"  {model} (old): {dist:.4f}")
        
        # New models
        new_distances = []
        for model in family_models['new']:
            if model in df_new.index:
                dist = calculate_distance_to_asi(df_new, model)
                new_distances.append(dist)
                print(f"  {model} (new): {dist:.4f}")
        
        # Calculate change
        if old_distances and new_distances:
            avg_old = np.mean(old_distances)
            avg_new = np.mean(new_distances)
            improvement = avg_old - avg_new
            improvement_pct = (improvement / avg_old) * 100 if avg_old > 0 else 0
            
            status = "IMPROVED ✅" if improvement > 0 else "REGRESSED ❌"
            print(f"  Change: {improvement:+.4f} ({improvement_pct:+.1f}%) - {status}")
            
            results.append({
                'Family': family_name,
                'Old_Avg_Distance': avg_old,
                'New_Avg_Distance': avg_new,
                'Improvement': improvement,
                'Improvement_Pct': improvement_pct,
                'Status': 'Improved' if improvement > 0 else 'Regressed'
            })
        elif new_distances:
            avg_new = np.mean(new_distances)
            print(f"  New family distance: {avg_new:.4f}")
            
            results.append({
                'Family': family_name,
                'Old_Avg_Distance': np.nan,
                'New_Avg_Distance': avg_new,
                'Improvement': np.nan,
                'Improvement_Pct': np.nan,
                'Status': 'New'
            })
        
        print()
    
    return pd.DataFrame(results)

def create_simple_plots(results_df):
    """Create simple visualization plots"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Distance comparison
    ax1 = axes[0, 0]
    comparison_df = results_df.dropna(subset=['Old_Avg_Distance'])
    if not comparison_df.empty:
        x = np.arange(len(comparison_df))
        width = 0.35
        
        ax1.bar(x - width/2, comparison_df['Old_Avg_Distance'], width, 
               label='Old Models', alpha=0.7, color='lightcoral')
        ax1.bar(x + width/2, comparison_df['New_Avg_Distance'], width, 
               label='New Models', alpha=0.7, color='lightblue')
        
        ax1.set_xlabel('Model Family')
        ax1.set_ylabel('Distance to ASI')
        ax1.set_title('Distance to ASI: Old vs New Models')
        ax1.set_xticks(x)
        ax1.set_xticklabels(comparison_df['Family'], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: Improvement percentage
    ax2 = axes[0, 1]
    improvement_df = results_df.dropna(subset=['Improvement_Pct'])
    if not improvement_df.empty:
        colors = ['green' if x > 0 else 'red' for x in improvement_df['Improvement_Pct']]
        ax2.bar(improvement_df['Family'], improvement_df['Improvement_Pct'], 
               color=colors, alpha=0.7)
        ax2.set_xlabel('Model Family')
        ax2.set_ylabel('Improvement (%)')
        ax2.set_title('Performance Change (Positive = Better)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: TST scores comparison
    ax3 = axes[1, 0]
    tst_old = [df_old.loc[m, 'tst'] for family in model_families.values() 
               for m in family['old'] if m in df_old.index]
    tst_new = [df_new.loc[m, 'tst'] for family in model_families.values() 
               for m in family['new'] if m in df_new.index]
    
    ax3.hist([tst_old, tst_new], bins=10, alpha=0.7, 
            label=['Old Models', 'New Models'], color=['lightcoral', 'lightblue'])
    ax3.set_xlabel('TST Score')
    ax3.set_ylabel('Number of Models')
    ax3.set_title('TST Score Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: All models ranking by TST
    ax4 = axes[1, 1]
    all_models_new = df_new[df_new.index != 'ASI'].sort_values('tst', ascending=True)
    colors = ['lightblue' if any(model in family['new'] for family in model_families.values()) 
             else 'lightcoral' if any(model in family['old'] for family in model_families.values()) 
             else 'lightgray' for model in all_models_new.index]
    
    y_pos = np.arange(len(all_models_new))
    ax4.barh(y_pos, all_models_new['tst'], color=colors, alpha=0.7)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(all_models_new.index, fontsize=8)
    ax4.set_xlabel('TST Score')
    ax4.set_title('All Models TST Ranking')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def print_summary(results_df):
    """Print analysis summary"""
    print("\n" + "="*50)
    print("SUMMARY RESULTS")
    print("="*50)
    
    print("\n📊 Results Table:")
    print(results_df.round(4).to_string(index=False))
    
    # Key insights
    improved = results_df[results_df['Status'] == 'Improved']
    regressed = results_df[results_df['Status'] == 'Regressed']
    new_families = results_df[results_df['Status'] == 'New']
    
    print(f"\n🎯 Key Insights:")
    print(f"  • {len(improved)} families improved")
    print(f"  • {len(regressed)} families regressed")
    print(f"  • {len(new_families)} new families")
    
    if not improved.empty:
        best_improvement = improved.loc[improved['Improvement_Pct'].idxmax()]
        print(f"  • Best improvement: {best_improvement['Family']} (+{best_improvement['Improvement_Pct']:.1f}%)")
    
    if not regressed.empty:
        worst_regression = regressed.loc[regressed['Improvement_Pct'].idxmin()]
        print(f"  • Worst regression: {worst_regression['Family']} ({worst_regression['Improvement_Pct']:.1f}%)")
    
    # Overall trend
    valid_improvements = results_df['Improvement_Pct'].dropna()
    if not valid_improvements.empty:
        avg_change = valid_improvements.mean()
        trend = "improving" if avg_change > 0 else "regressing"
        print(f"  • Overall trend: Models are {trend} (avg: {avg_change:+.1f}%)")
    
    print("\n💡 Interpretation:")
    print("  • Distance to ASI: Lower = Better (0 = perfect ASI performance)")
    print("  • TST score is weighted 5x more than other metrics")
    print("  • Positive improvement % = family got closer to ASI")
    print("  • Negative improvement % = family moved away from ASI")

# Main execution
if __name__ == "__main__":
    print("🚀 Analyzing Model Progression Towards ASI...\n")
    
    # Run analysis
    results = analyze_progression()
    
    # Print summary
    print_summary(results)
    
    # Create plots
    print("\n📊 Generating plots...")
    fig = create_simple_plots(results)
    
    print("\n✅ Analysis complete!")