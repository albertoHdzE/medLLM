# Model Progression Analysis - Notebook Cell
# This code can be copied into your Jupyter notebook

import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming you have df_ranking2 from your notebook, we'll extract the data
# If you need to recreate the data manually, use the dictionaries below

# Define model families and their evolution
model_families = {
    'Claude': {
        'old': ['claude_3.5', 'claude_3.7'],
        'new': ['opus_4', 'claude_sonnet_4']
    },
    'Gemini': {
        'old': ['gemini'],
        'new': ['gemini_2.5_pro']
    },
    'Mistral': {
        'old': ['mistral'],
        'new': ['mistral_large2405']
    },
    'Qwen': {
        'old': ['qwen'],
        'new': ['qwen3']
    },
    'Deepseek': {
        'old': ['deepseek'],
        'new': ['deepseek_r1_0528']
    },
    'Grok': {
        'old': ['grok_3'],
        'new': ['grok4']
    },
    'LLaMA': {
        'old': [],  # No old LLaMA in the dataset
        'new': ['llama_4_scout']
    }
}

def calculate_weighted_distance_to_asi(df, model_name, weights=None):
    """Calculate weighted distance to ASI, giving more importance to 'tst' score"""
    if weights is None:
        # Give higher weight to 'tst' as it's the most important metric
        weights = {'p1': 1, 'p2': 1, 'p3': 1, 'p4': 1, 'r1': 1, 'r2': 1, 'r3': 1, 'tst': 5}
    
    # Find ASI row (should have perfect scores)
    asi_scores = {'p1': 1.0, 'p2': 0.0, 'p3': 0.0, 'p4': 0.0, 'r1': 1.0, 'r2': 0.0, 'r3': 1.0, 'tst': 1.0}
    
    # Get model scores
    if model_name in df.index:
        model_row = df.loc[model_name]
    else:
        print(f"Warning: {model_name} not found in dataframe")
        return float('inf')
    
    weighted_distance = 0
    total_weight = 0
    
    for metric, weight in weights.items():
        if metric in model_row.index:
            diff = abs(asi_scores[metric] - model_row[metric])
            weighted_distance += (diff ** 2) * weight
            total_weight += weight
    
    return np.sqrt(weighted_distance / total_weight)

def analyze_model_progression_from_ranking(df_ranking):
    """Analyze model progression using the ranking dataframe from your notebook"""
    results = []
    
    print("=== MODEL PROGRESSION ANALYSIS ===")
    print("Distance to ASI (lower is better, 0 = perfect ASI performance)\n")
    
    for family_name, family_models in model_families.items():
        print(f"\n--- {family_name} Family ---")
        
        # Calculate distances for old models
        old_distances = []
        for old_model in family_models['old']:
            if old_model in df_ranking.index:
                dist = calculate_weighted_distance_to_asi(df_ranking, old_model)
                old_distances.append(dist)
                print(f"{old_model} (old): {dist:.4f}")
        
        # Calculate distances for new models
        new_distances = []
        for new_model in family_models['new']:
            if new_model in df_ranking.index:
                dist = calculate_weighted_distance_to_asi(df_ranking, new_model)
                new_distances.append(dist)
                print(f"{new_model} (new): {dist:.4f}")
        
        # Calculate improvement/regression
        if old_distances and new_distances:
            avg_old = np.mean(old_distances)
            avg_new = np.mean(new_distances)
            improvement = avg_old - avg_new  # Positive = improvement, Negative = regression
            improvement_pct = (improvement / avg_old) * 100 if avg_old > 0 else 0
            
            status = "IMPROVED ✅" if improvement > 0 else "REGRESSED ❌"
            print(f"Average change: {improvement:+.4f} ({improvement_pct:+.1f}%) - {status}")
            
            results.append({
                'Family': family_name,
                'Old_Distance': avg_old,
                'New_Distance': avg_new,
                'Improvement': improvement,
                'Improvement_Pct': improvement_pct,
                'Status': status
            })
        elif new_distances and not old_distances:
            # New family with no old models
            avg_new = np.mean(new_distances)
            print(f"New family - Distance to ASI: {avg_new:.4f}")
            
            results.append({
                'Family': family_name,
                'Old_Distance': np.nan,
                'New_Distance': avg_new,
                'Improvement': np.nan,
                'Improvement_Pct': np.nan,
                'Status': 'NEW'
            })
    
    return pd.DataFrame(results)

def create_progression_visualization(results_df, df_ranking):
    """Create visualizations of model progression"""
    plt.figure(figsize=(16, 12))
    
    # Plot 1: Distance comparison (only for families with both old and new)
    plt.subplot(2, 3, 1)
    comparison_df = results_df.dropna(subset=['Old_Distance'])
    if not comparison_df.empty:
        x = np.arange(len(comparison_df))
        width = 0.35
        
        plt.bar(x - width/2, comparison_df['Old_Distance'], width, label='Old Models', alpha=0.8, color='lightcoral')
        plt.bar(x + width/2, comparison_df['New_Distance'], width, label='New Models', alpha=0.8, color='lightblue')
        
        plt.xlabel('Model Family')
        plt.ylabel('Distance to ASI')
        plt.title('Distance to ASI: Old vs New Models')
        plt.xticks(x, comparison_df['Family'], rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Plot 2: Improvement percentage
    plt.subplot(2, 3, 2)
    improvement_df = results_df.dropna(subset=['Improvement_Pct'])
    if not improvement_df.empty:
        colors = ['green' if x > 0 else 'red' for x in improvement_df['Improvement_Pct']]
        plt.bar(improvement_df['Family'], improvement_df['Improvement_Pct'], color=colors, alpha=0.7)
        plt.xlabel('Model Family')
        plt.ylabel('Improvement (%)')
        plt.title('Improvement Percentage (Positive = Better)')
        plt.xticks(rotation=45)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.grid(True, alpha=0.3)
    
    # Plot 3: All models distance to ASI
    plt.subplot(2, 3, 3)
    all_distances = []
    all_models = []
    
    for model in df_ranking.index:
        if model != 'ASI':  # Exclude ASI itself
            dist = calculate_weighted_distance_to_asi(df_ranking, model)
            all_distances.append(dist)
            all_models.append(model)
    
    # Sort by distance
    sorted_data = sorted(zip(all_distances, all_models))
    sorted_distances, sorted_models = zip(*sorted_data)
    
    # Color code by family
    colors = []
    for model in sorted_models:
        found_family = False
        for family_name, family_models in model_families.items():
            if model in family_models['old'] + family_models['new']:
                if model in family_models['new']:
                    colors.append('lightblue')  # New models
                else:
                    colors.append('lightcoral')  # Old models
                found_family = True
                break
        if not found_family:
            colors.append('lightgray')  # Other models
    
    plt.barh(range(len(sorted_models)), sorted_distances, color=colors, alpha=0.7)
    plt.yticks(range(len(sorted_models)), sorted_models)
    plt.xlabel('Distance to ASI')
    plt.title('All Models Ranking (Lower = Better)')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: TST score comparison (most important metric)
    plt.subplot(2, 3, 4)
    tst_scores = df_ranking['tst'].sort_values(ascending=False)
    colors_tst = []
    for model in tst_scores.index:
        found_family = False
        for family_name, family_models in model_families.items():
            if model in family_models['old'] + family_models['new']:
                if model in family_models['new']:
                    colors_tst.append('lightblue')  # New models
                else:
                    colors_tst.append('lightcoral')  # Old models
                found_family = True
                break
        if not found_family:
            colors_tst.append('lightgray')  # Other models
    
    plt.barh(range(len(tst_scores)), tst_scores.values, color=colors_tst, alpha=0.7)
    plt.yticks(range(len(tst_scores)), tst_scores.index)
    plt.xlabel('TST Score')
    plt.title('TST Score Ranking (Higher = Better)')
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Family performance summary
    plt.subplot(2, 3, 5)
    family_best_scores = []
    family_names = []
    
    for family_name, family_models in model_families.items():
        all_family_models = family_models['old'] + family_models['new']
        family_tst_scores = []
        
        for model in all_family_models:
            if model in df_ranking.index:
                family_tst_scores.append(df_ranking.loc[model, 'tst'])
        
        if family_tst_scores:
            family_best_scores.append(max(family_tst_scores))
            family_names.append(family_name)
    
    plt.bar(family_names, family_best_scores, alpha=0.7, color='skyblue')
    plt.xlabel('Model Family')
    plt.ylabel('Best TST Score')
    plt.title('Best TST Score by Family')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Progress trajectory for families with both old and new
    plt.subplot(2, 3, 6)
    for i, row in comparison_df.iterrows():
        plt.plot([0, 1], [row['Old_Distance'], row['New_Distance']], 
                marker='o', linewidth=2, label=row['Family'], markersize=8)
    
    plt.xlabel('Generation')
    plt.ylabel('Distance to ASI')
    plt.title('Model Evolution Trajectory')
    plt.xticks([0, 1], ['Old', 'New'])
    if not comparison_df.empty:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return plt.gcf()

# Main execution code for notebook
print("🚀 Starting Model Progression Analysis...")
print("Make sure you have df_ranking2 available in your notebook!\n")

# Run the analysis (assuming df_ranking2 exists in your notebook)
try:
    # This will work if df_ranking2 is defined in your notebook
    progression_results = analyze_model_progression_from_ranking(df_ranking2)
    
    print("\n=== SUMMARY TABLE ===")
    print(progression_results.to_string(index=False, float_format='%.4f'))
    
    # Create visualizations
    fig = create_progression_visualization(progression_results, df_ranking2)
    
    # Calculate overall statistics
    print("\n=== OVERALL STATISTICS ===")
    
    # Best performing new models
    new_models_all = []
    for family_models in model_families.values():
        new_models_all.extend(family_models['new'])
    
    new_models_in_data = [m for m in new_models_all if m in df_ranking2.index]
    if new_models_in_data:
        best_new_model = max(new_models_in_data, key=lambda x: df_ranking2.loc[x, 'tst'])
        print(f"Best performing new model: {best_new_model} (TST: {df_ranking2.loc[best_new_model, 'tst']:.4f})")
    
    # Models that improved the most
    improved_families = progression_results[progression_results['Improvement'] > 0].sort_values('Improvement', ascending=False)
    if not improved_families.empty:
        print(f"Most improved family: {improved_families.iloc[0]['Family']} (+{improved_families.iloc[0]['Improvement_Pct']:.1f}%)")
    
    # Models that regressed the most
    regressed_families = progression_results[progression_results['Improvement'] < 0].sort_values('Improvement')
    if not regressed_families.empty:
        print(f"Most regressed family: {regressed_families.iloc[0]['Family']} ({regressed_families.iloc[0]['Improvement_Pct']:.1f}%)")
    
    print("\n✅ Analysis complete!")
    
except NameError:
    print("❌ Error: df_ranking2 not found!")
    print("Please make sure you have run the previous cells to create df_ranking2")
    print("Or manually create the ranking dataframe with the data you provided.")