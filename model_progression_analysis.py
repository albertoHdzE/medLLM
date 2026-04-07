import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import seaborn as sns

# Define the old and new ranking data
old_data = {
    'Model': ['gpt_4o', 'claude_3.5', 'gpt_4o_mini', 'cursor_small', 'gemini', 'meta', 
              'o1_mini', 'o1_preview', 'mistral', 'qwen', 'grok_3', 'deepseek', 'claude_3.7', 'chatgpt_4.5', 'ASI'],
    'p1': [0.00, 0.06, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00],
    'p2': [0.00, 0.14, 0.00, 0.00, 0.00, 0.00, 0.64, 0.29, 0.00, 0.00, 0.02, 0.00, 0.81, 1.00, 0.00],
    'p3': [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    'p4': [1.00, 0.80, 0.00, 0.00, 0.00, 1.00, 0.36, 0.71, 0.00, 0.00, 0.98, 0.00, 0.19, 0.00, 0.00],
    'r1': [0.000, 0.449, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000],
    'r2': [0.000, 0.428, 0.000, 0.000, 0.000, 0.000, 0.537, 0.423, 0.000, 0.000, 0.318, 0.000, 0.407, 0.419, 0.000],
    'r3': [0.000, 0.000, 0.762, 0.762, 0.762, 0.000, 0.000, 0.000, 0.710, 0.710, 0.000, 0.710, 0.000, 0.000, 1.000],
    'tst': [0.000, 0.033, 0.008, 0.008, 0.008, 0.000, 0.034, 0.012, 0.007, 0.007, 0.001, 0.007, 0.033, 0.042, 1.000]
}

new_data = {
    'Model': ['gpt_4o', 'claude_3.5', 'gpt_4o_mini', 'cursor_small', 'gemini', 'meta', 
              'o1_mini', 'o1_preview', 'mistral', 'qwen', 'grok_3', 'deepseek', 'claude_3.7', 'chatgpt_4.5',
              'opus_4', 'claude_sonnet_4', 'gemini_2.5_pro', 'mistral_large2405', 'qwen3', 'deepseek_r1_0528',
              'grok4', 'llama_4_scout', 'ASI'],
    'p1': [0.00, 0.06, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
           0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.01, 1.00],
    'p2': [0.00, 0.14, 0.00, 0.00, 0.00, 0.00, 0.64, 0.29, 0.00, 0.00, 0.02, 0.00, 0.81, 1.00,
           0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    'p3': [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'p4': [1.00, 0.80, 0.00, 0.00, 0.00, 1.00, 0.36, 0.71, 0.00, 0.00, 0.98, 0.00, 0.19, 0.00,
           1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 0.99, 0.00],
    'r1': [0.000, 0.449, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
           0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.450, 1.000],
    'r2': [0.000, 0.428, 0.000, 0.000, 0.000, 0.000, 0.537, 0.423, 0.000, 0.000, 0.318, 0.000, 0.407, 0.419,
           0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
    'r3': [0.000, 0.000, 0.762, 0.762, 0.762, 0.000, 0.000, 0.000, 0.710, 0.710, 0.000, 0.710, 0.000, 0.000,
           0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000],
    'tst': [0.000, 0.033, 0.008, 0.008, 0.008, 0.000, 0.034, 0.012, 0.007, 0.007, 0.001, 0.007, 0.033, 0.042,
            0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.004, 1.000]
}

# Create DataFrames
df_old = pd.DataFrame(old_data)
df_new = pd.DataFrame(new_data)

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
    }
}

def calculate_distance_to_asi(df, model_name):
    """Calculate Euclidean distance from a model to ASI across all metrics"""
    asi_row = df[df['Model'] == 'ASI'].iloc[0]
    model_row = df[df['Model'] == model_name].iloc[0]
    
    # Use all performance metrics
    metrics = ['p1', 'p2', 'p3', 'p4', 'r1', 'r2', 'r3', 'tst']
    
    asi_vector = [asi_row[metric] for metric in metrics]
    model_vector = [model_row[metric] for metric in metrics]
    
    return euclidean(asi_vector, model_vector)

def calculate_weighted_distance_to_asi(df, model_name, weights=None):
    """Calculate weighted distance to ASI, giving more importance to 'tst' score"""
    if weights is None:
        # Give higher weight to 'tst' as it's the most important metric
        weights = {'p1': 1, 'p2': 1, 'p3': 1, 'p4': 1, 'r1': 1, 'r2': 1, 'r3': 1, 'tst': 5}
    
    asi_row = df[df['Model'] == 'ASI'].iloc[0]
    model_row = df[df['Model'] == model_name].iloc[0]
    
    weighted_distance = 0
    total_weight = 0
    
    for metric, weight in weights.items():
        diff = abs(asi_row[metric] - model_row[metric])
        weighted_distance += (diff ** 2) * weight
        total_weight += weight
    
    return np.sqrt(weighted_distance / total_weight)

def analyze_model_progression():
    """Analyze the progression of AI models towards ASI"""
    results = []
    
    # ASI reference point
    asi_distance = 0  # ASI distance to itself
    
    print("=== MODEL PROGRESSION ANALYSIS ===")
    print("Distance to ASI (lower is better, 0 = perfect ASI performance)\n")
    
    for family_name, family_models in model_families.items():
        print(f"\n--- {family_name} Family ---")
        
        # Calculate distances for old models
        old_distances = []
        for old_model in family_models['old']:
            if old_model in df_old['Model'].values:
                dist = calculate_weighted_distance_to_asi(df_old, old_model)
                old_distances.append(dist)
                print(f"{old_model} (old): {dist:.4f}")
        
        # Calculate distances for new models
        new_distances = []
        for new_model in family_models['new']:
            if new_model in df_new['Model'].values:
                dist = calculate_weighted_distance_to_asi(df_new, new_model)
                new_distances.append(dist)
                print(f"{new_model} (new): {dist:.4f}")
        
        # Calculate improvement/regression
        if old_distances and new_distances:
            avg_old = np.mean(old_distances)
            avg_new = np.mean(new_distances)
            improvement = avg_old - avg_new  # Positive = improvement, Negative = regression
            improvement_pct = (improvement / avg_old) * 100 if avg_old > 0 else 0
            
            status = "IMPROVED" if improvement > 0 else "REGRESSED"
            print(f"Average change: {improvement:+.4f} ({improvement_pct:+.1f}%) - {status}")
            
            results.append({
                'Family': family_name,
                'Old_Distance': avg_old,
                'New_Distance': avg_new,
                'Improvement': improvement,
                'Improvement_Pct': improvement_pct,
                'Status': status
            })
    
    return pd.DataFrame(results)

def create_progression_visualization(results_df):
    """Create visualizations of model progression"""
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Distance comparison
    plt.subplot(2, 2, 1)
    x = np.arange(len(results_df))
    width = 0.35
    
    plt.bar(x - width/2, results_df['Old_Distance'], width, label='Old Models', alpha=0.8)
    plt.bar(x + width/2, results_df['New_Distance'], width, label='New Models', alpha=0.8)
    
    plt.xlabel('Model Family')
    plt.ylabel('Distance to ASI')
    plt.title('Distance to ASI: Old vs New Models')
    plt.xticks(x, results_df['Family'], rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Improvement percentage
    plt.subplot(2, 2, 2)
    colors = ['green' if x > 0 else 'red' for x in results_df['Improvement_Pct']]
    plt.bar(results_df['Family'], results_df['Improvement_Pct'], color=colors, alpha=0.7)
    plt.xlabel('Model Family')
    plt.ylabel('Improvement (%)')
    plt.title('Improvement Percentage (Positive = Better)')
    plt.xticks(rotation=45)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Progression trajectory
    plt.subplot(2, 2, 3)
    for i, row in results_df.iterrows():
        plt.plot([0, 1], [row['Old_Distance'], row['New_Distance']], 
                marker='o', linewidth=2, label=row['Family'])
    
    plt.xlabel('Generation')
    plt.ylabel('Distance to ASI')
    plt.title('Model Evolution Trajectory')
    plt.xticks([0, 1], ['Old', 'New'])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Summary ranking
    plt.subplot(2, 2, 4)
    sorted_results = results_df.sort_values('New_Distance')
    plt.barh(sorted_results['Family'], sorted_results['New_Distance'], alpha=0.7)
    plt.xlabel('Distance to ASI')
    plt.title('Current Model Ranking (Lower = Better)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return plt.gcf()

def calculate_asi_approach_rate():
    """Calculate how fast models are approaching ASI"""
    print("\n=== ASI APPROACH RATE ANALYSIS ===")
    
    # Calculate overall progress
    all_old_models = []
    all_new_models = []
    
    for family_models in model_families.values():
        all_old_models.extend(family_models['old'])
        all_new_models.extend(family_models['new'])
    
    # Calculate average distances
    old_distances = []
    for model in all_old_models:
        if model in df_old['Model'].values:
            old_distances.append(calculate_weighted_distance_to_asi(df_old, model))
    
    new_distances = []
    for model in all_new_models:
        if model in df_new['Model'].values:
            new_distances.append(calculate_weighted_distance_to_asi(df_new, model))
    
    avg_old_distance = np.mean(old_distances)
    avg_new_distance = np.mean(new_distances)
    
    overall_improvement = avg_old_distance - avg_new_distance
    overall_improvement_pct = (overall_improvement / avg_old_distance) * 100
    
    print(f"Average distance to ASI (old models): {avg_old_distance:.4f}")
    print(f"Average distance to ASI (new models): {avg_new_distance:.4f}")
    print(f"Overall improvement: {overall_improvement:+.4f} ({overall_improvement_pct:+.1f}%)")
    
    if overall_improvement > 0:
        print("✅ Models are generally getting CLOSER to ASI")
    else:
        print("❌ Models are generally getting FURTHER from ASI")
    
    return {
        'avg_old_distance': avg_old_distance,
        'avg_new_distance': avg_new_distance,
        'overall_improvement': overall_improvement,
        'overall_improvement_pct': overall_improvement_pct
    }

# Main analysis
if __name__ == "__main__":
    # Run the analysis
    progression_results = analyze_model_progression()
    
    print("\n=== SUMMARY TABLE ===")
    print(progression_results.to_string(index=False, float_format='%.4f'))
    
    # Calculate ASI approach rate
    approach_rate = calculate_asi_approach_rate()
    
    # Create visualizations
    fig = create_progression_visualization(progression_results)
    
    # Save results
    progression_results.to_csv('model_progression_analysis.csv', index=False)
    fig.savefig('model_progression_visualization.png', dpi=300, bbox_inches='tight')
    
    print("\n=== ANALYSIS COMPLETE ===")
    print("Results saved to: model_progression_analysis.csv")
    print("Visualization saved to: model_progression_visualization.png")