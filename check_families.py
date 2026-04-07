import pandas as pd
from comprehensive_plots import get_family, get_sorted_families

df = pd.read_csv('comparison_models_summary.csv')
EXCLUDE_MODELS = {
    'claude-3.7', 'claude-3-5-sonnet', 'deepseek_r1_0525', 'gemini-2.5-pro',
    'grok-3', 'gpt-4o-mini', 'chatgpt-4.5', 'gemini-3-pro'
}
cols = [c for c in df.columns if c not in EXCLUDE_MODELS and c not in ['case', 'Measure', 'complexity']]
models = cols
print("Models:", models)

families = get_sorted_families(models)
print("\nFamilies:", families.keys())

for fam, members in families.items():
    print(f"{fam}: {members}")
