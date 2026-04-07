
import pandas as pd
import numpy as np

def debug_case(case_name):
    print(f"Debugging {case_name}...")
    try:
        summary_df = pd.read_csv('comparison_models_summary.csv')
    except FileNotFoundError:
        print("comparison_models_summary.csv not found.")
        return

    print(f"Summary shape: {summary_df.shape}")
    print(f"Columns: {summary_df.columns.tolist()}")
    
    if not summary_df.empty:
        # Filter for this experiment
        df_case = summary_df[summary_df['case'] == case_name]
        print(f"Filtered shape: {df_case.shape}")
        
        if df_case.empty:
            print(f"No data for case '{case_name}'")
            # Check unique cases
            print(f"Available cases: {summary_df['case'].unique()}")
            return
            
        # Identify valid models for this case (columns with non-zero values)
        # We assume columns are ordered in the CSV as intended
        metadata_cols = ['case', 'Measure', 'complexity']
        candidate_models = [c for c in summary_df.columns if c not in metadata_cols]
        
        models = []
        for m in candidate_models:
            # Check if this model has any non-zero data in this case
            # We need to check if ANY value is non-zero
            # Use abs() to handle negative if any (metrics shouldn't be negative but safe)
            s = df_case[m].abs().sum()
            print(f"Model {m}: sum={s}")
            if s != 0:
                models.append(m)
                
        print(f"Found {len(models)} models for {case_name}")
        print(models)

debug_case('multiple formulae')
debug_case('multiple script')
