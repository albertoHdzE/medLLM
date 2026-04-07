#!/usr/bin/env python3
"""
Test script for the Simple Formula Agent with list format
Tests that each sequence gets at least 3 formulas in the format ["formula1", "formula2", "formula3"]
"""

from simple_formula_agent import SimpleFormulaAgent
import pandas as pd
import json

def test_list_format():
    print("Testing Simple Formula Agent with list format...")
    
    # Initialize agent
    agent = SimpleFormulaAgent('multi-formula-time-series.csv')
    
    # Load CSV
    df = agent.load_csv()
    
    # Get first 3 sequences for testing
    test_sequences = []
    for i in range(min(3, len(df))):
        sequence_str = df.iloc[i]['sequence']
        test_sequences.append(sequence_str)
        print(f"Sequence {i+1}: {sequence_str}")
    
    print(f"\nTesting with {len(test_sequences)} sequences...")
    
    # Test one model
    test_model = 'claude_3.7'
    print(f"\nTesting model: {test_model}")
    
    formula_lists = agent.generate_formulas_for_model(test_model, test_sequences)
    
    print(f"\nResults for {test_model}:")
    for i, formula_list_str in enumerate(formula_lists):
        print(f"  Sequence {i+1}: {formula_list_str}")
        
        # Verify format
        try:
            formula_list = eval(formula_list_str)  # Safe since we control the format
            if isinstance(formula_list, list) and len(formula_list) >= 3:
                print(f"    ✓ Valid format: {len(formula_list)} formulas")
                for j, formula in enumerate(formula_list[:3]):  # Show first 3
                    print(f"      {j+1}. \"{formula}\"")
            else:
                print(f"    ✗ Invalid format: expected list with ≥3 items, got {type(formula_list)} with {len(formula_list) if isinstance(formula_list, list) else 'N/A'} items")
        except Exception as e:
            print(f"    ✗ Parse error: {e}")
    
    print("\nFormat verification completed!")

if __name__ == "__main__":
    test_list_format()