#!/usr/bin/env python3
"""
Test script for the Simple Formula Agent
Tests on first 3 sequences to validate the approach
"""

from simple_formula_agent import SimpleFormulaAgent
import pandas as pd

def test_simple_agent():
    print("Testing Simple Formula Agent...")
    
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
    
    formulas = agent.generate_formulas_for_model(test_model, test_sequences)
    
    print(f"\nResults for {test_model}:")
    for i, formula in enumerate(formulas):
        print(f"  Sequence {i+1}: \"{formula}\"")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_simple_agent()