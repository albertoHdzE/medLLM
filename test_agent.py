#!/usr/bin/env python3
"""
Test script for the sequence formula agent.
Tests the agent on a small subset of sequences to ensure it works correctly.
"""

import pandas as pd
from sequence_formula_agent import SequenceFormulaAgent
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_agent():
    """Test the agent on a small subset of sequences."""
    
    # Initialize the agent
    agent = SequenceFormulaAgent('/Users/alberto/Documents/projects/medLLM/multi-formula-time-series.csv')
    
    # Load the CSV
    df = agent.load_csv()
    logger.info(f"Loaded CSV with {len(df)} rows")
    
    # Test on first 3 sequences
    test_sequences = df.head(3)
    
    logger.info("Testing agent on first 3 sequences:")
    for idx, row in test_sequences.iterrows():
        sequence = row['sequence']
        complexity = row['Complexity']
        
        logger.info(f"\nTesting sequence {idx + 1}: {sequence}")
        logger.info(f"Complexity: {complexity}")
        
        # Test with one model (claude_3.7)
        try:
            result = agent.process_single_sequence(sequence, 'claude_3.7')
            logger.info(f"Result for claude_3.7: {result}")
        except Exception as e:
            logger.error(f"Error processing sequence: {e}")
            
    logger.info("\nTest completed!")

if __name__ == "__main__":
    test_agent()