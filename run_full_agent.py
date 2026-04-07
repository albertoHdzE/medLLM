#!/usr/bin/env python3
"""
Full agent processing script.
Processes all sequences in the CSV and generates new model columns.
"""

import pandas as pd
from sequence_formula_agent import SequenceFormulaAgent
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Run the full agent processing."""
    
    # Initialize the agent
    agent = SequenceFormulaAgent('/Users/alberto/Documents/projects/medLLM/multi-formula-time-series.csv')
    
    logger.info("Starting full agent processing...")
    
    # Process all sequences
    agent.process_sequences()
    
    # Save the updated CSV
    output_path = '/Users/alberto/Documents/projects/medLLM/multi-formula-time-series-extended.csv'
    agent.save_csv(output_path)
    
    logger.info(f"Processing completed! Extended CSV saved to: {output_path}")

if __name__ == "__main__":
    main()