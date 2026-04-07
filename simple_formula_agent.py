#!/usr/bin/env python3
"""
Simple Formula Agent - Direct LLM approach for sequence formula generation
No verification, no complex workflows - just direct prompting as you specified.
"""

import pandas as pd
import ollama
import json
import re
from typing import List, Dict, Optional
import time
import argparse

class SimpleFormulaAgent:
    def __init__(self, csv_path: str, only_models: Optional[List[str]] = None):
        self.csv_path = csv_path
        self.df = None
        
        # Model configurations - simple personalities
        self.models = {
            'claude_3.7': {
                'name': 'llama3.1:8b',  # Fast and good
                'personality': 'analytical and precise'
            },
            'gpt_4o_mini': {
                'name': 'llama3:8b',  # Faster alternative to deepseek-coder:6.7b
                'personality': 'concise and mathematical'
            },
            'llama_4_scout': {
                'name': 'llama3.1:8b',  # Faster alternative to codellama:34b
                'personality': 'pattern-focused'
            },
            'qwen3': {
                'name': 'llama3:8b',
                'personality': 'efficient and clear'
            },
            'chatgpt_5': {
                'name': 'deepseek-coder:latest',  # Keep this one, it's smaller
                'personality': 'comprehensive and detailed'
            },
            'deepseek_r1_0528': {
                'name': 'gemma3:latest',
                'personality': 'thorough and systematic'
            },
            'opus_4': {
                'name': 'gemma3:4b',
                'personality': 'logical and step-by-step'
            },
            'mistral_large2405': {
                'name': 'llama3.1:8b',  # Fast model
                'personality': 'quick and intuitive'
            },
            'gemini_2.5_pro': {
                'name': 'llama3:8b',  # Fast model
                'personality': 'deep reasoning focused'
            },
            'claude_sonnet_4': {
                'name': 'gemma3:4b',  # Fast model
                'personality': 'code-oriented mathematical'
            }
        }

        # If only_models provided, filter to those keys
        if only_models:
            filtered = {k: v for k, v in self.models.items() if k in only_models}
            # Keep order by the original dict; if any not found, ignore silently
            self.models = filtered
    
    def load_csv(self) -> pd.DataFrame:
        """Load the CSV file"""
        self.df = pd.read_csv(self.csv_path)
        print(f"Loaded CSV with {len(self.df)} rows")
        return self.df
    
    def create_direct_prompt(self, sequences: List[str], model_personality: str) -> str:
        """Create the direct prompt for list format with at least 3 formulas per sequence"""
        
        prompt = f"""You are a {model_personality} mathematical assistant. For each of the {len(sequences)} sequences provided below, provide at least 3 different formulas, mathematical models, or general expressions that could describe the sequence. Output exactly {len(sequences)} lines, each containing a list of at least 3 formulas.

The output must follow this exact format, with one list per line:
["formula_1", "formula_2", "formula_3"]
["formula_1", "formula_2", "formula_3"]
...

Each formula should be a concise mathematical expression (e.g., "n^2 + 1", "2*n + 3", "fibonacci(n)") that fully captures the sequence for positive integers n (starting from n=1 unless otherwise implied by the sequence). The formula should use standard mathematical notation, be human-readable, and avoid programming-specific syntax. If the sequence follows a well-known pattern (e.g., arithmetic, geometric, Fibonacci), use the standard mathematical form (e.g., "a_n = a_(n-1) + a_(n-2)" for Fibonacci). 

Provide multiple valid formulas when possible - different mathematical representations of the same pattern, or alternative ways to express the sequence. If no valid formula can be determined for a sequence, include "not found" as one of the formulas in the list.

Here are the {len(sequences)} sequences to analyze:

"""
        
        for i, sequence in enumerate(sequences, 1):
            prompt += f"{i}. {sequence}\n"
        
        prompt += f"\nProvide exactly {len(sequences)} lines, each with a list of at least 3 formulas in the format: [\"formula_1\", \"formula_2\", \"formula_3\"]"
        
        return prompt
    
    def parse_response(self, response: str, num_sequences: int) -> List[str]:
        """Parse the LLM response to extract formula lists in the format ['formula1', 'formula2', 'formula3']"""
        lines = response.strip().split('\n')
        formula_lists = []
        
        for line in lines:
            line = line.strip()
            if line:
                # Try to parse as JSON list first
                try:
                    import json
                    if line.startswith('[') and line.endswith(']'):
                        parsed_list = json.loads(line)
                        if isinstance(parsed_list, list) and len(parsed_list) >= 3:
                            # Ensure at least 3 formulas, pad with "not found" if needed
                            while len(parsed_list) < 3:
                                parsed_list.append("not found")
                            formula_lists.append(str(parsed_list))
                            continue
                except:
                    pass
                
                # Fallback: look for list-like patterns with regex
                import re
                list_match = re.search(r'\[(.*?)\]', line)
                if list_match:
                    content = list_match.group(1)
                    # Extract quoted strings
                    formulas = re.findall(r'"([^"]*)"', content)
                    if len(formulas) >= 3:
                        formula_lists.append(str(formulas[:3]))  # Take first 3
                        continue
                    elif len(formulas) > 0:
                        # Pad with "not found" to reach 3
                        while len(formulas) < 3:
                            formulas.append("not found")
                        formula_lists.append(str(formulas))
                        continue
                
                # If no valid list found, create default
                formula_lists.append('["not found", "not found", "not found"]')
        
        # Ensure we have exactly the right number of formula lists
        while len(formula_lists) < num_sequences:
            formula_lists.append('["not found", "not found", "not found"]')
        
        return formula_lists[:num_sequences]
    
    def generate_formulas_for_sequence(self, model_name: str, sequence: str, seq_index: int, total_sequences: int) -> str:
        """Generate formulas for a single sequence using the specified model"""
        model_config = self.models[model_name]
        ollama_model = model_config['name']
        personality = model_config['personality']
        
        # Create the prompt for single sequence
        prompt = self.create_direct_prompt([sequence], personality)
        
        try:
            # Generate response
            response = ollama.chat(
                model=ollama_model,
                messages=[{
                    'role': 'user',
                    'content': prompt
                }]
            )
            
            response_text = response['message']['content']
            formula_lists = self.parse_response(response_text, 1)
            
            # Show progress
            formula_list_str = formula_lists[0] if formula_lists else '["not found", "not found", "not found"]'
            try:
                formula_list = eval(formula_list_str)
                print(f"    ✅ Sequence {seq_index + 1}/{total_sequences}: {formula_list}")
            except:
                print(f"    ✅ Sequence {seq_index + 1}/{total_sequences}: {formula_list_str}")
            
            return formula_list_str
            
        except Exception as e:
            print(f"    ❌ Sequence {seq_index + 1}/{total_sequences} - Error: {e}")
            return '["not found", "not found", "not found"]'
    
    def process_all_sequences(self):
        """Process all sequences and generate formulas for all models"""
        if self.df is None:
            self.load_csv()
        
        # Get all sequences
        sequences = []
        for _, row in self.df.iterrows():
            sequence_str = row['sequence']
            sequences.append(sequence_str)
        
        print(f"🚀 Starting formula generation for {len(sequences)} sequences...")
        print(f"🎯 Target: {len(self.models)} new model columns")
        print("=" * 60)
        
        # Process each model
        model_count = 0
        total_models = len(self.models)
        
        for model_name in self.models.keys():
            model_count += 1

            if model_name not in self.df.columns:
                print(f"\n📈 Progress: {model_count}/{total_models} models")
                print(f"🔧 Processing model: {model_name}")
                print(f"🤖 Using: {self.models[model_name]['name']} ({self.models[model_name]['personality']})")
                print("-" * 40)
                
                # Process sequences one by one
                formula_lists = []
                for i, sequence in enumerate(sequences):
                    formula_list_str = self.generate_formulas_for_sequence(model_name, sequence, i, len(sequences))
                    formula_lists.append(formula_list_str)
                    
                    # Save progress every 10 sequences
                    if (i + 1) % 10 == 0:
                        print(f"    💾 Checkpoint: {i + 1}/{len(sequences)} sequences processed")
                
                # Add to dataframe
                self.df[model_name] = formula_lists
                
                print(f"✅ Completed {model_name}: {len(formula_lists)} formula lists generated")
                print(f"💾 Column '{model_name}' added to CSV")
                
                # Save intermediate results to the same CSV path
                self.save_csv(self.csv_path)
                print(f"💾 Intermediate save completed to same CSV")
                
                # Small delay between models
                time.sleep(1)
            else:
                print(f"\n⏭️  Model {model_name} already exists in CSV, skipping...")
                
        print("\n" + "=" * 60)
        print(f"🎉 Completed! Processed {total_models} models for {len(sequences)} sequences")
    
    def save_csv(self, output_path: str = None):
        """Save the updated CSV"""
        if output_path is None:
            output_path = self.csv_path.replace('.csv', '_extended.csv')
        
        self.df.to_csv(output_path, index=False)
        print(f"Saved updated CSV to: {output_path}")
        return output_path

def main():
    print("🔬 Simple Formula Agent - Starting...")
    print("=" * 60)

    parser = argparse.ArgumentParser(description="Run Simple Formula Agent")
    parser.add_argument("--csv", default="multi-formula-time-series.csv", help="Path to CSV file")
    parser.add_argument("--models", nargs="*", help="Specific model columns to process (e.g., mistral_large2405 gemini_2.5_pro claude_sonnet_4)")
    args = parser.parse_args()

    # Initialize agent
    print("📂 Initializing agent...")
    agent = SimpleFormulaAgent(args.csv, only_models=args.models)

    # Process all sequences
    print("🔄 Beginning sequence processing...")
    agent.process_all_sequences()

    # Save results to same CSV
    print("\n💾 Saving results...")
    output_file = agent.save_csv(args.csv)
    print(f"✅ Results saved to: {output_file}")
    print("🎉 All done!")
    print("=" * 60)

if __name__ == "__main__":
    main()