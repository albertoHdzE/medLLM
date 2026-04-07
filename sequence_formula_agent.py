#!/usr/bin/env python3
"""
Agentic System for Generating Mathematical Formulas for Time Series Sequences
Uses LangGraph and Ollama to simulate different AI models generating formulas
"""

import pandas as pd
import json
import random
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import ollama
from langgraph.graph import Graph, StateGraph, END
from langgraph.prebuilt import ToolExecutor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelPersonality(Enum):
    """Different AI model personalities for generating formulas"""
    CLAUDE_37 = "claude_3.7"
    GPT_4O_MINI = "gpt_4o_mini"
    LLAMA_4_SCOUT = "llama_4_scout"
    QWEN3 = "qwen3"
    CHATGPT_5 = "chatgpt_5"
    GROK4 = "grok4"
    DEEPSEEK_R1_0528 = "deepseek_r1_0528"
    OPUS_4 = "opus_4"
    MISTRAL_LARGE2405 = "mistral_large2405"
    GEMINI_25_PRO = "gemini_2.5_pro"
    CLAUDE_SONNET_4 = "claude_sonnet_4"

@dataclass
class SequenceState:
    """State for processing a sequence through the agent"""
    sequence: str
    complexity: int
    current_model: Optional[str] = None
    generated_formulas: Dict[str, List[str]] = None
    ollama_model: Optional[str] = None
    
    def __post_init__(self):
        if self.generated_formulas is None:
            self.generated_formulas = {}

class SequenceFormulaAgent:
    """Main agent class for generating mathematical formulas"""
    
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = None
        self.available_ollama_models = [
            "llama3.1:8b", "codellama:34b", "deepseek-coder:6.7b", 
            "llama3:8b", "deepseek-coder:latest", "gemma3:latest", "gemma3:4b"
        ]
        self.target_models = [model.value for model in ModelPersonality]
        self.model_personalities = self._create_model_personalities()
        self.graph = self._create_graph()
        
    def _create_model_personalities(self) -> Dict[str, Dict[str, Any]]:
        """Create personality profiles for each target model"""
        return {
            ModelPersonality.CLAUDE_37.value: {
                "style": "analytical and precise",
                "preference": "mathematical notation with clear explanations",
                "complexity_handling": "breaks down complex patterns step by step",
                "fallback_rate": 0.05,
                "ollama_model": "llama3.1:8b"
            },
            ModelPersonality.GPT_4O_MINI.value: {
                "style": "concise and practical",
                "preference": "simple formulas with alternative representations",
                "complexity_handling": "focuses on most obvious patterns first",
                "fallback_rate": 0.08,
                "ollama_model": "gemma3:latest"
            },
            ModelPersonality.LLAMA_4_SCOUT.value: {
                "style": "exploratory and creative",
                "preference": "multiple formula variations and recursive definitions",
                "complexity_handling": "attempts novel approaches to complex sequences",
                "fallback_rate": 0.12,
                "ollama_model": "llama3:8b"
            },
            ModelPersonality.QWEN3.value: {
                "style": "systematic and thorough",
                "preference": "algorithmic descriptions and code-like formulas",
                "complexity_handling": "provides step-by-step computational approaches",
                "fallback_rate": 0.07,
                "ollama_model": "deepseek-coder:6.7b"
            },
            ModelPersonality.CHATGPT_5.value: {
                "style": "balanced and educational",
                "preference": "clear mathematical expressions with context",
                "complexity_handling": "explains reasoning behind formula choices",
                "fallback_rate": 0.06,
                "ollama_model": "llama3.1:8b"
            },
            ModelPersonality.GROK4.value: {
                "style": "unconventional and witty",
                "preference": "creative formula representations, sometimes humorous",
                "complexity_handling": "finds unexpected patterns and connections",
                "fallback_rate": 0.15,
                "ollama_model": "gemma3:4b"
            },
            ModelPersonality.DEEPSEEK_R1_0528.value: {
                "style": "technical and code-oriented",
                "preference": "programming-style formulas and recursive definitions",
                "complexity_handling": "algorithmic approach with implementation details",
                "fallback_rate": 0.09,
                "ollama_model": "deepseek-coder:latest"
            },
            ModelPersonality.OPUS_4.value: {
                "style": "sophisticated and comprehensive",
                "preference": "elegant mathematical expressions with multiple approaches",
                "complexity_handling": "provides deep mathematical insights",
                "fallback_rate": 0.04,
                "ollama_model": "codellama:34b"
            },
            ModelPersonality.MISTRAL_LARGE2405.value: {
                "style": "efficient and direct",
                "preference": "optimized formulas with practical applications",
                "complexity_handling": "focuses on computational efficiency",
                "fallback_rate": 0.10,
                "ollama_model": "llama3:8b"
            },
            ModelPersonality.GEMINI_25_PRO.value: {
                "style": "versatile and adaptive",
                "preference": "context-aware formulas with multiple representations",
                "complexity_handling": "adapts approach based on sequence characteristics",
                "fallback_rate": 0.08,
                "ollama_model": "gemma3:latest"
            },
            ModelPersonality.CLAUDE_SONNET_4.value: {
                "style": "poetic and insightful",
                "preference": "elegant formulas with beautiful mathematical structure",
                "complexity_handling": "finds aesthetic patterns in mathematical relationships",
                "fallback_rate": 0.06,
                "ollama_model": "llama3.1:8b"
            }
        }
    
    def _create_graph(self) -> StateGraph:
        """Create the LangGraph workflow"""
        workflow = StateGraph(SequenceState)
        
        # Add nodes
        workflow.add_node("analyze_sequence", self._analyze_sequence)
        workflow.add_node("generate_formulas", self._generate_formulas)
        workflow.add_node("format_output", self._format_output)
        
        # Add edges
        workflow.add_edge("analyze_sequence", "generate_formulas")
        workflow.add_edge("generate_formulas", "format_output")
        workflow.add_edge("format_output", END)
        
        # Set entry point
        workflow.set_entry_point("analyze_sequence")
        
        return workflow.compile()
    
    def _analyze_sequence(self, state: SequenceState) -> SequenceState:
        """Analyze the sequence to understand its pattern"""
        logger.info(f"Analyzing sequence: {state.sequence}")
        
        # Parse sequence
        try:
            numbers = [int(x.strip()) for x in state.sequence.split(',')]
            logger.info(f"Parsed numbers: {numbers[:5]}...")
        except ValueError as e:
            logger.error(f"Error parsing sequence: {e}")
            
        return state
    
    def _generate_formulas(self, state: SequenceState) -> SequenceState:
        """Generate formulas for all target models"""
        logger.info(f"Generating formulas for all models")
        
        for model_name in self.target_models:
            try:
                formulas = self._generate_formula_for_model(state.sequence, model_name, state.complexity)
                state.generated_formulas[model_name] = formulas
                logger.info(f"Generated formulas for {model_name}: {formulas}")
            except Exception as e:
                logger.error(f"Error generating formulas for {model_name}: {e}")
                state.generated_formulas[model_name] = ["*not found"]
                
        return state
    
    def _generate_formula_for_model(self, sequence: str, model_name: str, complexity: int) -> List[str]:
        """Generate formulas for a specific model using Ollama"""
        personality = self.model_personalities[model_name]
        ollama_model = personality["ollama_model"]
        
        # Check if we should return "*not found" based on complexity and fallback rate
        if complexity >= 3 and random.random() < personality["fallback_rate"]:
            return ["*not found"]
        
        # Create model-specific prompt
        prompt = self._create_prompt(sequence, model_name, personality, complexity)
        
        try:
            # Call Ollama
            response = ollama.chat(
                model=ollama_model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.7, "top_p": 0.9}
            )
            
            # Parse response to extract formulas
            formulas = self._parse_formula_response(response['message']['content'], model_name)
            return formulas if formulas else ["*not found"]
            
        except Exception as e:
            logger.error(f"Ollama error for {model_name}: {e}")
            return ["*not found"]
    
    def _create_prompt(self, sequence: str, model_name: str, personality: Dict[str, Any], complexity: int) -> str:
        """Create a model-specific prompt for formula generation"""
        
        base_prompt = f"""You are simulating the AI model "{model_name}" with the following characteristics:
- Style: {personality['style']}
- Preference: {personality['preference']}
- Complexity handling: {personality['complexity_handling']}

Given the mathematical sequence: {sequence}

Your task is to generate 1-3 mathematical formulas that describe this sequence. 

IMPORTANT FORMATTING RULES:
1. Return ONLY a JSON list of strings, like: ["formula1", "formula2", "formula3"]
2. Each formula should be a string describing the mathematical relationship
3. Use standard mathematical notation (n, a_n, etc.)
4. If you cannot find a pattern, return: ["*not found"]

Examples of good responses:
- ["2n", "n + n", "2*n"]
- ["a_n = n^2", "a_n = a_{{n-1}} + 2n - 1"]
- ["Fibonacci(n)", "F_n = F_{{n-1}} + F_{{n-2}}"]

CRITICAL: All formulas must be enclosed in double quotes "" within the JSON array, not single quotes ''.

Sequence to analyze: {sequence}
"""

        # Add complexity-specific guidance
        if complexity == 1:
            base_prompt += "\nThis is a simple sequence (complexity 1). Focus on basic arithmetic patterns."
        elif complexity == 2:
            base_prompt += "\nThis is a moderate sequence (complexity 2). Look for polynomial, exponential, or well-known mathematical sequences."
        else:
            base_prompt += "\nThis is a complex sequence (complexity 3). It may not have an obvious pattern. Consider returning '*not found' if no clear pattern emerges."
            
        return base_prompt
    
    def _parse_formula_response(self, response: str, model_name: str) -> List[str]:
        """Parse the Ollama response to extract formulas"""
        try:
            # Try to find JSON list in the response
            import re
            
            # Look for JSON array pattern with double quotes
            json_pattern = r'\[.*?\]'
            matches = re.findall(json_pattern, response, re.DOTALL)
            
            if matches:
                # Try to parse the first match as JSON
                try:
                    formulas = json.loads(matches[0])
                    if isinstance(formulas, list) and all(isinstance(f, str) for f in formulas):
                        # Ensure all formulas are properly formatted as strings
                        return [str(f) for f in formulas[:3]]  # Limit to 3 formulas
                except json.JSONDecodeError:
                    pass
            
            # Fallback: extract strings within double quotes only
            double_quoted_pattern = r'"([^"]+)"'
            double_quoted_matches = re.findall(double_quoted_pattern, response)
            if double_quoted_matches:
                return double_quoted_matches[:3]
            
            # If no structured format found, return not found
            return ["*not found"]
            
        except Exception as e:
            logger.error(f"Error parsing response for {model_name}: {e}")
            return ["*not found"]
    
    def _format_output(self, state: SequenceState) -> SequenceState:
        """Format the output for CSV insertion"""
        logger.info("Formatting output for CSV")
        
        # Ensure all formulas are properly formatted as JSON strings
        for model_name in state.generated_formulas:
            formulas = state.generated_formulas[model_name]
            # Convert to JSON string format expected by CSV
            state.generated_formulas[model_name] = json.dumps(formulas)
            
        return state
    
    def load_csv(self):
        """Load the CSV file"""
        try:
            self.df = pd.read_csv(self.csv_path)
            logger.info(f"Loaded CSV with {len(self.df)} rows and {len(self.df.columns)} columns")
            return self.df
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            raise
    
    def process_single_sequence(self, sequence: str, model_name: str, complexity: int = 1) -> List[str]:
        """Process a single sequence for testing purposes"""
        try:
            logger.info(f"Processing sequence: {sequence} with model: {model_name}")
            result = self._generate_formula_for_model(sequence, model_name, complexity)
            logger.info(f"Generated result: {result}")
            return result
        except Exception as e:
            logger.error(f"Error processing single sequence: {e}")
            import traceback
            traceback.print_exc()
            return ["*not found"]
    
    def process_sequences(self):
        """Process all sequences in the CSV"""
        if self.df is None:
            self.load_csv()
        
        # Add new columns for target models
        for model_name in self.target_models:
            if model_name not in self.df.columns:
                self.df[model_name] = ""
        
        # Process each row
        for idx, row in self.df.iterrows():
            logger.info(f"Processing row {idx + 1}/{len(self.df)}")
            
            # Create state
            state = SequenceState(
                sequence=row['sequence'],
                complexity=row['Complexity']
            )
            
            # Run through graph
            result = self.graph.invoke(state)
            
            # Update DataFrame
            for model_name in self.target_models:
                self.df.at[idx, model_name] = result.generated_formulas.get(model_name, '["*not found"]')
    
    def save_csv(self, output_path: Optional[str] = None):
        """Save the updated CSV"""
        if output_path is None:
            output_path = self.csv_path
        
        try:
            self.df.to_csv(output_path, index=False)
            logger.info(f"Saved updated CSV to {output_path}")
        except Exception as e:
            logger.error(f"Error saving CSV: {e}")
            raise

def main():
    """Main function to run the agent"""
    csv_path = "/Users/alberto/Documents/projects/medLLM/multi-formula-time-series.csv"
    
    # Create agent
    agent = SequenceFormulaAgent(csv_path)
    
    # Process sequences
    logger.info("Starting sequence processing...")
    agent.process_sequences()
    
    # Save results
    agent.save_csv()
    logger.info("Processing complete!")

if __name__ == "__main__":
    main()