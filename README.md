# medLLM: Comprehensive Evaluation Framework for Large Language Models

## Abstract

This repository constitutes a multi-faceted research initiative dedicated to the rigorous, empirical evaluation of Large Language Models (LLMs) across complex cognitive and computational domains. While originating as an architecture for medical domain adaptation, the scope of this repository has expanded significantly. It now provides a robust analytical framework for assessing abstract mathematical reasoning, generalized programmatic generation, zero-shot time-series forecasting, and the longitudinal progression of model families toward Artificial Super Intelligence (ASI) benchmarks.

## Core Research Vectors

### 1. Clinical and Medical Domain Adaptation
Initial paradigms within this project assess the viability of specialized foundation models localized to medical corpora. The repository features complete experimental pipelines for causal language modeling, fine-tuning, and robust querying. 
* **Models Analyzed**: BioGPT, ClinicalGPT, MedLLaMA, and DoctorGPT.
* **Methodology**: Optimization of models on contextual question answering and diagnostic inference against benchmarks derived from curated datasets (e.g., PubMedQA).

### 2. Algorithmic Sequence Generation and Mathematical Reasoning
A primary focus of our current empirical work lies in decoupling mere pattern memorization from genuine algorithmic reasoning. State-of-the-art models (including Claude Opus 4, Gemini 2.5 Pro, Deepseek R1, and LLaMA 4 Scout) are subjected to advanced sequence extrapolation tests.
* **Mathematical Formula Synthesis**: Evaluating model capacity to derive closed-form algebraic expressions for complex deterministic series.
* **Polyglot Code Generation**: Systematic validation of generated computational approaches across languages including Python, C++, Mathematica, MATLAB, JavaScript, R, and esoteric languages such as ArnoldC.
* **Validation Engine**: Responses are not merely token-matched; they are compiled, executed, and mathematically verified for algebraic equivalence and programmatic correctness through our custom `search_engine` validation suite.

### 3. Zero-Shot Time-Series Forecasting
Rigorous experiments testing the bounds of temporal foundation models against both deterministic mathematical sequences and highly stochastic processes (e.g., random binary sequences).
* **Analyzed Architectures**: TimeGPT, Lag-Llama, Chronos, and MOMENT.
* **Focus**: Measuring parameter efficiency, lag predictability, compression limitations, and comparative evaluations against standard classical autoregressive methodologies.

### 4. Longitudinal Progression toward ASI Benchmarks
We propose and implement a quantitative framework dedicated to measuring the architectural evolution of model families (Claude, Gemini, Mistral, Qwen, Deepseek). 
* **Weighted Euclidean Distance**: A specific multidimensional metric system measuring deviance from theoretically perfect Artificial Super Intelligence baselines.
* **Key Findings**: Identification of uniformity in performance ceilings and the revelation of architectural regressions in next-generation generalized models, highlighting constraints in current evaluation methodologies and training distributions.

## Repository Architecture

This codebase is structured to facilitate high-throughput, reproducible academic research:

* **`/data/`**: Centralized storage for large-scale evaluation results, simulation matrices, and structured medical JSON inputs.
* **`/search_engine/`**: A proprietary algorithmic evaluation engine capable of mathematical expression evaluation, symbolic logic verification, and programmatic execution.
* **`/plots/`**: Contains high-resolution, peer-review-ready visualizations (PDF/SVG) detailing robustness analyses, dimensionality clustering, and complexity thresholds.
* **Foundational Scripts**: Highly automated execution pipelines (`formula_eval_agent.py`, `simulate_new_llms.py`, `full_analysis.py`) designed to interface with model APIs asynchronously and measure outputs stochastically.
* **Experimental Notebooks**: Over 30 serialized Jupyter notebooks containing the theoretical derivations, hyperparameter optimizations, and visualization logic utilized throughout the project lifecycle.

## Reproducibility and Usage

The repository is built to support immediate extension by external researchers.

### Environment Initialization

```bash
git clone https://github.com/albertoHdzE/medLLM.git
cd medLLM
pip install -r requirements.txt
```

### Analytical Execution

To invoke the standalone analysis detailing the progression of model generations toward the ASI baseline, execute the following module:

```bash
python simple_model_analysis.py
```

Results, including statistical deltas and regressive indicators, are actively piped to terminal outputs and localized log directories.

## Scientific Integrity and Validation

All quantitative results within this repository adhere strictly to rigorous academic standards. Trivial syntax emulation is heavily penalized in the evaluation scripts; models must exhibit true isomorphic logic mapped to test sequences to achieve passing correctness Boolean flags. Codebase metrics reflect zero tolerance for data contamination or superficial heuristic pattern matching.
