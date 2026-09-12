# Experiment prompts

Rescued from scripts that are otherwise being deleted, because the prompt *is*
the method: a new model can only be added to the benchmark if it is asked the
same question as the existing 28.

Read the provenance notes carefully — one of these is solid, one is missing, and
one is a fabrication tool that must not be mistaken for either.

---

## 1. Formula generation (Figures 3 and 4, SI Figure 1)

**Status: recovered, and consistent with the published results.**
Source: `simple_formula_agent.py:79`, `create_direct_prompt()`.

```
You are a {model_personality} mathematical assistant. For each of the
{len(sequences)} sequences provided below, provide at least 3 different
formulas, mathematical models, or general expressions that could describe the
sequence. Output exactly {len(sequences)} lines, each containing a list of at
least 3 formulas.

The output must follow this exact format, with one list per line:
["formula_1", "formula_2", "formula_3"]
["formula_1", "formula_2", "formula_3"]
...

Each formula should be a concise mathematical expression (e.g., "n^2 + 1",
"2*n + 3", "fibonacci(n)") that fully captures the sequence for positive
integers n (starting from n=1 unless otherwise implied by the sequence). The
formula should use standard mathematical notation, be human-readable, and avoid
programming-specific syntax. If the sequence follows a well-known pattern (e.g.,
arithmetic, geometric, Fibonacci), use the standard mathematical form (e.g.,
"a_n = a_(n-1) + a_(n-2)" for Fibonacci).

Provide multiple valid formulas when possible - different mathematical
representations of the same pattern, or alternative ways to express the sequence.
If no valid formula can be determined for a sequence, include "not found" as one
of the formulas in the list.

Here are the {len(sequences)} sequences to analyze:

1. {sequence}
2. {sequence}
...

Provide exactly {len(sequences)} lines, each with a list of at least 3 formulas
in the format: ["formula_1", "formula_2", "formula_3"]
```

Two things about this prompt are load-bearing for the analysis:

* **"at least 3 formulas"** is what makes the equivalence metric meaningful.
  Equivalence is the pairwise agreement between the *sequences produced by* a
  model's several formulas, so a model returning one formula has produced
  nothing that can corroborate itself and scores 0. That is a measurement of the
  model, not an artefact: `grok_4.1`, `gemini-3-pro` and `mistral-large-3`
  returned a single formula on every one of the 90 sequences.
* **the `["a", "b", "c"]` output format** is what the ingestion pipeline parses
  into the `<model>_<i>` / `<model>_<i>_eval` columns. Answers that ignore the
  format still count — they are transcribed as given and evaluated as given.

`{model_personality}` was a knob in the local-agent harness. There is no evidence
it was used when querying the frontier models, and no record of what values it
took. Treat the prompt as beginning "You are a mathematical assistant."

---

## 2. Python script generation (Figures 5 and 6)

**Status: NOT RECOVERED.**

No prompt for the script-generation task survives anywhere in this repository.
Searched: every notebook and every script, for prompt-shaped strings mentioning
scripts or programs. Nothing.

What can be inferred from the data in `multi-python-script-time-series.csv`:

* each model cell holds a **stringified Python list of scripts**, e.g.
  `["print([2 * i for i in range(1, 11)])", "..."]` — so the output format was
  specified, and multiple solutions were requested, exactly as for formulae;
* the scripts are one-liners using semicolons for statement separation, which
  suggests a length or single-line constraint;
* solutions use only the standard library (see `superarc/sandbox.py`).

**Before adding a new model to Figures 5/6, this prompt must be reconstructed
and confirmed by the authors.** Using a different prompt would make the new
model's results incomparable with the published 29.

---

## 3. Model impersonation — NOT AN EXPERIMENT PROMPT

**Status: fabrication tooling. Recorded here only so it is never mistaken for
the method.**
Source: `sequence_formula_agent.py`, now deleted.

```
You are simulating the AI model "{model_name}" with the following
characteristics:
- Style: {personality['style']}
- Preference: {personality['preference']}
- Complexity handling: {personality['complexity_handling']}
...
```

This asked a *local* model to impersonate a frontier model and invent plausible
answers for it. Its output never reached the published datasets — verified by
comparing the rates claimed in `simulation_summary.md` against the live CSVs,
which disagree, and by the absence of its output file
`seriesWithLLMs_ext_expanded.csv`. It is deleted along with the other
synthetic-answer generators (`simulate_new_llms.py`, `add_claude37.py`,
`add_deepseek_r1.py`, `add_model_data.py`, `add_remaining_models.py`,
`update_csv_with_models.py`).

Answers must come from the model being evaluated. Nothing else counts.
