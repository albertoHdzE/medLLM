# Plan — SuperARC repository: clean, refactor, and extend with new models

## Context

`doc/nature-paper.pdf` is the published record (Nature Communications, doi:10.1038/s41467-026-73289-5).
The task is to add newly-tested frontier models to the *live* experiments and regenerate the affected
figures and tables, without disturbing anything already published.

After Nature requested high-definition versions of every plot, a small set of producers was written
that generates the final published figures. **Locating that set was the key to this plan**, and it is
much smaller than the repository suggests: **7 files** out of 40 notebooks and 32 scripts.

Goal: a repository where dropping in new model answers and running **one command** regenerates every
live figure and table, with a verification pass proving nothing previously published moved.

## Ground rules (from the user, non-negotiable)

1. **Append-only data.** New results come from new data. Existing columns are never rewritten.
2. **Published results are preserved** exactly as printed.
3. **Never repair a model's answer.** An error made by a model is data. Translation to an executable
   form is allowed; silent fixing is not.
4. **Scripts *and* notebooks.** Notebooks verify the process is correct and stay.
5. **A comparison phase is mandatory.** Any change to an existing result is an error to find and fix.

---

## The canonical producer set (verified against the published PDF)

Every published figure was matched by rendering each page of `doc/nature-paper.pdf` in full and
comparing the printed figure against the candidate output files. The published figures come from
**two directories**, not one:

| Published | Producer | Canonical output | Status |
|---|---|---|---|
| Fig 1 | `22_Timeseries_LLM_experiments.ipynb` | `plots/highResolution/figure01` | static |
| Fig 2 | `22_Timeseries_LLM_experiments.ipynb` | **never saved to disk** | static |
| Fig 3 | `30-1_multiFormula_experiment.py` | `new_plots/figure03-up` | **live** |
| Fig 4 | `30-1_multiFormula_experiment.py` | `new_plots/figure03-bottom` | **live** |
| Fig 5 | `31-1_multiScript_experiment.py` | `new_plots/figure04-up` | **live** |
| Fig 6 | `31-1_multiScript_experiment.py` | `new_plots/figure04-bottom` | **live** |
| Table 1 | `34-2_S-ARC_ext.py` | **never written** | **live** |
| Fig 7 (ranking) | `34-2_S-ARC_ext.py` | `new_plots/figure05` | **live** |
| p₁–p₄ facets | `34-2_S-ARC_ext.py` `plot06` | `new_plots/figure06` (≡ `figpercent.png`) | **live** |
| φ by sequence type | `34-2_S-ARC_ext.py` `plot07` | `new_plots/figure07` (≡ `figtst.png`) | **live** |
| bootstrap tiers | `34-2_S-ARC_ext.py` | `new_plots/figure08` (≡ `robustness_tst.png`) | **live** |
| SI Fig 1 | `26_programming_codes.ipynb` cell 22 | `plots/highResolution/figure10` | **live** |
| SI Figs 2–4 | `24_PLOTS_paper.ipynb` | `plots/highResolution/figure11/12/13` | static |
| SI Figs 5–6 | `35_summary_statistics.py` | `plots/highResolution/ALL_FAMILIES_evolution_Part1/2` | **live** |
| SI Tables 1–2 | hand-written LaTeX | `plots/refactored_analysis.tex` | **live** |

**Seven producers total:** `30-1`, `31-1`, `34-2`, `35_summary_statistics.py`, and notebooks
`22`, `24`, `26`. Nothing else in the repository produces a published figure.

### ⚠ Published figure/caption misalignment — confirm with the journal

Rendering PDF pages 13–15 in full shows each caption describing a *different* figure than the one
printed above it. The three are rotated by one position:

| PDF page | Figure printed | Caption printed |
|---|---|---|
| 13 | bootstrap tiers (High/Medium/Low Performance) | **Fig. 8** — "Percentages by output types: p₁…p₄" |
| 14 | p₁–p₄ faceted dot plot | **Fig. 9** — "Test scores when different types of sequences are considered" |
| 15 | φ (Harmonic Mean Ratio) dot plot by Sequence Type | **Fig. 10** — "Bootstrap procedure…" |

Correct pairing: p₁–p₄ ↔ Fig 8, φ/test-scores ↔ Fig 9, bootstrap ↔ Fig 10.
This is a defect in the published article, not in the code — `34-2` generates all three correctly.
**Decision needed:** whether the update silently emits the correct pairing, or whether this warrants a
correction notice to Nature Communications. Either way the pipeline will emit figure↔caption pairs
explicitly so it cannot recur.

### Three corrections this search forced

1. **`34-2_S-ARC_ext.py` is the sole canonical producer of Figs 7–10**, and contains the bootstrap, so
   it is complete. My first plan listed it for deletion as a "superseded proposal"; my second listed
   `34-1` as co-canonical. Both were wrong.
2. **`34-1_S-ARC_ext.py` and `34_S-ARC_ext.ipynb` are superseded.** Their `figure06a/b/c` and
   `figure07a/b/c` grouped-bar variants appear nowhere in the paper; `34-2` replaced them with the
   faceted dot plots that were actually printed. They also collide with `34-2` on `figure05`/`figure08`,
   so run order silently decides the output.
3. **`plots/highResolution/figure03-*, figure04-*, figure05, figure06*, figure07*, figure08` are
   superseded** — landscape, side-by-side layouts from notebooks 30/31/34. The paper uses the stacked,
   right-legend layouts in `new_plots/`. Verified on pages 5, 7, 12 (`highResolution/figure04-up` is
   2440×1067 landscape; `new_plots/figure03-up` is 2129×1791 stacked; the paper shows stacked).
   `plots/highResolution/` stays canonical only for `figure01`, `figure10`, `figure11/12/13` and the
   evolution plots. `figure08-all`, `figure14`, `figure15` have no producer — orphans.

### `new_plots/` provenance, fully resolved

Only the four `-1`/`-2` scripts ever write to `new_plots/`. The remaining files are hand-made copies:
`figpercent.png` ≡ `figure06.png` and `figtst.png` ≡ `figure07.png` (identical MD5s);
`robustness_tst.png` is the renamed `figure08.png` (which is why no `figure08.png` exists);
`metricsForFormulasGeneration.png` is referenced by **no code at all** — a hand export of SI Fig 1,
whose code path writes `plots/highResolution/figure10`. All four are duplicates to remove.

### Static vs live

**Static** (closed experiments — polyglot programming languages, time-series forecasting): Figs 1, 2
and SI Figs 2–4. Kept reproducible, never re-run with new models.
**Live** (new models are added): Figs 3–10, Table 1, SI Fig 1, SI Figs 5–6, SI Tables 1–2.

---

## Reproducibility audit (measured, not assumed)

Ran the committed producers twice and compared pixel-by-pixel against the committed artifacts:

- **Run A vs run B → 0.0000% differ.** The code is deterministic. (My earlier claim of
  nondeterminism was wrong and is retracted.)
- Figs 3, 4, 7 → **byte-for-byte identical** to the committed artifacts. These pipelines are sound.
- Figs 5, 6 → **do not reproduce.** One accuracy series at complexity 2 reads ≈20% in the published
  PDF and ≈26.7% today — exactly +2 of 30 scripts now passing. Confirmed against *both* committed
  artifacts (`new_plots/figure04-up` and `plots/highResolution/figure04-up` both show ≈20%).
- `multi-python-script-time-series.csv` was **not** modified by the commit that added the figures.

**Cause:** Figs 5/6 are the only figures whose numbers come from *executing* generated Python
(`31-1_multiScript_experiment.py:88`, `exec(code, globals())`). Their values depend on interpreter and
library versions, and `requirements.txt` has been deleted, so nothing pins the environment.

## Defects blocking a one-command run

- **The SuperARC pipeline exits 1.** `setuptools` absent → `pkg_resources` missing; the script patches a
  shim in the parent but `joblib`/`loky` workers don't inherit it, so `from pybdm import BDM` fails in
  the bootstrap → `BrokenProcessPool`. Verified on `34-1`; `34-2` shares the identical `Parallel` call
  (line 453) and will fail the same way. **Table 1 and Figs 7–10 are currently unproducible.**
- Table 1 is never written — the script evaluates `np.round(...)` as a bare expression (a no-op).
- `34-1` computes the bootstrap **twice** (lines 260–312 and 541–591), discarding the first. `34-2`
  already dropped the redundant block — one more reason it supersedes `34-1`.
- `comparison_models_summary.csv` (input to SI Figs 5–6) **has no producer**; hand-maintained, and
  already contains duplicate columns for the same model under two naming schemes.
- `24_PLOTS_paper.ipynb` reads `/Users/beto/Documents/Projects/medLLM/...` — another machine's home.
- `processing_answers.py` **self-imports** (line 18) and instantiates `MOMENT-1-large` at import time.
  Every consumer pays this, including SI Fig 1, which needs none of it.
- SI Fig 1 replaces `***`/NaN with an **unseeded** random string before computing BDM — not reproducible.
- `tests/test_multiFormula_analysis.py` imports `scripts.multiFormula_analysis`, which does not exist.
- Four incompatible model-key conventions (`gpt_4o` / `chatgpt-4o` / `chatgpt_4o` / `gpt-4o`), with a
  hand-written 28-entry rename dict in each producer.
- **Credentials:** `processing_answers.py:41-42` hold two Nixtla API keys, committed and in git history.
  Deleting the lines will not remove them — **rotate them.**

## Existing work to reuse, not rebuild

| file | role |
|---|---|
| `formula_split_agent.py` | raw answer → individual formulas, via a local LLM (Ollama) |
| `formula_eval_agent.py` | formula → `_eval`; deterministic parser first, LLM fallback; `choose_best_range` resolves 0-9 vs 1-10 indexing |
| `deepseek_eval_columns.py` | safest variant: AST allowlist, `normalize_formula`, `strip_lhs_assignment`, deterministic `local_eval_line` first |
| `search_engine/math_exp_eval_engine.py` | `NumericStringParser` (also embedded in `formula_eval_agent.py` — three copies to collapse) |
| `31-1` `execute_code_safely` · `30-1` `compare_sequences` | script execution, sequence equality |

These already implement *deterministic first, model only when stuck*. They are the foundation.

**To delete — fabricates synthetic answers:** `simulate_new_llms.py`, `add_claude37.py`,
`add_deepseek_r1.py`, `add_model_data.py`, `add_remaining_models.py`, `update_csv_with_models.py`,
`simulation_summary.md`. Verified *not* to have contaminated published data (rates in
`simulation_summary.md` don't match the live CSVs; its output file doesn't exist). They are also the
**only** code that writes `-formula-correctness`/`-ordinal`/`-copy_seq`.

## The adjudication gap

`-formula-correctness`, `-formula-ordinal`, `-formula-copy_seq` are the entire input to the SuperARC-seq
score φ (Table 1, Figs 7–10). **No legitimate code computes them** — they arrive hand-labelled. That is
the manual bottleneck. The ~2,800 existing hand adjudications (100 binary sequences × 28 models) are the
gold set for automating it.

---

## Target architecture

```
superarc/
  registry.py          # ONE model registry: canonical key, per-dataset alias, display name,
                       #   family, release order — retires the four naming conventions
  datasets.py          # loaders + schema validation for datasets A–E
  metrics/             # equivalence · accuracy · classification · bdm · compression
  adjudicate/          # correctness · ordinal · copy_seq  (deterministic)
  ingest/              # extractor backends: lmstudio | claude-code | manual
  pipelines/           # formulae · scripts · superarc_seq · compression_metrics ·
                       #   evolution · timeseries · languages
  figures/             # one module per figure, named by PUBLISHED number
  compare.py           # old-vs-new drift report
  cli.py
data/raw/              # the ~33 genuine inputs + extended sequences.xlsx (sequence catalogue)
data/derived/          # model_summary.csv, ranking_table.csv (Table 1)
data/baseline/         # frozen published values + artifact hashes
outputs/<version>/     # versioned run: figures, tables, manifest.json
notebooks/             # thin verification drivers importing superarc
tests/
```

`python -m superarc.cli --all --out outputs/v2-2026-09` → every figure and table, plus a manifest with
input hashes, model list, package versions and git SHA.

Because the canonical producers are already four standalone `.py` files, Phase 3 is largely *moving
verified code into modules*, not rewriting it.

---

## Phases

### Phase 0 — Restore and freeze the baseline

1. `git checkout -- new_plots/` — exploratory smoke tests overwrote 10 tracked files. Restore first.
2. Pin the environment: rebuild `requirements.txt` from the venv **plus `setuptools`**, add a lockfile,
   record the Python version.
3. Build `data/baseline/`: SHA-256 of every existing CSV column and every published artifact, plus the
   numeric series behind each published figure. This is the permanent reference for the comparison phase.
4. Confirm SI Fig 1's canonical source (`plots/highResolution/figure10` from notebook 26) against
   `doc/sup-info-1.pdf`, since `new_plots/metricsForFormulasGeneration.png` has no producer.
5. Raise the figure/caption misalignment (pages 13–15) for a decision before any regeneration.

### Phase 1 — Reproducibility reconciliation (blocking)

Nothing ships until Figs 5/6 reproduce, or the deviation is understood and signed off.

1. Instrument the executor to record per-script pass/fail; diff against the published series to name the
   exact scripts whose outcome changed (target: the ≈20% → ≈26.7% cell).
2. Bisect Python and library versions to find the combination reproducing the published values. Pin it.
3. Harden the executor so it cannot recur: isolated namespace per script (not `exec(code, globals())`),
   subprocess with timeout, fixed `PYTHONHASHSEED`, seeded `np.random` (`31-1:409` is unseeded).
4. **Decision gate:** if the published values cannot be reproduced under any reconstructible environment,
   stop and report the exact per-model delta for a ruling. Nothing is republished without sign-off.
5. Fix `setuptools`/`pkg_resources` and run `34-2` end to end — currently it cannot complete, so
   Figs 7–10 and Table 1 have **never been parity-checked**. Verify all four reproduce the published
   artifacts before `34-1` and notebook 34 are deleted.
6. Repeat parity for every live figure. Figs 3, 4, 7 already pass.

### Phase 2 — Cleaning

On branch `clean`; `master` retains everything.

| category | count | examples |
|---|---|---|
| CSVs referenced by no code | 32 | `*_backup.csv`, `multi-formula-time-series-TEST.csv`, `print_filter1..4.csv` |
| CSVs referenced only by deleted code | ~32 | medical/BioGPT and agent-branch evaluation sets |
| exact duplicate | 1 | `seriesWithLLMs_ext_Aug2025.csv` (identical MD5 to `Dic2025`) |
| out-of-scope notebooks | ~33 | medical adaptation (00–10), fine-tuning (11–12), Lag-Llama colab copies |
| fabricated-data scripts | 7 | listed above |
| superseded / debug scripts | ~13 | `34-1_S-ARC_ext.py`, `34_S-ARC_ext.ipynb`, `full_analysis.py`, `model_progression_*.py`, `check_families.py`, `debug_models.py`, `fix_altair_charts.py`, `test_*.py` |
| superseded figure outputs | 14 | `plots/highResolution/figure03-*, figure04-*, figure05, figure06*, figure07*, figure08` (landscape, not published) |
| duplicate outputs in `new_plots/` | 4 | `figpercent.png`, `figtst.png`, `robustness_tst.png`, `metricsForFormulasGeneration.png` |
| orphan outputs | 6 | `figure08-all`, `figure14`, `figure15` (.pdf/.svg) |
| stale root PNGs | 58 | all regenerable or superseded |
| empty files / dirs | 5 | `08_gptplusplus.ipynb`, `31_..._PROPOSALS_v3.py`, `Sequence-Searcher/` |
| build artifacts | 8 | `nature_cover_letter.*`, `octave-workspace`, `out` |

**Keep:** the 7 producers (`30-1`, `31-1`, **`34-2`**, `35_summary_statistics.py`, notebooks 22/24/26),
`search_engine/`, `tests/`, `plots/refactored_analysis.tex`, the three ingestion agents,
`extended sequences.xlsx`.

`34-1_S-ARC_ext.py` and `34_S-ARC_ext.ipynb` are deleted only *after* Phase 1 proves `34-2` reproduces
Figs 7–10 — they are the fallback if the environment bisect needs them.

`28_full_random_bin_seq_process.ipynb` is deleted with the rest. Flagging once, factually: it contains
cells headed `# PUNISHING CLAUDE` with hand-tuned ranking formulas — superseded exploratory scratch,
never published, unrelated to φ. Better removed than found by a stranger.

Gate: after cleaning, every live figure still regenerates identically to the Phase 0 baseline.

### Phase 3 — Refactor

1. Build `superarc/`; move logic in from the 7 producers, leaving no copies behind.
2. Collapse the three `NumericStringParser` copies to one owner.
3. Name every output by its **published figure number**, paired explicitly with its caption, so the
   page-13/14/15 misalignment cannot recur.
4. Fix: install `setuptools`, drop the `pkg_resources` monkey-patch; **emit Table 1** to
   `data/derived/ranking_table.csv` + LaTeX; repair the `/Users/beto/` path; add the missing `savefig`
   for Fig 2; seed SI Fig 1's RNG.
5. Split `processing_answers.py`: pure functions into `superarc/metrics/`; MOMENT instantiation behind a
   function so import is free; remove the self-import; strip the API keys.
6. **Generate `comparison_models_summary.csv`** from datasets A and B so SI Figs 5–6 derive from raw data.
   Validate the generated file reproduces the current one before switching over.
7. Rename outputs to published numbering; one output directory, not two.
8. Notebooks: six thin verification drivers (formulae, scripts, superarc-seq, evolution, compression,
   comparison) that `import superarc`, show the intermediate tables at each stage, then render the
   figure. No logic inside; each runs in minutes.
9. Revive `tests/`: retarget to `superarc.pipelines.formulae`; add regression tests pinning the published
   Table 1 values and the Phase 0 baseline series.

### Phase 4 — Ingestion assistant

**Principle: the LLM extracts and translates. Deterministic code adjudicates.** A benchmark concluding
that LLMs only simulate comprehension cannot have an LLM deciding whether an answer was correct;
`correctness` is decidable by running the formula.

Verification is asymmetric, and that is what makes it tractable:
> If translated code reproduces the target sequence, that is **proof** — the translator cannot fake it.
> If it does not, it is **ambiguous** (model wrong, or translator wrong) → review queue.

False positives are near-impossible and the review queue scopes itself, with no confidence heuristics
and no model judging anything.

1. **Calibrate before building.** `gpt_4o` in `multi-formula-time-series.csv` has 6 formula slots and
   4.36 formulas/row across 90 sequences — ~390 cases where the raw answer *and* the hand-produced
   `_eval` both already exist. Run split → translate → evaluate and score exact match against it.
   **Read-only; scratch output; no CSV opened for writing.** Compare `qwen2.5-coder-32b-instruct-mlx`
   vs `llama-3.3-70b-instruct` (LM Studio on :1234, confirmed running). Avoid the two Claude-distilled
   Qwen3.5 models — provenance optics on a benchmark that ranks Claude. Then repeat on a messier model
   (`claude_sonnet_4`: 9 slots, 2.10/row). **Gate: ≥95% exact match, <5% flagged.**
2. Build `superarc/adjudicate/` — deterministic `correctness`, `copy_seq`, `ordinal` per the paper's
   definitions — and validate against the ~2,800 existing hand labels. Disagreements go to the user;
   rulings fold back in as test cases. This validates the historical labels *and* automates future ones.
3. Build `superarc/ingest/` around the three existing agents: one `Extractor` interface, backends
   `lmstudio` | `claude-code` | `manual`, shared JSON schema, plus the verification layer —
   **span grounding** (every extraction carries a verbatim quote asserted to occur in the raw file,
   killing hallucinations mechanically), coverage checks, temperature 0, a cache keyed on
   `hash(raw_text + prompt + model)` doubling as a provenance record, and a `needs_review` queue.
4. Provenance column per model: `hand-2024` vs `assistant-<model>-v1`. One dataset, every cell labelled
   with who adjudicated it.

### Phase 5 — Ingest new models and regenerate

1. `superarc ingest --model <name> --answers raw/<name>.txt` — appends columns only.
2. **Frozen-baseline gate:** hash every pre-existing column before and after; any run that would modify
   a published value aborts.
3. Regenerate live figures into `outputs/v2-2026-09/`.
4. `superarc compare` against `data/baseline/`: every previously-published series must be unchanged.
   Drift is a bug to fix, not a result to accept.
5. Update SI Tables 1–2 in `plots/refactored_analysis.tex`, checking quoted numbers against the
   regenerated data rather than retyping them.

---

## Verification

- `pytest -q` — unit tests plus regression tests pinning published Table 1 values.
- `python -m superarc.cli --all --out outputs/<v>` — full run from a clean checkout.
- `python -m superarc.compare --baseline data/baseline --run outputs/<v>` — must report **zero drift**
  on every previously-published series. This is the acceptance criterion for the whole project.
- Pixel parity: each live figure regenerated twice must be identical, and identical to baseline for the
  models present in both.
- The six verification notebooks run top to bottom without error and display their intermediate tables.
- Manifest present with input hashes, model list, package versions, git SHA.

## Risks

1. **Figs 5/6 may not be reproducible** if the April environment can't be reconstructed. Phase 1 step 4
   is an explicit stop-and-ask gate; nothing is republished without a ruling.
2. **A local model may miss the 95% bar.** Found out in Phase 4 step 1, before anything is built around
   it; fall back to a larger backend or manual ingestion.
3. **Translation false negatives** would look exactly like model failures. Mitigated by routing every
   non-reproducing case to review, and by regression tests that plant known-broken formulas and assert
   they stay broken.
4. **Deletion regret.** Everything survives on `master`; Phase 2 is gated on figure parity, and nothing
   is deleted until its replacement has been proven to reproduce the published artifact. The `34-1`/`34-2`
   question was answered wrongly twice before the PDF settled it — the gate is necessary, not ceremonial.
5. **Figure/caption misalignment** in the published article (pages 13–15). Needs a decision from the
   authors, and possibly the journal, before the updated figures are placed.

---

# STATUS — updated after Phase 3.3

Ten commits on branch `clean`, from `17038a3` to `b4e5322`. 40 tests pass. Working tree clean.

## Done

| phase | outcome |
|---|---|
| 0 | Baseline frozen: 67 artifacts, 97 inputs, **532 dataset columns** hashed. Environment pinned (`setuptools<81` is load-bearing — 81 deprecated and 84 removed `pkg_resources`, which pybdm needs). |
| 1 | Every published figure reproduces. Two root causes fixed. |
| 2 | 33 notebooks, 62 data files, 58 PNGs, 20 scripts removed. 40 notebooks → 5; 97 CSVs → 35. |
| 3.1 | `superarc/registry.py` — one model registry replacing six hand-written rename dicts. |
| 3.2 | `superarc/table1.py` — Table 1 emitted for the first time; reproduces published exactly. |
| 3.3 | Prefix-collision corrected in the formulae figures; SI evolution figures added to the gate. |

## Two commands define the guarantee

    python -m superarc.parity            # regenerate everything, compare to published
    python -m superarc.baseline verify   # no published value has moved

## Authorised corrections (author-ruled, each recorded in superarc/parity.py)

1. **Formulae figures** — columns were matched by prefix, so five models absorbed a
   longer-named model's answers. DeepSeek accuracy at complexity 2: 16.19 → 43.33.
   Qwen equivalence at complexity 1: 42.22 → 0.00 (correct: Qwen alone has one variant).
2. **Integrated Script Analysis** — the Gemini-2.5-Pro bar was drawn from the `gemini`
   column, understating valid-script volume ninefold (82 → 750).
3. **Bootstrap tiers** — `np.random.default_rng()` ignored the seed above it. Now seeded;
   three labels corrected to their exact values (o1-Mini 0.035 → 0.034, Claude-3.5
   0.034 → 0.033, Llama-4-Scout 0.005 → 0.004).
4. **Script sandbox** — pinned to the standard library. Four o1-Preview scripts call
   `sympy.primerange`; sympy's accidental presence had moved o1-Preview from 20.00% to
   26.67%.

## Do not repeat these mistakes

- **Never identify a figure by its number.** The published caption/figure pairing is
  offset on several pages. Use the title. See `superarc/parity.py` for the mapping.
- **Never match a model column by prefix.** The model name is an atomic keyword;
  resolve suffix-first then longest-name-wins (`superarc.registry.resolve_column`).
- Supplementary Figures 5/6 **do** reproduce pixel-identically. They are not broken;
  they merely cannot yet take a new model.
- **Never call a figure file an orphan without rendering it.** `figure14` and
  `figure15` were listed above as having no producer; they are the
  high-resolution Supplementary Figures 5/6, identical in content to
  `ALL_FAMILIES_evolution_Part1/2`. Duplicates, not orphans.
- **A missing artifact is not evidence that code is dead.** `25_BDM.ipynb` was
  deleted in Phase 2 because nothing on disk pointed to it. It produces a
  published panel.

## Phases 3.4 – 3.6 (done)

| commit | outcome |
|---|---|
| `360f03b` | The three panels of Figure 1 and all of Figure 2 are saved for the first time. |
| `36e911c` | Supplementary Figure 1 is producible; `processing_answers.py` is importable. |
| `d5a3634` | `python -m superarc.cli --out outputs/<v>` regenerates everything, with a manifest. |

### Figure 1 is three images, and only one was ever saved

The typesetter stacked three separately produced figures. Their producers:

| panel | producer | was it saved? |
|---|---|---|
| Success Rate — Simple climbers | `superarc.timeseries` | yes → `figure01` |
| Success Rate — Random Binary Sequences | `superarc.timeseries` | **no** |
| BDM / Shannon / zip / lzw by complexity | `superarc.complexity_measures` | **no** |

The one that *was* saved is the control: it regenerates at 0.0000% pixel
difference, which is what licenses trusting the other two.

**`25_BDM.ipynb` was restored.** Deleting it in Phase 2 was a cleaning error — it
is the sole producer of the bottom panel. The Phase 2 gate did not catch this
because the panel had no artifact to compare against, which is exactly the hole
that saving these figures closes.

### Two more figures that are not what they look like

- **Figure 2's image IS in the published article — on page 9, under the caption
  numbered Fig. 6.** ~~absent from the published article~~ was wrong: it was
  reached by looking for the image beneath its own caption, which is exactly the
  inference this document warns against three lines below. Corrected 2026-09-14
  after the author pointed at the printed page.
  The article composes the three panels into one 2×2 grid under a shared title.
  **No code ever did that.** `22_Timeseries_LLM_experiments.ipynb` cell 53 draws
  one `plt.figure` per (seq_type, method) and calls `plt.show()`; `suptitle` and
  `subplots` appear zero times in the notebook, and the cell emitted six separate
  inline images. The grid was assembled by hand from those. So the panels stay
  one per file — not because a combined layout would be invented, but because
  searching history for its producer returns nothing to find. (Verified across
  all 185 `.py`/`.ipynb` blobs ever committed. A first scan reported zero hits
  for everything, including strings known to be present: `git` was not on the
  PATH inside the loop, so every lookup failed silently and the scan proved
  nothing. **A search that finds nothing must be tested against a known hit
  before it is believed.**)
- **The notebook's stored outputs are not a reference — they are wrong.**
  Cell 53 carries six rendered panels, which look like the missing artifact for
  Figure 2. They disagree with the printed page at complexity 3 (stored
  `gral_simi` 6.02, printed 1.57). Cause: the cell does
  `total_df["complexity"].replace({3: 4, 4: 3})` **in place**, so running it an
  even number of times swaps the two hardest tiers back, and the committed
  output came from such a session. `average_by_complexity` does it on a copy.
  The regenerated panels match the article; the committed images do not.
  Pinned in `tests/test_timeseries.py`.
- **The article does not print the high-resolution Figure 1.** Sampling page 4
  gives matplotlib's default `tab10` cycle — the draft that preceded the
  restyle. The bar heights are identical, so no result is affected.

### Verified rather than assumed

- **Levenshtein.** `processing_answers.py` took it from `enchant.utils`, a
  binding to a C library that is not installed. Reimplemented and held to all
  1920 published distances. Recovering them required noticing that the two call
  sites disagree: the Chronos/TimeGPT columns come from `num_list_to_string`
  (`"1 11 21"`), the lag-llama columns from the bracketed `str(list)`. Both
  preserved as published.
- **The bottom panel's four series were typed in as literals.** Recomputed now.
  All agree to the digits printed except the third BDM point, transcribed as
  `549.678` where it computes to `549.687` — two digits transposed, far below one
  pixel on an axis spanning 470–550.

### Supplementary Figure 1 is reproducible only up to a random draw

Answers that were `***` or empty are replaced by a random 45-character string, so
that "no answer" scores as incompressible rather than as zero. The draw was
unseeded. Seeded now, and measured — spread across five seeds as a fraction of
each panel's range:

| panel | sensitivity | | panel | sensitivity |
|---|---|---|---|---|
| Avg LZW | 0.0% | | Avg BDM (ZIP) | 3.7% |
| Avg ZIP | 0.0% | | Avg BDM | 7.4% |
| Avg Shannon | 3.3% | | Avg BDM (LZW) | 10.6% |

The compressed-length panels are exact — the compressed length of a random string
does not depend on which characters were drawn. The BDM panels are not, because
BDM is sensitive to the exact bit pattern, which is precisely why it is the
measure the paper is built on. Seed 42 lands 3.3% of pixels from the published
figure and two other seeds land 3.1% from each other, so the draw is the whole
gap. **Choosing a reference seed is an author's decision.**

### `processing_answers.py`

Downloaded, initialised and printed a MOMENT-1-large pipeline at import time. The
variable was referenced exactly twice — to create it and to print it. Every
importer paid a `torch` dependency and a model download for nothing, which is why
the module could not be imported at all on a clean install. Now behind
`load_moment_pipeline()`; self-import removed; `torch` imported where used; the
Nixtla API key removed. **Removing it does not remove it from the history — the
key must be rotated.**

## Next

1. README rewrite, and the remaining verification notebooks (22, 25 and 26 are
   already thin drivers).
2. Phase 4 ingestion assistant: calibrate on `gpt_4o` (~390 hand-done cases)
   before building anything; gate at >=95% exact match.

## Open for the authors

- Figure/caption misalignment in the published PDF — may warrant a correction notice.
- **The Python-script prompt survives nowhere** (`doc/prompts.md`). New models cannot be
  added to the script figures until it is reconstructed.
- `DeepSeek-R1-0525` vs `-0528`: one model, two labels in print.
- Whether to switch the family-evolution figures onto the generated summary
  (`superarc/model_summary.py`, written but deliberately unwired).
