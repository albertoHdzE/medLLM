# SuperARC — working repository

Code and data behind:

> **SuperARC: a test for artificial superintelligence based on compressed modelling,
> recursive prediction and problem complexity**
> Alberto Hernández-Espinosa, Luan Ozelim, Felipe S. Abrahão & Hector Zenil
> *Nature Communications* (2026) — received 23 June 2025, accepted 8 May 2026
>
> **Read it: https://www.nature.com/articles/s41467-026-73289-5**
> DOI: [10.1038/s41467-026-73289-5](https://doi.org/10.1038/s41467-026-73289-5)

## Which repository is this?

**This is the working repository.** It is where the analysis is developed,
corrected and extended, and it moves. Figures here may be ahead of what was
printed, and where they are, the difference is recorded and explained rather than
quietly applied — see [Corrections to the published figures](#corrections-to-the-published-figures).

**The official repository is [`AlgoDynLab/SuperintelligenceTest`](https://github.com/AlgoDynLab/SuperintelligenceTest)**,
maintained by the **Algorithmic Dynamics Lab, Center of Molecular Medicine,
Karolinska Institute & King's College London**. That is the repository the paper
points to and the one to cite or fork if you want the published state.

If you are reproducing the paper, start there. If you want to see how the results
were checked, re-derived and extended after publication, you are in the right
place.

---

## Reproduce everything, in one command

```bash
python -m superarc.cli --out outputs/v2-2026-09
```

Regenerates every published figure and table into `outputs/`, plus a
`manifest.json` recording the SHA-256 of every input dataset, the models found in
each, the installed package versions, the Python version and the git commit.
A figure without that is not reproducible, only re-drawable.

**No credential is needed.** The forecasting experiment is closed and its results
are committed CSVs; both back ends stay `None`. See `.env.example` if you want to
*re-run* the forecasts, which is not part of reproducing the paper.

## Prove nothing moved

```bash
python -m superarc.parity        # regenerate, compare against the published artifacts
python -m superarc.baseline verify   # no published value has moved
pytest                           # 212 tests
```

`parity` is deliberately a separate command from `cli`: one produces a new
version, the other proves the old one did not change. It rasterises and compares
pixel by pixel — PDF and SVG both embed a creation timestamp, so bytes never
match even when content does — and every figure lands in exactly one of four
states:

| state | meaning |
|---|---|
| `IDENTICAL` | regenerates pixel for pixel |
| `authorised` | deliberately differs; the reason is recorded in `superarc/parity.py` |
| `newly saved` | the code drew it and never wrote it to disk; no reference exists |
| `NOT PRODUCED` | a failure |

Corrected and newly-saved figures are additionally held to run-to-run
determinism, so an unintended change on top of an authorised one is still caught.

## Layout

```
superarc/           the package — all logic lives here
  registry.py         one model registry; four datasets name the same model four ways
  table1.py           the SuperARC-seq score φ
  parity.py           the acceptance gate, and the record of every authorised change
  cli.py              one-command regeneration + manifest
  baseline.py         frozen published values and artifact hashes
  formulae.py  scripts.py  superarc_seq.py  timeseries.py
  languages.py  evolution.py  complexity_measures.py  compression_metrics.py
  sandbox.py          stdlib-only executor for model-generated Python
notebooks/          eight verification drivers; they import, display and plot, never compute
tests/              212 tests, including structural guards against duplicated logic
doc/                the published article, supplementary information and peer review
plots/ new_plots/   the published artifacts — the reference, never written to
outputs/            git-ignored; where regeneration goes
```

The four top-level scripts with numeric names — `30-1_multiFormula_experiment.py`,
`31-1_multiScript_experiment.py`, `34-2_S-ARC_ext.py`, `35_summary_statistics.py` —
are the names that produced the published figures. They are kept as forwarders
onto the package so that every result in the paper still traces to the name it was
produced under.

## Install

```bash
python -m venv venv && ./venv/bin/pip install -r requirements.txt
./venv/bin/pip install -e . --no-deps
```

Python 3.13. `requirements.txt` is the curated set, and the one to use —
`setuptools>=70,<81` is pinned there and is load-bearing: `pybdm` imports
`pkg_resources` at module load, `joblib`'s worker processes do not inherit a
monkey-patched shim, and setuptools 84 removed `pkg_resources` outright, which
breaks the bootstrap behind Figure 10. `requirements.lock.txt` is the full
134-package capture of the environment the reproducibility audit ran in; it does
*not* constrain setuptools, so use it to match that environment exactly, not on
its own.

Figures 5 and 6 execute model-generated Python, so their values depend on this
environment. Do not bump these without re-running `python -m superarc.parity`.

For the notebooks, register the kernel and select **SuperARC (venv)**:

```bash
./venv/bin/python -m ipykernel install --user --name superarc --display-name "SuperARC (venv)"
```

See `notebooks/README.md` — there is a `.venv/` in this repository that is *not*
the project environment, and selecting it fails on the first cell.

## Corrections to the published figures

Reproducing the figures exactly turned up defects in the code that drew them.
Each is recorded in full in `superarc/parity.py`, with what moved and by how much.
In summary:

- **Figures 3 and 4** — model columns were matched by prefix, so five models
  absorbed a longer-named model's answers.
- **Figures 5 and 6** — accuracy selected columns with a substring test; and 216
  cells holding the string `"*not found*"` were indexed as if they were a list of
  eleven programs, so a missing answer became eleven wrong ones.
- **Figure 6, and Supplementary Figure 5** — the bar labelled Gemini-2.5-Pro was
  drawn from the `gemini` column, understating its volume ninefold (82 vs 750).
- **Figure 8** — the reshape deciding which row each model occupies used
  `order='F'`; 173 of 448 points showed another model's value.
- **Figure 10** — the bootstrap called `np.random.seed(42)` and then
  `np.random.default_rng()`, which ignores it. All 28 published means are
  recovered; see the note in `superarc/superarc_seq.py`.

Two further things a reader should know:

**Figure identity is by title, never by number.** In the published PDF the caption
printed under a figure sometimes describes a different one — pages 5, 8, 9, 13 and
15 are offset. `superarc/parity.py` holds the mapping from content to producer.

**Two panels cannot be recovered and are not claimed to be.** Supplementary
Figure 1 substitutes an unseeded random string for missing answers, and the lower
panel of Supplementary Figure 2 is packed using Python's per-process string hash.
Both are seeded now; neither published image is recoverable, and the underlying
data is pinned in tests instead.

## Method, in one paragraph

Models are asked to extend integer sequences of increasing complexity, and to
produce a *generating model* — a formula, or a program — rather than the next
terms. An answer that reproduces the sequence is proof; an answer that does not is
ambiguous rather than false. Scoring is by algorithmic complexity (BDM/CTM) rather
than by string match, which is what makes the test human-agnostic: printing the
sequence back scores as incompressible, and is worth a hundredth of a real answer.

Model answers are **never repaired**. An error made by a model is data — it is the
model's intelligence under evaluation, so a hallucination or a syntax error is
passed through as given. Translation into an executable form is allowed; silent
fixing is not, and no code in this repository generates, simulates or completes a
model's answer.

## Citation

```bibtex
@article{HernandezEspinosa2026SuperARC,
  title   = {SuperARC: a test for artificial superintelligence based on
             compressed modelling, recursive prediction and problem complexity},
  author  = {Hern{\'a}ndez-Espinosa, Alberto and Ozelim, Luan and
             Abrah{\~a}o, Felipe S. and Zenil, Hector},
  journal = {Nature Communications},
  year    = {2026},
  doi     = {10.1038/s41467-026-73289-5},
  url     = {https://www.nature.com/articles/s41467-026-73289-5}
}
```

The preprint that preceded it: [arXiv:2503.16743](https://arxiv.org/abs/2503.16743).
