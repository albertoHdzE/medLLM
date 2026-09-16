# Verification notebooks

One notebook per experiment, in the order the science runs rather than the order
the files accumulated. Each one regenerates the figures it owns and tells you
whether they still agree with the published paper.

| | notebook | produces |
|---|---|---|
| 01 | `the-test-sequences` | BDM / Shannon / zip / lzw — Figure 1, bottom panel |
| 02 | `time-series-forecasters` | Figure 1 top and middle, Figure 2 |
| 03 | `formulae-generation` | Equivalence + Accuracy, Integrated Formulae Analysis |
| 04 | `script-generation` | Script Equivalence + Accuracy, Integrated Script Analysis |
| 05 | `superarc-seq` | Table 1, ranking, p₁–p₄ and φ by sequence type, bootstrap |
| 06 | `model-evolution` | Supplementary Figures 5 and 6 — family evolution |
| 07 | `formulae-complexity` | Metrics Comparison ... for Formulae generation |
| 08 | `programming-languages` | Supplementary Figures 2, 3 and 4 — seven languages |

## What these are for

They are **verification**, not production. `python -m superarc.cli --out <dir>`
regenerates everything without opening Jupyter; these exist so that a reader can
follow the reasoning and see the numbers at each step before the figure appears.

Every notebook has the same five-beat shape:

1. what this measures, and why it is evidence for something
2. load the data, and show what is in it
3. **the intermediate table the figure is drawn from**
4. the figure
5. does it still match what was published?

Step 3 is the point. A figure you cannot trace back to numbers is a picture.

## Rules

**No logic lives in a notebook.** Everything is imported from `superarc`. The
cells load, display and plot; they do not compute. This is not tidiness — the
original notebooks held logic that silently diverged from the scripts beside
them, which is how a figure came to be drawn from numbers somebody had retyped
by hand.

**Nothing writes to the published figures.** Output goes to `outputs/latest/`,
which is git-ignored. The tracked artifacts in `plots/` and `new_plots/` are the
reference every regeneration is checked against, and a reference you can destroy
by accident is not a reference. Each notebook ends by confirming it left them
alone.

## Running them

They need the project virtualenv, not the system Python:

    ./venv/bin/python -m pip install -e . --no-deps
    ./venv/bin/python -m ipykernel install --user --name superarc \
        --display-name "SuperARC (venv)"

Then open a notebook and select the **SuperARC (venv)** kernel. The notebooks
declare it, so Jupyter and VS Code should pick it automatically.

The editable install is what lets `import superarc` work from this subdirectory.
Without it the first cell fails, and adding `sys.path` hacks to every notebook is
the wrong fix.

### Two kernels that look right and are not

**`.venv` is the wrong one.** There are two virtualenvs in this repository:

| directory | Python | `import superarc` |
|---|---|---|
| `venv/` | 3.13 | works — the project is installed editable here |
| `.venv/` | 3.14 | **fails** — a separate environment, the project was never installed into it |

Selecting `.venv` gives `ModuleNotFoundError: No module named 'superarc'` on the
first cell of every notebook. The name is one dot away from the right one and
editors offer it first, so this is easy to hit.

`medllm_py` is the other trap: it points at the pyenv interpreter, which is
missing `altair` and other packages the producers need.

**Use `SuperARC (venv)`.** If it is not offered, register it with the two
commands above. To check which interpreter a running kernel is actually on:

    import sys; print(sys.executable)

It must end in `medLLM/venv/bin/python`.

## Outputs are committed

Deliberately. These notebooks are the record that the process is correct, so
their results should be readable without running anything. The cost is noisier
diffs; the benefit is that a reviewer can open the file and see that it worked.
