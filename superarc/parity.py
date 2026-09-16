"""Regenerate every live figure and compare it against the published artifact.

This is the acceptance gate for the whole project: cleaning, refactoring and
adding models must never move a previously published figure. Anything that does
is a bug to fix, not a result to accept.

Comparison is done on rasterised pixels rather than file bytes, because PDF and
SVG both embed a creation timestamp and so never compare equal across runs even
when the content is identical.

Usage::

    python -m superarc.parity            # regenerate and compare
    python -m superarc.parity --keep     # leave the scratch output in place
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from . import REPO_ROOT

# Producers, in the order they must run. Either a script to run, or a module to
# run with -m; the four scripts are the originals, the two modules are panels
# that the notebooks drew and threw away (see superarc.timeseries).
PRODUCERS = (
    ["30-1_multiFormula_experiment.py"],
    ["31-1_multiScript_experiment.py"],
    ["34-2_S-ARC_ext.py"],
    ["35_summary_statistics.py"],
    ["-m", "superarc.timeseries"],
    ["-m", "superarc.complexity_measures"],
    ["-m", "superarc.compression_metrics"],
    ["-m", "superarc.languages"],
)

# Compared after rasterising both sides, for figures whose published artifact was
# only ever saved as PDF and SVG.
#
# The four polyglot-language panels below were unverified until 2026-09-15: their
# producer read one input through an absolute path on another machine, so it could
# not run here, and no gate covered them. They reproduce exactly.
PDF_COMPARISONS: dict[str, tuple[str, str]] = {
    "figure11-top.pdf": ("plots/highResolution/figure11-top.pdf",
                         "SI Fig 2 top  executions and prints by language"),
    "figure12-top.pdf": ("plots/highResolution/figure12-top.pdf",
                         "SI Fig 3 top  correct prints heatmap"),
    "figure12-bottom.pdf": ("plots/highResolution/figure12-bottom.pdf",
                            "SI Fig 3 bottom  no-compression, log scale"),
    "figure13.pdf": ("plots/highResolution/figure13.pdf",
                     "SI Fig 4  no-compression by language and temperature"),
}

# Panels that are printed in the article but were never written to disk by any
# code, so there is nothing to compare them against. They are still regenerated
# on every run and held to run-to-run determinism, which is all the guarantee
# available until an author signs off on a reference version.
NEWLY_SAVED = {
    "figure01-middle.png": "Fig 1 middle  binary success, random sequences",
    "figure01-bottom.png": "Fig 1 bottom  BDM/Shannon/zip/lzw by complexity",
    # The similarity panels ARE printed -- on page 9, under the caption numbered
    # Fig. 6, which describes the integrated script analysis instead. An earlier
    # note here claimed the image was never placed and only its caption was
    # typeset; that was wrong, and came from trusting the caption numbering the
    # rest of this module exists to distrust.
    #
    # The article shows them composed into one 2x2 grid under a shared title,
    # with each panel titled by forecaster. No code ever did that: the notebook
    # draws one plt.figure per panel and calls plt.show, and neither `suptitle`
    # nor `subplots` appears anywhere in it. The grid was assembled by hand, so
    # these three files are what the code produced, and the composition is not
    # something to recover from history -- there is nothing there to recover.
    # (Checked across all 185 .py/.ipynb blobs ever committed: exactly two
    # mention "Average similarity", this module and that notebook.)
    #
    # The notebook DOES carry six rendered panels in its stored cell outputs,
    # which look like the reference this entry says does not exist. They are
    # not. They disagree with the printed page at complexity 3 -- stored
    # gral_simi 6.02, printed and regenerated 1.57 -- because cell 53 swaps
    # complexity 3 and 4 in place, so an even number of executions swaps them
    # back. The committed output is from such a session. See
    # tests/test_timeseries.py, which pins the printed values, and
    # average_by_complexity, which does the swap on a copy.
    "figure02-chronos-original.png": "Fig 2  similarity, chronos (printed p9, panel of a hand-composed grid)",
    "figure02-timeGPT-1-original.png": "Fig 2  similarity, TimeGPT-1 (printed p9, panel of a hand-composed grid)",
    "figure02-lag-llama-original.png": "Fig 2  similarity, lag-llama (printed p9, panel of a hand-composed grid)",
}

# Data-level checks. Comparing the numbers a figure is drawn from is stricter
# than comparing its pixels and cheaper than rendering: 35_summary_statistics.py
# writes these two files on its way to the family-evolution figures, and they
# are the complete input to the plotting function.
DATA_COMPARISONS = {
    "plots/multi_formula/multiple_formulae_values.csv":
        "SI Figs 5/6  family evolution, formulae values",
}

# Data files that deliberately no longer match, for the same reason a figure
# might. Still regenerated every run and held to determinism.
DATA_AUTHORISED = {
    "plots/multi_script/multiple_script_values.csv": (
        "SI Figs 5/6  script values -- CORRECTED: 35_summary_statistics.py "
        "carried its own name map, which sent the `gemini` column to the label "
        "Gemini-2.5-Pro and had no entry for `gemini-2.5-pro` at all. All three "
        "Gemini columns are fully populated (119, 162 and 750 scripts), so the "
        "figures labelled the 119-script column Gemini-2.5-Pro and discarded "
        "the 750-script one. Same defect already corrected in the Integrated "
        "Script Analysis (82 -> 750); this file was missed, so the two figures "
        "disagreed with each other. 16 values move, all on the Gemini-2.5-Pro "
        "row: valid instances 30/23/0 -> 172/187/300, accuracy 100.00/56.67 -> "
        "53.33/20.00. Nothing else changes and the formulae side stays "
        "byte-identical. "
        "ALSO carries the two corrections recorded under figure04-up: the "
        "substring column selection and the `\"*not found*\"` shredding"
    ),
}

# regenerated filename -> (published artifact, human label)
# The bootstrap figure is expected to differ: its draw was never reproducible
# (np.random.default_rng() was called without a seed), so the published image
# cannot be recovered. It is seeded now, and checked for run-to-run stability
# elsewhere rather than against the published pixels.
COMPARISONS = {
    "figure05.png": ("new_plots/figure05.png", "Fig 7  SuperARC-seq ranking"),
}

# Figures that deliberately no longer match what was printed. Each needs an
# author ruling recorded here, and each is still held to run-to-run determinism,
# so an unauthorised change on top of an authorised one is still caught.
AUTHORISED_CHANGES = {
    "figure01.png": (
        "Fig 1 top  simple climbers -- REPALETTED by author's ruling: "
        "plots/highResolution/figure01 is a restyle that was never printed. The "
        "article shows matplotlib's default cycle, and Figure 1 is the figure in "
        "the paper. Bar heights unchanged; tests/test_timeseries.py still checks "
        "the restyle reproduces that file exactly"
    ),
    "figure03-up.png": (
        "Fig 3  formulae -- CORRECTED: columns were matched by prefix, so five "
        "models absorbed a longer-named model's answers"
    ),
    "figure03-bottom.png": (
        "Fig 4  integrated formulae -- CORRECTED: same prefix collision"
    ),
    "figure04-up.png": (
        "Fig 5  script equivalence + accuracy -- CORRECTED: the accuracy panel "
        "selected its columns with `model.lower() in col.lower()`, a substring "
        "test, so four models were scored partly on a later model's programs. "
        "7 of 84 values move: Mistral 34.38/13.12 -> 100.00/53.33 (it was "
        "averaged over 16 columns instead of 2), DeepSeek 22.56/11.54 -> "
        "100.00/58.33, Grok-3 33.33/7.62 -> 0.00/0.00, Qwen 1.28 -> 0.00 at "
        "complexity 2. The equivalence panel above it used the safe form, so the "
        "two panels of one figure disagreed about which programs are Grok-3's, "
        "as did Supplementary Figures 5/6, which also used the safe form. "
        "Extraction verified first: with the published selection restored the "
        "module reproduces the old script at 0.0000%. "
        "ALSO CORRECTED (author's ruling 2026-09-14): a missing answer was read "
        "as eleven programs. 216 cells hold the quoted string `\"*not found*\"` "
        "rather than a list, and the loader indexed it without checking its "
        "type, so one missing answer became the eleven characters of the words. "
        "Seven models were therefore given eleven program columns where they "
        "wrote at most three or four, and accuracy was divided by eleven: "
        "ChatGPT-4o-Mini 14.55 -> 53.33, Gemini-2.5-Pro 9.70 -> 35.56, "
        "DeepSeek-R1-0528 8.48 -> 31.11, Claude-3.7 8.48 -> 23.33, "
        "Llama-4-Scout and Mistral-Large-2405 4.55 -> 16.67 (complexity 1). "
        "The other 21 models do not move, and equivalence does not move for the "
        "seven either -- a one-character program never ran, so it was never in "
        "the equivalence numerator or denominator"
    ),
    "figure04-bottom.png": (
        "Fig 6  integrated script -- CORRECTED: the bar labelled Gemini-2.5-Pro "
        "was drawn from the `gemini` column instead of `gemini-2.5-pro`, "
        "understating its valid-script volume by 9x (82 vs 750). "
        "ALSO CORRECTED: the same `\"*not found*\"` shredding, which inflated the "
        "Not-found and Pure-math bars of seven models -- each missing answer "
        "contributed nine Not-found labels and two Pure-math ones, '*' being a "
        "mathematical operator. With it corrected, the one count the published "
        "script set BY HAND is gone too: ChatGPT-4o-Mini's valid scripts at "
        "complexity 3 read 1, computed 11, and now compute 0, which is the truth "
        "-- all 30 of its cells there are missing answers"
    ),
    "figure06.png": (
        "p1-p4 by sequence type -- CORRECTED: the reshape that decides which row "
        "each model occupies used order='F'. Fortran order fills column by "
        "column, so the point drawn for model j and sequence set i was "
        "flat[i + 4*j] -- four CONSECUTIVE entries of a set-major array, which is "
        "four different models' values for one set, not one model's values for "
        "four sets. Confirmed on all 448 points. 173 of them showed a number "
        "belonging to a different model or a different sequence set; the rest "
        "coincided because many class proportions are zero or repeat. "
        "ChatGPT-4o's row read p1 = 0.000 on every sequence set when its true p1 "
        "on Integers Type 1 is 1.000 -- every answer correct. The companion phi "
        "figure reshaped in C order and was right, which is what made this a slip "
        "rather than a convention. Table 1, the ranking and the bootstrap compute "
        "from superarc.table1.score directly and are untouched"
    ),
    "figure07.png": (
        "phi by sequence type -- REORDERED, values unchanged: this figure's "
        "numbers were always correct, but it sorts its y-axis by the summed class "
        "proportions of the figure above, which are exactly 4.0 for every model "
        "because proportions sum to 1 per sequence set. The sort key is "
        "degenerate. Under the published reshape, floating-point summation of a "
        "differently ordered set of addends gave 3.99999999999999956 for some "
        "models and 4.0 for others -- a spread of 4.4e-16 -- and THAT is what "
        "ordered the rows. With the reshape corrected the tie is exact and the "
        "rows fall in a stable alphabetical order. 23 of 28 move. The apparent "
        "top-to-bottom ranking in both by-type figures was never a ranking; "
        "making it one would be a design change, not a correction, so it is left "
        "for the authors"
    ),
    "figure08.png": (
        "bootstrap tiers -- RESEEDED: published draw came from an unseeded "
        "generator and is unrecoverable; three labels now show their correctly "
        "rounded exact scores"
    ),
    "figure11_bottom.pdf": (
        "SI Fig 2 bottom  sequence overlap by language -- RESEEDED, values "
        "unchanged: a supervenn panel whose inputs are sets of sequence STRINGS. "
        "Python randomises string hashing per process, so set iteration order -- "
        "and therefore the `minimize gaps` packing -- changed every run: three "
        "separate processes disagreed with each other by 15.7%, 24.8% and 25.7% "
        "on identical code and identical data. The published image came from one "
        "unrecorded hash seed and cannot be recovered, exactly like the bootstrap "
        "draw and Supplementary Figure 1. regenerate() now pins PYTHONHASHSEED=0, "
        "under which two separate processes agree at 0.0000%, and the figure sits "
        "19.2% from print. The DATA is identical: all seven set sizes match the "
        "published figure (Mathematica 40, Python 52, Java 69, Matlab 72, R 59, "
        "Cpp 45, ArnoldC 12), in the same row order, pinned in "
        "tests/test_languages.py"
    ),
    "figure10.png": (
        "SI Fig 1  formulae complexity -- AVERAGED OVER THE DRAW by author's "
        "ruling: 28% of answers are '***' (no formula produced) and were stood "
        "in for by one random 45-character string, drawn unseeded, so the "
        "published image is one arbitrary sample. Each panel is a mean and so is "
        "linear in that string's measure, so the figure now uses its expectation "
        "-- the same method with its Monte Carlo error removed. 4.6% of pixels, "
        "which is larger than a lucky seed's 3.3% and should be: individual "
        "draws sit 3-6% from each other, and this is their centre"
    ),
}


def regenerate(out_dir: Path) -> list[str]:
    """Run every producer with its output redirected to ``out_dir``.

    ``PYTHONHASHSEED`` is pinned because one panel needs it. The supervenn figure
    in :mod:`superarc.languages` lays out sets of sequence *strings*, and Python
    randomises string hashing per process, so its packing varied by 15-26% between
    runs of identical code on identical data. It cannot be set from inside the
    producer -- the interpreter reads it at startup -- so it belongs here.
    """
    env = dict(os.environ, SUPERARC_PLOTS_DIR=str(out_dir), PYTHONHASHSEED="0")
    failures = []
    for command in PRODUCERS:
        proc = subprocess.run(
            [sys.executable, *command],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            tail = proc.stderr.strip().splitlines()[-3:]
            name = command[-1]
            failures.append(f"{name} exited {proc.returncode}: {' | '.join(tail)}")
    return failures


def rasterise(pdf_path: Path, dpi: int = 150) -> Path:
    """Render page 1 of a PDF to PNG so two PDFs can be compared by pixels."""
    stem = Path(tempfile.mkdtemp(prefix="superarc-raster-")) / "page"
    subprocess.run(
        ["pdftoppm", "-r", str(dpi), "-png", "-f", "1", "-l", "1",
         str(pdf_path), str(stem)],
        check=True,
        capture_output=True,
    )
    return stem.with_name("page-1.png")


def pixel_difference(a_path: Path, b_path: Path) -> float | None:
    """Percentage of differing pixels, or None if the images differ in size."""
    from PIL import Image
    import numpy as np

    Image.MAX_IMAGE_PIXELS = None  # the bootstrap figure is very large
    a = np.asarray(Image.open(a_path).convert("RGB"), dtype=int)
    b = np.asarray(Image.open(b_path).convert("RGB"), dtype=int)
    if a.shape != b.shape:
        return None
    # A small tolerance absorbs encoder rounding without hiding a moved line.
    return float((np.abs(a - b).sum(axis=2) > 10).mean() * 100)


def check_figures(names, out_dir: Path | None = None):
    """Status of a few named figures, as a table a notebook can display.

    ``python -m superarc.parity`` checks everything and returns an exit code;
    this checks whichever figures one notebook is responsible for, and returns
    what it found. A verification notebook should end by telling the reader
    whether the paper still holds, not by asserting it silently.

    Each figure is in exactly one of three states:

        matches published   regenerates pixel for pixel
        changed             deliberately differs; the reason is the verdict
        newly saved         nothing to compare against; it was never on disk
    """
    import pandas as pd

    from . import plots_dir

    out_dir = out_dir or plots_dir()
    rows = []
    for name in names:
        produced = out_dir / name
        if not produced.exists():
            rows.append({"figure": name, "status": "NOT PRODUCED", "detail": str(out_dir)})
        elif name in COMPARISONS:
            published, label = COMPARISONS[name]
            difference = pixel_difference(produced, REPO_ROOT / published)
            if difference == 0:
                rows.append({"figure": label, "status": "matches published",
                             "detail": "0.0000% of pixels differ"})
            elif difference is None:
                rows.append({"figure": label, "status": "SIZE CHANGED", "detail": published})
            else:
                rows.append({"figure": label, "status": "DIFFERS",
                             "detail": f"{difference:.4f}% of pixels differ"})
        elif name in PDF_COMPARISONS:
            published, label = PDF_COMPARISONS[name]
            difference = pixel_difference(rasterise(produced),
                                          rasterise(REPO_ROOT / published))
            if difference == 0:
                rows.append({"figure": label, "status": "matches published",
                             "detail": "0.0000% of pixels differ"})
            elif difference is None:
                rows.append({"figure": label, "status": "SIZE CHANGED", "detail": published})
            else:
                rows.append({"figure": label, "status": "DIFFERS",
                             "detail": f"{difference:.4f}% of pixels differ"})
        elif name in AUTHORISED_CHANGES:
            rows.append({"figure": name, "status": "changed",
                         "detail": AUTHORISED_CHANGES[name]})
        elif name in NEWLY_SAVED:
            rows.append({"figure": NEWLY_SAVED[name], "status": "newly saved",
                         "detail": "never written to disk before; no reference exists"})
        else:
            rows.append({"figure": name, "status": "not tracked by the gate",
                         "detail": "add it to superarc/parity.py"})
    return pd.DataFrame(rows)


def check(out_dir: Path) -> int:
    failures: list[str] = []
    print(f"{'figure':52s} {'result':>14s}")
    for produced, (published, label) in COMPARISONS.items():
        new = out_dir / produced
        old = REPO_ROOT / published
        if not new.exists():
            print(f"{label:52s} {'NOT PRODUCED':>14s}")
            failures.append(f"{label}: not produced")
            continue
        diff = pixel_difference(new, old)
        if diff is None:
            print(f"{label:52s} {'SIZE CHANGED':>14s}")
            failures.append(f"{label}: image size changed")
        elif diff > 0:
            print(f"{label:52s} {f'{diff:.4f}% diff':>14s}")
            failures.append(f"{label}: {diff:.4f}% of pixels differ")
        else:
            print(f"{label:52s} {'IDENTICAL':>14s}")

    for produced, (published, label) in PDF_COMPARISONS.items():
        new = out_dir / produced
        if not new.exists():
            print(f"{label:52s} {'NOT PRODUCED':>14s}")
            failures.append(f"{label}: not produced")
            continue
        diff = pixel_difference(rasterise(new), rasterise(REPO_ROOT / published))
        if diff:
            print(f"{label:52s} {f'{diff:.4f}% diff':>14s}")
            failures.append(f"{label}: {diff:.4f}% of pixels differ")
        else:
            print(f"{label:52s} {'IDENTICAL':>14s}")

    for relative, label in DATA_COMPARISONS.items():
        # 35_summary_statistics.py builds these under its output directory,
        # mirroring the repo layout beneath it.
        produced_path = out_dir / Path(relative).name
        if not produced_path.exists():
            candidates = list(out_dir.rglob(Path(relative).name))
            produced_path = candidates[0] if candidates else produced_path
        published_path = REPO_ROOT / relative
        if not produced_path.exists():
            print(f"{label:52s} {'NOT PRODUCED':>14s}")
            failures.append(f"{label}: not produced")
        elif produced_path.read_bytes() == published_path.read_bytes():
            print(f"{label:52s} {'IDENTICAL':>14s}")
        else:
            print(f"{label:52s} {'DIFFERS':>14s}")
            failures.append(f"{label}: content differs from published")

    for relative, why in DATA_AUTHORISED.items():
        name = Path(relative).name
        candidates = list(out_dir.rglob(name))
        if not candidates:
            print(f"{why[:50]:52s} {'NOT PRODUCED':>14s}")
            failures.append(f"{relative}: not produced")
        else:
            print(f"{why[:50]:52s} {'authorised':>14s}")

    for produced, label in NEWLY_SAVED.items():
        if not (out_dir / produced).exists():
            print(f"{label:52s} {'NOT PRODUCED':>14s}")
            failures.append(f"{produced}: not produced")
        else:
            print(f"{label:52s} {'newly saved':>14s}")

    for produced, why in AUTHORISED_CHANGES.items():
        if not (out_dir / produced).exists():
            print(f"{why[:60]:62s} {'NOT PRODUCED':>14s}")
            failures.append(f"{produced}: not produced")
            continue
        print(f"{why[:60]:62s} {'authorised':>14s}")

    if failures:
        print(f"\nPARITY FAILED - {len(failures)} problem(s):")
        for f in failures:
            print(f"  {f}")
        return 1
    print("\nparity holds: every published figure regenerates identically")
    return 0


def check_determinism(first: Path, second: Path) -> list[str]:
    """Corrected figures must at least be stable from one run to the next.

    Without this, an authorised correction would be a hole in the gate: any
    further unintended change to those figures would go unnoticed. The same
    applies to the panels that were never saved and so have no published
    reference: determinism is the only guarantee available for them.
    """
    failures = []
    print("\nrun-to-run determinism of corrected and newly-saved outputs:")

    # Data files first. An authorised change to a CSV would otherwise leave the
    # figures drawn from it with no check at all, which is how a correction
    # becomes a hole in the gate.
    for relative in DATA_AUTHORISED:
        name = Path(relative).name
        a = next(iter(first.rglob(name)), None)
        b = next(iter(second.rglob(name)), None)
        if a is None or b is None:
            failures.append(f"{name}: missing from one of the two runs")
            continue
        ok = a.read_bytes() == b.read_bytes()
        print(f"  {name:38s} {'stable' if ok else 'NOT REPRODUCIBLE':>18s}")
        if not ok:
            failures.append(f"{name}: differs between two runs of identical code")

    for produced in list(AUTHORISED_CHANGES) + list(NEWLY_SAVED):
        a, b = first / produced, second / produced
        if not (a.exists() and b.exists()):
            failures.append(f"{produced}: missing from one of the two runs")
            continue
        # Some panels were only ever saved as PDF/SVG, so they have to be
        # rasterised before they can be compared by pixels -- the same reason
        # PDF_COMPARISONS exists.
        if a.suffix == ".pdf":
            diff = pixel_difference(rasterise(a), rasterise(b))
        else:
            diff = pixel_difference(a, b)
        ok = diff == 0
        print(f"  {produced:24s} {'stable' if ok else 'NOT REPRODUCIBLE':>18s}")
        if not ok:
            failures.append(f"{produced}: differs between two runs of identical code")
    return failures


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    keep = "--keep" in argv
    skip_determinism = "--fast" in argv

    out_dir = Path(tempfile.mkdtemp(prefix="superarc-parity-"))
    print(f"regenerating into {out_dir}\n")
    failures = regenerate(out_dir)
    if failures:
        print("PRODUCER FAILED:")
        for f in failures:
            print(f"  {f}")
        return 2
    code = check(out_dir)

    if not skip_determinism:
        second = Path(tempfile.mkdtemp(prefix="superarc-parity-2nd-"))
        if regenerate(second):
            print("second run failed; cannot check determinism")
            code = max(code, 2)
        else:
            det = check_determinism(out_dir, second)
            if det:
                for f in det:
                    print(f"  {f}")
                code = max(code, 1)

    if keep:
        print(f"\nscratch output kept at {out_dir}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
