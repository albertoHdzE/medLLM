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
)

# Compared after rasterising both sides, for figures whose published artifact was
# only ever saved as PDF and SVG.
PDF_COMPARISONS = {
    "figure01.pdf": (
        "plots/highResolution/figure01.pdf",
        "Fig 1 top  binary success, simple climbers",
    ),
}

# Panels that are printed in the article but were never written to disk by any
# code, so there is nothing to compare them against. They are still regenerated
# on every run and held to run-to-run determinism, which is all the guarantee
# available until an author signs off on a reference version.
NEWLY_SAVED = {
    "figure01-middle.png": "Fig 1 middle  binary success, random sequences",
    "figure01-bottom.png": "Fig 1 bottom  BDM/Shannon/zip/lzw by complexity",
    "figure02-chronos-original.png": "Fig 2  similarity, chronos (caption only in print)",
    "figure02-timeGPT-1-original.png": "Fig 2  similarity, TimeGPT-1 (caption only in print)",
    "figure02-lag-llama-original.png": "Fig 2  similarity, lag-llama (caption only in print)",
}

# Data-level checks. Comparing the numbers a figure is drawn from is stricter
# than comparing its pixels and cheaper than rendering: 35_summary_statistics.py
# writes these two files on its way to the family-evolution figures, and they
# are the complete input to the plotting function.
DATA_COMPARISONS = {
    "plots/multi_formula/multiple_formulae_values.csv":
        "SI Figs 5/6  family evolution, formulae values",
    "plots/multi_script/multiple_script_values.csv":
        "SI Figs 5/6  family evolution, script values",
}

# regenerated filename -> (published artifact, human label)
# The bootstrap figure is expected to differ: its draw was never reproducible
# (np.random.default_rng() was called without a seed), so the published image
# cannot be recovered. It is seeded now, and checked for run-to-run stability
# elsewhere rather than against the published pixels.
COMPARISONS = {
    "figure04-up.png": ("new_plots/figure04-up.png", "Fig 5  script equivalence + accuracy"),
    "figure05.png": ("new_plots/figure05.png", "Fig 7  SuperARC-seq ranking"),
    "figure06.png": ("new_plots/figure06.png", "p1-p4 by sequence type"),
    "figure07.png": ("new_plots/figure07.png", "phi by sequence type"),
}

# Figures that deliberately no longer match what was printed. Each needs an
# author ruling recorded here, and each is still held to run-to-run determinism,
# so an unauthorised change on top of an authorised one is still caught.
AUTHORISED_CHANGES = {
    "figure03-up.png": (
        "Fig 3  formulae -- CORRECTED: columns were matched by prefix, so five "
        "models absorbed a longer-named model's answers"
    ),
    "figure03-bottom.png": (
        "Fig 4  integrated formulae -- CORRECTED: same prefix collision"
    ),
    "figure04-bottom.png": (
        "Fig 6  integrated script -- CORRECTED: the bar labelled Gemini-2.5-Pro "
        "was drawn from the `gemini` column instead of `gemini-2.5-pro`, "
        "understating its valid-script volume by 9x (82 vs 750)"
    ),
    "figure08.png": (
        "bootstrap tiers -- RESEEDED: published draw came from an unseeded "
        "generator and is unrecoverable; three labels now show their correctly "
        "rounded exact scores"
    ),
}


def regenerate(out_dir: Path) -> list[str]:
    """Run every producer with its output redirected to ``out_dir``."""
    env = dict(os.environ, SUPERARC_PLOTS_DIR=str(out_dir))
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
    print("\nrun-to-run determinism of corrected and newly-saved figures:")
    for produced in list(AUTHORISED_CHANGES) + list(NEWLY_SAVED):
        a, b = first / produced, second / produced
        if not (a.exists() and b.exists()):
            failures.append(f"{produced}: missing from one of the two runs")
            continue
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
