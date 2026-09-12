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

# Producer scripts, in the order they must run.
PRODUCERS = (
    "30-1_multiFormula_experiment.py",
    "31-1_multiScript_experiment.py",
    "34-2_S-ARC_ext.py",
)

# regenerated filename -> (published artifact, human label)
# The bootstrap figure is expected to differ: its draw was never reproducible
# (np.random.default_rng() was called without a seed), so the published image
# cannot be recovered. It is seeded now, and checked for run-to-run stability
# elsewhere rather than against the published pixels.
COMPARISONS = {
    "figure03-up.png": ("new_plots/figure03-up.png", "Fig 3  formulae equivalence + accuracy"),
    "figure03-bottom.png": ("new_plots/figure03-bottom.png", "Fig 4  integrated formulae"),
    "figure04-up.png": ("new_plots/figure04-up.png", "Fig 5  script equivalence + accuracy"),
    "figure04-bottom.png": ("new_plots/figure04-bottom.png", "Fig 6  integrated script"),
    "figure05.png": ("new_plots/figure05.png", "Fig 7  SuperARC-seq ranking"),
    "figure06.png": ("new_plots/figure06.png", "p1-p4 by sequence type"),
    "figure07.png": ("new_plots/figure07.png", "phi by sequence type"),
}

EXPECTED_TO_DIFFER = {
    "figure08.png": "bootstrap tiers - reseeded, published draw unrecoverable",
}


def regenerate(out_dir: Path) -> list[str]:
    """Run every producer with its output redirected to ``out_dir``."""
    env = dict(os.environ, SUPERARC_PLOTS_DIR=str(out_dir))
    failures = []
    for script in PRODUCERS:
        proc = subprocess.run(
            [sys.executable, script],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            tail = proc.stderr.strip().splitlines()[-3:]
            failures.append(f"{script} exited {proc.returncode}: {' | '.join(tail)}")
    return failures


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
    print(f"{'figure':42s} {'result':>14s}")
    for produced, (published, label) in COMPARISONS.items():
        new = out_dir / produced
        old = REPO_ROOT / published
        if not new.exists():
            print(f"{label:42s} {'NOT PRODUCED':>14s}")
            failures.append(f"{label}: not produced")
            continue
        diff = pixel_difference(new, old)
        if diff is None:
            print(f"{label:42s} {'SIZE CHANGED':>14s}")
            failures.append(f"{label}: image size changed")
        elif diff > 0:
            print(f"{label:42s} {f'{diff:.4f}% diff':>14s}")
            failures.append(f"{label}: {diff:.4f}% of pixels differ")
        else:
            print(f"{label:42s} {'IDENTICAL':>14s}")

    for produced, why in EXPECTED_TO_DIFFER.items():
        state = "produced" if (out_dir / produced).exists() else "MISSING"
        print(f"{'(skipped) ' + why:42s} {state:>14s}")
        if state == "MISSING":
            failures.append(f"{produced}: not produced")

    if failures:
        print(f"\nPARITY FAILED - {len(failures)} problem(s):")
        for f in failures:
            print(f"  {f}")
        return 1
    print("\nparity holds: every published figure regenerates identically")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    keep = "--keep" in argv
    out_dir = Path(tempfile.mkdtemp(prefix="superarc-parity-"))
    print(f"regenerating into {out_dir}\n")
    failures = regenerate(out_dir)
    if failures:
        print("PRODUCER FAILED:")
        for f in failures:
            print(f"  {f}")
        return 2
    code = check(out_dir)
    if keep:
        print(f"\nscratch output kept at {out_dir}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
