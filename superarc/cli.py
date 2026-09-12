"""One command that regenerates every figure and table of the paper.

    python -m superarc.cli --out outputs/v2-2026-09

This is the point of the refactor. Adding a model should mean appending its
answers to the datasets and running this once, rather than opening forty
notebooks and remembering which cell to execute in which order.

What it does, in order:

1. Runs every producer with its output redirected into ``--out``, so the tracked
   artifacts in ``plots/`` and ``new_plots/`` are never touched by a regeneration.
2. Emits Table 1, which no code ever wrote -- the script that computes it
   evaluates ``np.round(...)`` as a bare expression, a no-op outside a notebook,
   so the published table was transcribed from a notebook cell by hand.
3. Writes a manifest recording exactly what produced the run: the hash of every
   input dataset, the models found in each, the installed package versions, the
   Python version and the git commit. A figure without this is not reproducible,
   only re-drawable.

It does not check the result against the published figures. That is
``python -m superarc.parity``, deliberately a separate command: this one is for
producing a new version, that one is for proving the old one did not move.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import REPO_ROOT, __version__
from .parity import PRODUCERS

MANIFEST_NAME = "manifest.json"


def git_revision() -> dict[str, str]:
    def run(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
            ).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "unknown"

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        # A dirty tree means the run cannot be reproduced from the commit alone.
        "dirty": bool(run("status", "--porcelain")),
    }


def input_hashes() -> dict[str, str]:
    """SHA-256 of every dataset a producer reads."""
    from .baseline import DATASETS

    digests = {}
    for key, (name, _encoding) in DATASETS.items():
        path = REPO_ROOT / name
        if path.exists():
            digests[f"{key} ({name})"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digests


def models_present() -> dict[str, list[str]]:
    """Which models each dataset currently carries.

    Recorded per run because it is the one thing that changes when a model is
    added, and the one thing a reader of a regenerated figure needs to know.
    """
    from .registry import A_FORMULA, B_SCRIPT, C_SERIES, models_for

    return {
        dataset: [m.name for m in models_for(dataset)]
        for dataset in (A_FORMULA, B_SCRIPT, C_SERIES)
    }


def package_versions() -> dict[str, str]:
    """Versions of the packages whose behaviour the results depend on.

    ``setuptools`` is in this list for a reason: pybdm imports ``pkg_resources``
    at module load, which setuptools deprecated in 81 and removed in 84. On a
    version that removed it, the BDM computations fail inside joblib workers as a
    BrokenProcessPool, far from the cause.
    """
    from importlib.metadata import PackageNotFoundError, version

    names = (
        "numpy", "pandas", "matplotlib", "seaborn", "scipy", "pybdm",
        "altair", "setuptools", "sympy",
    )
    versions = {}
    for name in names:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def run_producers(out_dir: Path, quiet: bool = False) -> list[str]:
    env = dict(os.environ, SUPERARC_PLOTS_DIR=str(out_dir))
    failures = []
    for command in PRODUCERS:
        name = command[-1]
        if not quiet:
            print(f"  running {name}", flush=True)
        proc = subprocess.run(
            [sys.executable, *command],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            tail = " | ".join(proc.stderr.strip().splitlines()[-3:])
            failures.append(f"{name} exited {proc.returncode}: {tail}")
    return failures


def emit_table1(out_dir: Path) -> None:
    from .table1 import compute, to_latex

    table = compute()
    table.round(6).to_csv(out_dir / "ranking_table.csv", index=False)
    (out_dir / "ranking_table.tex").write_text(to_latex(table))


def write_manifest(out_dir: Path) -> Path:
    manifest = {
        "superarc_version": __version__,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_revision(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": package_versions(),
        "inputs": input_hashes(),
        "models": models_present(),
        "outputs": sorted(
            p.name for p in out_dir.iterdir() if p.name != MANIFEST_NAME
        ),
    }
    path = out_dir / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m superarc.cli",
        description="Regenerate every figure and table of the paper.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="output directory; created if absent. Use a new one per version.",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="only print the summary."
    )
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    out_dir = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.quiet:
        print(f"regenerating into {out_dir}")

    failures = run_producers(out_dir, quiet=args.quiet)
    if failures:
        print("\nFAILED:")
        for failure in failures:
            print(f"  {failure}")
        return 1

    emit_table1(out_dir)
    manifest = write_manifest(out_dir)

    figures = sorted(p.name for p in out_dir.glob("*.pdf"))
    print(f"\n{len(figures)} figures, Table 1 (csv + tex), and {manifest.name}")
    print(f"in {out_dir}")
    print("\nverify nothing published moved:  python -m superarc.parity")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
